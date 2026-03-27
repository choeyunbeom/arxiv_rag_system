"""
LLM Client
- Calls Ollama API for text generation
- Handles Qwen3 thinking mode suppression
- Supports both batch and streaming generation
"""

import json
import re
from collections.abc import AsyncIterator
from enum import Enum, auto

import httpx

from src.api.core.config import settings


class _ThinkState(Enum):
    """State machine for filtering <think> blocks from streaming tokens."""
    NORMAL = auto()
    MAYBE_OPEN = auto()   # accumulating chars that might be "<think>"
    INSIDE_THINK = auto()
    MAYBE_CLOSE = auto()  # accumulating chars that might be "</think>"


class LLMClient:
    def __init__(self, model: str = settings.LLM_MODEL):
        self.model = model
        self.base_url = f"http://{settings.OLLAMA_HOST}"
        self._http_client = httpx.AsyncClient(timeout=180.0)

    def __del__(self):
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._http_client.aclose())
            else:
                loop.run_until_complete(self._http_client.aclose())
        except Exception:
            pass

    def _clean_response(self, text: str) -> str:
        """Remove any <think> tags from Qwen3 output, including unclosed ones."""
        # Remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove unclosed <think> (model ran out of tokens mid-thinking)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    def _build_payload(self, prompt: str, system: str, temperature: float, *, stream: bool) -> dict:
        """Build the Ollama API request payload."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }
        if system:
            payload["system"] = "/no_think\n\n" + system
        return payload

    async def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generate a complete response from the LLM (non-streaming)."""
        payload = self._build_payload(prompt, system, temperature, stream=False)

        response = await self._http_client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        raw = response.json()["response"]
        return self._clean_response(raw)

    async def stream_generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> AsyncIterator[str]:
        """Stream tokens from the LLM, filtering out <think> blocks in real-time.

        Yields clean answer tokens only — thinking content is silently discarded.
        Uses a state machine to handle <think>...</think> tags that may arrive
        split across multiple streamed chunks.
        """
        payload = self._build_payload(prompt, system, temperature, stream=True)

        state = _ThinkState.NORMAL
        buf = ""  # buffer for partial tag matching
        OPEN_TAG = "<think>"
        CLOSE_TAG = "</think>"

        async with self._http_client.stream(
            "POST", f"{self.base_url}/api/generate", json=payload
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                token = chunk.get("response", "")
                if not token:
                    if chunk.get("done"):
                        # Flush any remaining buffer in NORMAL state
                        if state == _ThinkState.NORMAL and buf:
                            yield buf
                        break
                    continue

                for char in token:
                    if state == _ThinkState.NORMAL:
                        buf += char
                        if buf.endswith(OPEN_TAG):
                            # Remove the tag from buffer and emit what came before
                            before = buf[:-len(OPEN_TAG)]
                            if before:
                                yield before
                            buf = ""
                            state = _ThinkState.INSIDE_THINK
                        elif not OPEN_TAG.startswith(buf.split("\n")[-1] if "\n" in buf else buf):
                            # No chance of matching <think>, flush safe prefix
                            # Keep only chars that could be start of "<think>"
                            flush_to = len(buf)
                            for i in range(1, min(len(OPEN_TAG), len(buf)) + 1):
                                if OPEN_TAG[:i] == buf[-i:]:
                                    flush_to = len(buf) - i
                                    break
                            if flush_to > 0:
                                yield buf[:flush_to]
                                buf = buf[flush_to:]

                    elif state == _ThinkState.INSIDE_THINK:
                        buf += char
                        if buf.endswith(CLOSE_TAG):
                            buf = ""
                            state = _ThinkState.NORMAL
                        else:
                            # Trim buffer — only keep enough to detect </think>
                            if len(buf) > len(CLOSE_TAG):
                                buf = buf[-(len(CLOSE_TAG)):]

                # Check if generation is done
                if chunk.get("done"):
                    if state == _ThinkState.NORMAL and buf:
                        yield buf
                    break
