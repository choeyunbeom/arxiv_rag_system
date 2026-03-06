"""
LLM Client
- Calls Ollama API for text generation
- Handles Qwen3 thinking mode suppression
"""

import re

import httpx

from src.api.core.config import settings


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

    async def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generate a response from the LLM."""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }

        if system:
            payload["system"] = "/no_think\n\n" + system

        response = await self._http_client.post(
            f"{self.base_url}/api/generate",
            json=payload,
        )
        response.raise_for_status()
        raw = response.json()["response"]
        return self._clean_response(raw)
