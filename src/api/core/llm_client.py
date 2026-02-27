"""
LLM Client
- Calls Ollama API for text generation
- Handles Qwen3 thinking mode suppression
"""

import os
import re
import httpx

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")


class LLMClient:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.base_url = f"http://{OLLAMA_HOST}"

    def _clean_response(self, text: str) -> str:
        """Remove any <think> tags from Qwen3 output, including unclosed ones."""
        # Remove complete <think>...</think> blocks
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        # Remove unclosed <think> (model ran out of tokens mid-thinking)
        text = re.sub(r"<think>.*", "", text, flags=re.DOTALL)
        return text.strip()

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generate a response from the LLM."""
        payload = {
            "model": self.model,
            "prompt": "/no_think\n\n" + prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 2048,
            },
        }

        if system:
            payload["system"] = "/no_think\n\n" + system

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=180.0,
        )
        response.raise_for_status()
        raw = response.json()["response"]
        return self._clean_response(raw)
