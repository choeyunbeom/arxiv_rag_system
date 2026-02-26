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
        """Remove any leftover <think> tags from Qwen3 output."""
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generate a response from the LLM."""
        payload = {
            "model": self.model,
            "prompt": "/no_think\n\n" + prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 1024,
            },
        }

        if system:
            payload["system"] = system

        response = httpx.post(
            f"{self.base_url}/api/generate",
            json=payload,
            timeout=120.0,
        )
        response.raise_for_status()
        raw = response.json()["response"]
        return self._clean_response(raw)
