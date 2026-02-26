"""
LLM Client
- Calls Ollama API for text generation
- Supports streaming and non-streaming responses
"""

import os
import httpx


OLLAMA_HOST = os.getenv("OLLAMA_HOST", "localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:4b")


class LLMClient:
    def __init__(self, model: str = LLM_MODEL):
        self.model = model
        self.base_url = f"http://{OLLAMA_HOST}"

    def generate(self, prompt: str, system: str = "", temperature: float = 0.3) -> str:
        """Generate a response from the LLM."""
        payload = {
            "model": self.model,
            "prompt": prompt + " /no_think",
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
        return response.json()["response"]
