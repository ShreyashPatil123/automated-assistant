import logging
import json
import requests
import re
from typing import Optional

class LLMClient:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1):
        # We use llama3.2:3b as phi3 was corrupted in previous tests.
        self.model_name = "llama3.2:3b"
        self.ollama_url = "http://localhost:11434/api/chat"
        logging.info(f"LLMClient initialized using local Ollama Chat API (Model: {self.model_name})")

    def _post_ollama(self, payload: dict, timeout_seconds: int) -> requests.Response:
        try:
            return requests.post(self.ollama_url, json=payload, timeout=timeout_seconds)
        except requests.exceptions.ConnectionError as e:
            raise RuntimeError(
                "Failed to connect to Ollama at http://localhost:11434. "
                "Fix: install/start Ollama and run 'ollama serve', then ensure the model is available (e.g. 'ollama pull phi3')."
            ) from e
        except requests.exceptions.Timeout as e:
            raise RuntimeError(
                "Ollama request timed out. Fix: ensure Ollama is running and the model fits your machine; consider reducing context/max tokens."
            ) from e

    def generate_json(self, prompt: str, schema: dict = None, max_tokens: int = 1024, temperature: float = 0.1) -> dict:
        """
        Generates a JSON response from the LLM via Ollama Chat API.
        """
        try:
             payload = {
                 "model": self.model_name,
                 "messages": [{"role": "user", "content": prompt}],
                 "stream": False,
                 "format": "json",
                 "options": {
                     "temperature": temperature,
                     "num_predict": max_tokens,
                     "stop": ["</action>", "</plan>", "</intent>"]
                 }
             }
                 
             response = self._post_ollama(payload, timeout_seconds=120)
             response.raise_for_status()
             result = response.json()
             text = result.get("message", {}).get("content", "")
             
             # Extract json by taking the first { and the last }
             start = text.find('{')
             end = text.rfind('}')
             if start != -1 and end != -1 and start < end:
                 json_str = text[start:end+1]
                 json_str = re.sub(r',\s*\}', '}', json_str)
                 json_str = re.sub(r',\s*\]', ']', json_str)
                 try:
                     return json.loads(json_str)
                 except json.JSONDecodeError as e:
                     for i in range(end - 1, start, -1):
                         if text[i] == '}':
                             try:
                                 return json.loads(text[start:i+1])
                             except json.JSONDecodeError:
                                 continue
                     raise ValueError(f"Extracted json string was invalid: {e}\nRaw extracted string was:\n{json_str}")
             
             logging.error(f"RAW OLLAMA OUTPUT (No JSON found): {repr(text)}")
             raise ValueError("Could not extract JSON from LLM response.")
             
        except Exception as e:
             logging.error("LLM JSON generation failed: %s", e)
             raise
             
    def generate_text(self, prompt: str, max_tokens: int = 512, temperature: float = 0.3) -> str:
        """Generates raw text via Ollama Chat API."""
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
        res = self._post_ollama(payload, timeout_seconds=120)
        res.raise_for_status()
        return res.json().get("message", {}).get("content", "").strip()
