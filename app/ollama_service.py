from dataclasses import dataclass
import requests

@dataclass
class ServiceOllama:
    prompt: str
    url_ollama_generate_render: str = "https://ollama-llama3-96lb.onrender.com/api/generate"
    url_ollama_generate_local: str = ""
    model: str = "llama3"


    def __post_init__(self):
        self.response = self._response_ollama(prompt=self.prompt)


    def _generate_pyload(self, prompt: str) -> dict:
        return {
            "model": self.model,
            "prompt": prompt,
            "stream": False
        }


    def _response_ollama(self, prompt: str) -> str:
        pyload = self._generate_pyload(prompt=prompt)
        response = requests.post(self.url_ollama_generate_render, json=pyload)
        return response.json().get("response")