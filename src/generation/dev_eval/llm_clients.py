import os
import time
from typing import Literal, Optional
from dotenv import load_dotenv

load_dotenv()


BackendName = Literal["openai", "ollama", "mock"]


class LLMClient:
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OpenAICompatibleClient(LLMClient):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0,
        top_p: float = 0.95,
        max_tokens: Optional[int] = None,
        timeout_s: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL")
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        self.timeout_s = timeout_s
        self.max_retries = max_retries

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY is not set")
        if not self.base_url:
            raise ValueError("OPENAI_BASE_URL is not set")

        from openai import OpenAI

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout_s)

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                content = resp.choices[0].message.content
                return (content or "")
            except Exception as e:
                last_err = e
                sleep_s = min(2 ** attempt, 30) + 0.1 * attempt
                time.sleep(sleep_s)
        raise RuntimeError(f"OpenAI request failed after {self.max_retries} retries") from last_err


class OllamaClient(LLMClient):
    def __init__(
        self,
        model: str,
        host: Optional[str] = None,
        temperature: float = 0,
        top_p: float = 0.95,
        max_retries: int = 3,
    ):
        self.model = model
        self.host = host or os.getenv("OLLAMA_HOST") or "http://localhost:11434"
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries

        import ollama

        self._ollama = ollama

    def generate(self, prompt: str) -> str:
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                if self.host:
                    resp = self._ollama.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": self.temperature, "top_p": self.top_p},
                        host=self.host,
                    )
                else:
                    resp = self._ollama.chat(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        options={"temperature": self.temperature, "top_p": self.top_p},
                    )
                return (resp.get("message", {}) or {}).get("content", "")
            except Exception as e:
                last_err = e
                time.sleep(min(2 ** attempt, 10))
        raise RuntimeError(f"Ollama request failed after {self.max_retries} retries") from last_err


class MockClient(LLMClient):
    def generate(self, prompt: str) -> str:
        _ = prompt
        return "pass"


def make_client(
    backend: BackendName,
    model: str,
    temperature: float,
    top_p: float,
    max_tokens: Optional[int],
    timeout_s: float,
) -> LLMClient:
    if backend == "openai":
        return OpenAICompatibleClient(
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
        )
    if backend == "ollama":
        return OllamaClient(model=model, temperature=temperature, top_p=top_p)
    if backend == "mock":
        return MockClient()
    raise ValueError(f"Unknown backend: {backend}")

