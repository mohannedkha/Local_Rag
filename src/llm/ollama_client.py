"""Thin wrapper around the Ollama Python SDK for chat and streaming."""

from __future__ import annotations

from typing import Generator, Optional

import ollama


class OllamaClient:
    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        temperature: float = 0.1,
        context_window: int = 8192,
    ):
        self.model = model
        self.temperature = temperature
        self.context_window = context_window
        self._client = ollama.Client(host=base_url)

    def chat(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> str | Generator[str, None, None]:
        """Send a list of {role, content} messages. Returns full text or a generator."""
        options = {
            "temperature": self.temperature,
            "num_ctx": self.context_window,
        }

        if stream:
            return self._stream(messages, options)

        resp = self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
        )
        return resp["message"]["content"]

    def _stream(
        self, messages: list[dict], options: dict
    ) -> Generator[str, None, None]:
        for chunk in self._client.chat(
            model=self.model,
            messages=messages,
            options=options,
            stream=True,
        ):
            yield chunk["message"]["content"]

    def is_available(self) -> bool:
        """Check that Ollama is running and the model is pulled."""
        try:
            models = self._client.list()
            names = [m["name"].split(":")[0] for m in models.get("models", [])]
            return self.model.split(":")[0] in names
        except Exception:
            return False
