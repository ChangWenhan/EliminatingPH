from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class ChatConfig:
    api_base: str
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    timeout: float = 120.0
    retries: int = 2
    retry_backoff: float = 2.0


class ChatClient:
    def __init__(self, config: ChatConfig):
        self.config = config

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float,
        top_p: float,
        max_tokens: int,
        seed: int | None = None,
    ) -> str:
        url = self.config.api_base.rstrip("/") + "/chat/completions"
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.config.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        if seed is not None:
            payload["seed"] = seed
        last_error: Exception | None = None
        for attempt in range(self.config.retries + 1):
            try:
                response = requests.post(
                    url,
                    headers=headers,
                    json=payload,
                    timeout=self.config.timeout,
                )
                response.raise_for_status()
                data = response.json()
                return str(data["choices"][0]["message"]["content"]).strip()
            except (requests.RequestException, KeyError, IndexError, TypeError) as exc:
                if isinstance(exc, requests.HTTPError) and exc.response is not None:
                    body = exc.response.text.strip().replace("\n", " ")[:500]
                    exc = RuntimeError(f"{exc}; response={body}")
                last_error = exc
                if attempt >= self.config.retries:
                    break
                time.sleep(self.config.retry_backoff * (attempt + 1))
        raise RuntimeError(f"Chat completion failed after retries: {last_error}")
