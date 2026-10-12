"""Remote Ollama adapter for Qwen-style semantic scoring models.

Only the serialized prompt is sent to the remote server. Dataset files,
scene-graph files, and local filesystem paths remain on the caller machine.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
import urllib.error
import urllib.request


RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RemoteOllamaSemanticClient:
    """Expose Ollama's ``/api/generate`` endpoint as ``complete(prompt)``."""

    base_url: str
    model_name: str
    max_new_tokens: int = 4096
    context_length: int = 16384
    timeout: float = 300.0
    keep_alive: str = "10m"
    retry_attempts: int = 3

    def __post_init__(self) -> None:
        normalized_url = self.base_url.strip().rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not self.model_name.strip():
            raise ValueError("model_name must be non-empty")
        if self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be >= 1")
        if self.context_length < 1:
            raise ValueError("context_length must be >= 1")
        if self.timeout <= 0:
            raise ValueError("timeout must be > 0")
        if self.retry_attempts < 1:
            raise ValueError("retry_attempts must be >= 1")
        object.__setattr__(self, "base_url", normalized_url)
        object.__setattr__(self, "model_name", self.model_name.strip())

    def complete(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": "json",
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "seed": 0,
                "num_predict": int(self.max_new_tokens),
                "num_ctx": int(self.context_length),
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                generated = response_payload.get("response")
                if not isinstance(generated, str) or not generated.strip():
                    raise ValueError(
                        "Ollama response did not contain a non-empty 'response' string"
                    )
                return generated.strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Ollama HTTP {exc.code} for model {self.model_name!r}: {detail[:1000]}"
                )
                if exc.code not in RETRYABLE_HTTP_STATUSES or attempt >= self.retry_attempts:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = RuntimeError(
                    f"Ollama request failed for model {self.model_name!r}: {exc}"
                )
                if attempt >= self.retry_attempts:
                    raise last_error from exc

            time.sleep(min(2 ** (attempt - 1), 5))

        raise last_error or RuntimeError("Ollama request failed without an error")


__all__ = ["RemoteOllamaSemanticClient"]
