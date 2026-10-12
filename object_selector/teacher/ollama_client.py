"""Structured-output client for a remote Ollama semantic teacher."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any
import urllib.error
import urllib.request


RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


@dataclass(frozen=True)
class RemoteOllamaStructuredClient:
    """Expose only ``complete_json`` so the legacy scorer passes its schema."""

    base_url: str
    model_name: str
    max_new_tokens: int = 4096
    context_length: int = 16384
    timeout: float = 600.0
    keep_alive: str = "10m"
    retry_attempts: int = 3
    seed: int = 0

    def __post_init__(self) -> None:
        base_url = self.base_url.strip().rstrip("/")
        model_name = self.model_name.strip()
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if self.max_new_tokens < 1 or self.context_length < 1:
            raise ValueError("token and context limits must be positive")
        if self.timeout <= 0 or self.retry_attempts < 1:
            raise ValueError("timeout and retry_attempts must be positive")
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model_name", model_name)

    def complete_json(
        self,
        *,
        instructions: str,
        prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        del instructions, schema_name
        payload: dict[str, Any] = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": json_schema or "json",
            "keep_alive": self.keep_alive,
            "options": {
                "temperature": 0,
                "seed": int(self.seed),
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
                    raise ValueError("Ollama response has no non-empty response text")
                return generated.strip()
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"Ollama HTTP {exc.code} for {self.model_name!r}: {detail[:1000]}"
                )
                if exc.code not in RETRYABLE_HTTP_STATUSES or attempt >= self.retry_attempts:
                    raise last_error from exc
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = RuntimeError(
                    f"Ollama request failed for {self.model_name!r}: {exc}"
                )
                if attempt >= self.retry_attempts:
                    raise last_error from exc
            time.sleep(min(2 ** (attempt - 1), 5))

        raise last_error or RuntimeError("Ollama request failed without an error")


__all__ = ["RemoteOllamaStructuredClient"]
