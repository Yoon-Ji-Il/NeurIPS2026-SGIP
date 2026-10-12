"""Structured-output client for the OpenAI Responses API."""

from __future__ import annotations

from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
import json
import time
from typing import Any
import urllib.error
import urllib.request


RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}


def retry_delay_seconds(headers: Any, attempt: int) -> float:
    retry_after = headers.get("retry-after") if headers is not None else None
    if retry_after:
        try:
            return min(max(float(retry_after), 0.0), 60.0)
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(retry_after)
                delay = retry_at.timestamp() - time.time()
                return min(max(delay, 0.0), 60.0)
            except (TypeError, ValueError, OverflowError):
                pass
    return min(float(2 ** (attempt - 1)), 30.0)


@dataclass(frozen=True)
class OpenAIResponsesStructuredClient:
    """Call ``/v1/responses`` without storing or printing the API key."""

    api_key: str = field(repr=False)
    model_name: str = "gpt-5.6-terra"
    base_url: str = "https://api.openai.com/v1"
    max_output_tokens: int = 4096
    reasoning_effort: str = "none"
    timeout: float = 600.0
    retry_attempts: int = 5

    def __post_init__(self) -> None:
        api_key = self.api_key.strip()
        model_name = self.model_name.strip()
        base_url = self.base_url.strip().rstrip("/")
        if not api_key:
            raise ValueError("OpenAI API key must be non-empty")
        if not model_name:
            raise ValueError("model_name must be non-empty")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("base_url must start with http:// or https://")
        if self.max_output_tokens < 1:
            raise ValueError("max_output_tokens must be positive")
        if self.timeout <= 0 or self.retry_attempts < 1:
            raise ValueError("timeout and retry_attempts must be positive")
        if self.reasoning_effort not in {"none", "low", "medium", "high", "xhigh", "max"}:
            raise ValueError(f"unsupported reasoning_effort: {self.reasoning_effort!r}")
        object.__setattr__(self, "api_key", api_key)
        object.__setattr__(self, "model_name", model_name)
        object.__setattr__(self, "base_url", base_url)

    def complete_json(
        self,
        *,
        instructions: str,
        prompt: str,
        json_schema: dict[str, Any],
        schema_name: str,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.model_name,
            "instructions": instructions,
            "input": prompt,
            "max_output_tokens": int(self.max_output_tokens),
            "reasoning": {"effort": self.reasoning_effort},
            "store": False,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "schema": json_schema,
                    "strict": True,
                }
            },
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request = urllib.request.Request(
            f"{self.base_url}/responses",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "sif3d-object-selector/1.0",
            },
            method="POST",
        )

        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    response_payload = json.loads(response.read().decode("utf-8"))
                return self._extract_output_text(response_payload)
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                last_error = RuntimeError(
                    f"OpenAI HTTP {exc.code} for {self.model_name!r}: {detail[:1000]}"
                )
                if exc.code not in RETRYABLE_HTTP_STATUSES or attempt >= self.retry_attempts:
                    raise last_error from exc
                time.sleep(retry_delay_seconds(exc.headers, attempt))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_error = RuntimeError(
                    f"OpenAI request failed for {self.model_name!r}: {exc}"
                )
                if attempt >= self.retry_attempts:
                    raise last_error from exc
                time.sleep(retry_delay_seconds(None, attempt))

        raise last_error or RuntimeError("OpenAI request failed without an error")

    def _extract_output_text(self, payload: dict[str, Any]) -> str:
        status = payload.get("status")
        if status != "completed":
            error = payload.get("error") or payload.get("incomplete_details") or status
            raise RuntimeError(f"OpenAI response did not complete: {error}")

        texts: list[str] = []
        refusals: list[str] = []
        for output in payload.get("output", []):
            if not isinstance(output, dict) or output.get("type") != "message":
                continue
            for content in output.get("content", []):
                if not isinstance(content, dict):
                    continue
                if content.get("type") == "output_text":
                    text = content.get("text")
                    if isinstance(text, str) and text.strip():
                        texts.append(text.strip())
                elif content.get("type") == "refusal":
                    refusal = content.get("refusal")
                    if isinstance(refusal, str) and refusal.strip():
                        refusals.append(refusal.strip())
        if refusals:
            raise RuntimeError(f"OpenAI model refused the request: {' '.join(refusals)}")
        if not texts:
            raise ValueError("OpenAI response has no non-empty output_text")
        return "\n".join(texts)


__all__ = ["OpenAIResponsesStructuredClient"]
