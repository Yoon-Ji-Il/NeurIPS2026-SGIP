"""OpenAI Responses-API adapter for reasoning models.

Imports the openai SDK lazily so environments that don't need it (e.g. local
Qwen-only setups) can still import the rest of the teacher package.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


DEFAULT_TEACHER_MODEL = "gpt-5.4"
DEFAULT_REASONING_EFFORT = "high"


def load_env_file(path: Path | None) -> None:
    if path is None or not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#") or "=" not in text:
            continue
        key, value = text.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def extract_response_text(response: Any) -> str:
    output_text = getattr(response, "output_text", None)
    if output_text:
        return str(output_text)
    chunks: list[str] = []
    for item in getattr(response, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text = getattr(content, "text", None)
            if text:
                chunks.append(str(text))
    if not chunks:
        raise ValueError("Could not extract text from OpenAI response")
    return "".join(chunks)


class OpenAIReasoningAdapter:
    def __init__(
        self,
        *,
        model: str = DEFAULT_TEACHER_MODEL,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        env_file: Path | None = None,
        max_output_tokens: int = 4096,
        timeout: float = 120.0,
    ) -> None:
        load_env_file(env_file)
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "openai SDK not installed. `pip install openai`."
            ) from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is not set")
        self.client = OpenAI(timeout=timeout)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.max_output_tokens = max_output_tokens

    def complete(self, prompt: str) -> str:
        """Plain text completion — returned as raw string for semantic_scorer."""
        response = self.client.responses.create(
            model=self.model,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=self.max_output_tokens,
            store=False,
        )
        return extract_response_text(response)

    def complete_json(
        self,
        *,
        instructions: str,
        prompt: str,
        json_schema: dict[str, Any],
        schema_name: str = "teacher_output",
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        response = self.client.responses.create(
            model=self.model,
            instructions=instructions,
            input=prompt,
            reasoning={"effort": self.reasoning_effort},
            max_output_tokens=self.max_output_tokens,
            text={
                "format": {
                    "type": "json_schema",
                    "name": schema_name,
                    "strict": True,
                    "schema": json_schema,
                }
            },
            store=False,
        )
        text = extract_response_text(response)
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"OpenAI response was not valid JSON: {text[:500]}") from exc
        info = {
            "response_id": getattr(response, "id", ""),
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
        }
        return payload, info


__all__ = [
    "DEFAULT_REASONING_EFFORT",
    "DEFAULT_TEACHER_MODEL",
    "OpenAIReasoningAdapter",
    "extract_response_text",
    "load_env_file",
]
