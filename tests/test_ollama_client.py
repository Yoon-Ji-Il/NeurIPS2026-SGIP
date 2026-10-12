from __future__ import annotations

import json
from unittest import TestCase
from unittest.mock import patch

from object_selector_new.teacher.ollama_client import RemoteOllamaSemanticClient


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class RemoteOllamaSemanticClientTest(TestCase):
    def test_complete_uses_remote_nonthinking_json_generation(self) -> None:
        client = RemoteOllamaSemanticClient(
            base_url="http://gpu.example:11434/",
            model_name="qwen3:4b",
            max_new_tokens=1234,
            context_length=8192,
            keep_alive="5m",
        )

        with patch(
            "object_selector_new.teacher.ollama_client.urllib.request.urlopen",
            return_value=_FakeResponse({"response": '{"ok": true}'}),
        ) as mocked_urlopen:
            result = client.complete("scene graph prompt")

        self.assertEqual(result, '{"ok": true}')
        request = mocked_urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://gpu.example:11434/api/generate")
        payload = json.loads(request.data.decode("utf-8"))
        self.assertEqual(payload["model"], "qwen3:4b")
        self.assertEqual(payload["prompt"], "scene graph prompt")
        self.assertFalse(payload["stream"])
        self.assertFalse(payload["think"])
        self.assertEqual(payload["format"], "json")
        self.assertEqual(payload["keep_alive"], "5m")
        self.assertEqual(payload["options"]["temperature"], 0)
        self.assertEqual(payload["options"]["seed"], 0)
        self.assertEqual(payload["options"]["num_predict"], 1234)
        self.assertEqual(payload["options"]["num_ctx"], 8192)

    def test_rejects_invalid_configuration(self) -> None:
        with self.assertRaises(ValueError):
            RemoteOllamaSemanticClient(base_url="gpu:11434", model_name="qwen3:4b")
        with self.assertRaises(ValueError):
            RemoteOllamaSemanticClient(base_url="http://gpu:11434", model_name="")
        with self.assertRaises(ValueError):
            RemoteOllamaSemanticClient(
                base_url="http://gpu:11434", model_name="qwen3:4b", context_length=0
            )
