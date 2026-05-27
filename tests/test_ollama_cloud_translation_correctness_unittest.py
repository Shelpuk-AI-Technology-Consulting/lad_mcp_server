from __future__ import annotations

import json
import unittest

from lad_mcp_server.ollama_cloud_client import (
    normalize_ollama_model_name,
    translate_messages_to_ollama,
    translate_ollama_response,
)


class TestOllamaCloudReferenceCompatibility(unittest.TestCase):
    def test_normalizes_ollama_cloud_prefix(self) -> None:
        self.assertEqual(normalize_ollama_model_name("ollama_cloud/gpt-oss:120b"), "gpt-oss:120b")

    def test_flattens_content_blocks_to_string(self) -> None:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "first"},
                    {"type": "text", "text": "second"},
                ],
            }
        ]
        translated = translate_messages_to_ollama(messages)
        self.assertEqual(translated[0]["content"], "first\nsecond")

    def test_flattens_tool_message_content_blocks(self) -> None:
        messages = [
            {
                "role": "tool",
                "name": "read_file",
                "content": [{"type": "text", "text": "tool output"}],
            }
        ]
        translated = translate_messages_to_ollama(messages)
        self.assertEqual(translated[0]["content"], "tool output")

    def test_converts_request_tool_call_arguments_to_dict(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": '{"path":"src/main.py"}'},
                    }
                ],
            }
        ]
        translated = translate_messages_to_ollama(messages)
        args = translated[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, {"path": "src/main.py"})

    def test_bad_request_tool_call_json_arguments_becomes_empty_dict(self) -> None:
        messages = [
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "read_file", "arguments": "not-json"},
                    }
                ],
            }
        ]
        translated = translate_messages_to_ollama(messages)
        args = translated[0]["tool_calls"][0]["function"]["arguments"]
        self.assertEqual(args, {})

    def test_preserves_thinking_field(self) -> None:
        messages = [{"role": "assistant", "content": "x", "thinking": "hidden reasoning"}]
        translated = translate_messages_to_ollama(messages)
        self.assertEqual(translated[0]["thinking"], "hidden reasoning")

    def test_response_tool_calls_include_index(self) -> None:
        result = translate_ollama_response({
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"function": {"name": "read_file", "arguments": {"path": "a.py"}}}
                ],
            },
            "done_reason": "tool_calls",
        })
        self.assertEqual(result.tool_calls[0]["index"], 0)
        self.assertEqual(json.loads(result.tool_calls[0]["function"]["arguments"]), {"path": "a.py"})


if __name__ == "__main__":
    unittest.main()
