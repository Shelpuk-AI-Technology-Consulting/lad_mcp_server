"""The OpenAI-compatible client speaks plain OpenAI and leaks no ambient credential.

Issue #2 asks for LiteLLM and any other OpenAI-compatible gateway as a direct
provider. The delicate part is not the HTTP — it is that this client is configured
with a *user-supplied base URL*, so a credential picked up from the environment
would be sent somewhere the user did not intend.
"""

from __future__ import annotations

import json
import os
import unittest
from typing import Any
from unittest import mock

from lad_mcp_server.openai_compat_client import (
    OpenAiCompatClient,
    OpenAiCompatClientError,
    is_openai_compat_model,
    normalize_openai_compat_model_name,
)

_BASE_URL = "https://gateway.internal.example.com/v1"


class TestModelPrefixes(unittest.TestCase):
    """Both accepted prefixes route here; nothing else does."""

    def test_supported_prefixes_match_case_insensitively(self) -> None:
        """`openai_compat/` and `litellm/` both route, in any case."""
        for model in ("openai_compat/gpt-4o", "litellm/gpt-4o", "LiteLLM/gpt-4o", "OPENAI_COMPAT/gpt-4o"):
            with self.subTest(model=model):
                self.assertTrue(is_openai_compat_model(model))
                self.assertEqual(normalize_openai_compat_model_name(model), "gpt-4o")

    def test_other_providers_are_not_claimed(self) -> None:
        """A model belonging to another provider is left alone."""
        for model in ("deepseek/deepseek-v4", "z-ai/glm-5", "ollama/gemma4:31b", "openai/gpt-4o", "gpt-4o"):
            with self.subTest(model=model):
                self.assertFalse(is_openai_compat_model(model))

    def test_a_bare_prefix_does_not_normalise_to_an_empty_name(self) -> None:
        """Stripping must never yield "", which routing and budgeting could disagree on.

        `None` means "not routed here"; an empty string would be an ambiguous third
        state. Matches the convention in the other direct clients.
        """
        self.assertEqual(normalize_openai_compat_model_name("litellm/"), "litellm/")


class _FakeResponse:
    """Minimal stand-in for the object `urlopen` yields."""

    def __init__(self, payload: dict[str, Any]) -> None:
        """Store the payload to return.

        Args:
            payload: The JSON body the endpoint should appear to return.
        """
        self._payload = payload

    def read(self) -> bytes:
        """Return the encoded payload.

        Returns:
            UTF-8 JSON bytes.
        """
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        """Enter the context manager.

        Returns:
            This response.
        """
        return self

    def __exit__(self, *exc: object) -> bool:
        """Exit the context manager.

        Args:
            *exc: Exception details, ignored.

        Returns:
            ``False``, so exceptions propagate.
        """
        return False


def _capture_request(client: OpenAiCompatClient, messages: list[dict[str, Any]], **extra: Any) -> Any:
    """Run a stdlib-path call and return the urllib Request that was built.

    Args:
        client: The client under test.
        messages: Messages to send.
        **extra: Additional ``chat_completion`` arguments.

    Returns:
        The captured ``urllib.request.Request``.
    """
    import asyncio

    captured: list[Any] = []

    def _fake_urlopen(req: Any, timeout: int | None = None) -> _FakeResponse:
        captured.append(req)
        return _FakeResponse({"choices": [{"message": {"content": "ok", "tool_calls": None}}]})

    with mock.patch.object(client, "_get_client", return_value="stdlib"):
        with mock.patch("urllib.request.urlopen", _fake_urlopen):
            asyncio.run(
                client.chat_completion(
                    model="gpt-4o", messages=messages, timeout_seconds=5, max_output_tokens=100, **extra
                )
            )
    return captured[0]


def _capture_sdk_call(client: OpenAiCompatClient, messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    """Run an SDK-path call and return the kwargs handed to the SDK.

    The SDK path is the one production actually takes — ``openai`` is a hard
    dependency — so patching ``_get_client`` to ``"stdlib"`` everywhere would leave
    the shipping code path unexecuted.

    Args:
        client: The client under test.
        messages: Messages to send.
        **extra: Additional ``chat_completion`` arguments.

    Returns:
        The captured keyword arguments.
    """
    import asyncio

    captured: dict[str, Any] = {}

    async def _create(**kwargs: Any) -> Any:
        captured.update(kwargs)
        message = type("M", (), {"content": "ok", "tool_calls": None})()
        return type("R", (), {"choices": [type("C", (), {"message": message})()]})()

    fake = mock.Mock()
    fake.chat.completions.create = _create
    with mock.patch.object(client, "_get_client", return_value=fake):
        asyncio.run(
            client.chat_completion(
                model="gpt-4o", messages=messages, timeout_seconds=5, max_output_tokens=100, **extra
            )
        )
    return captured


class TestRequestShape(unittest.TestCase):
    """The wire format is plain OpenAI chat-completions."""

    def test_posts_chat_completions_under_the_configured_base_url(self) -> None:
        """The path is `<base>/chat/completions` and carries the OpenAI fields."""
        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="k", max_concurrent_requests=1)

        req = _capture_request(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(req.full_url, f"{_BASE_URL}/chat/completions")
        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["model"], "gpt-4o")
        self.assertEqual(body["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(body["max_tokens"], 100)

    def test_a_trailing_slash_on_the_base_url_does_not_double_up(self) -> None:
        """Users paste base URLs with and without a trailing slash."""
        client = OpenAiCompatClient(base_url=_BASE_URL + "/", api_key="k", max_concurrent_requests=1)

        req = _capture_request(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(req.full_url, f"{_BASE_URL}/chat/completions")

    def test_tools_and_extra_body_reach_the_request(self) -> None:
        """Tool calling is where the product's value lives, so it must survive the hop."""
        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="k", max_concurrent_requests=1)
        tools = [{"type": "function", "function": {"name": "read_file"}}]

        req = _capture_request(
            client,
            [{"role": "user", "content": "hi"}],
            tools=tools,
            tool_choice="auto",
            extra_body={"reasoning": {"effort": "high"}},
        )

        body = json.loads(req.data.decode("utf-8"))
        self.assertEqual(body["tools"], tools)
        self.assertEqual(body["tool_choice"], "auto")
        self.assertEqual(body["reasoning"], {"effort": "high"})

    def test_reasoning_content_is_stripped(self) -> None:
        """An arbitrary gateway may 4xx on the unknown field, and the SDK forwards it."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "reasoning_content": "hidden thoughts"},
        ]

        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="k", max_concurrent_requests=1)
        req = _capture_request(client, messages)

        body = json.loads(req.data.decode("utf-8"))
        self.assertNotIn("reasoning_content", json.dumps(body))
        # The original list must not be mutated — it is reused by the caller's loop.
        self.assertIn("reasoning_content", messages[1])


class TestSdkPath(unittest.TestCase):
    """The branch production actually takes, since `openai` is a hard dependency."""

    def test_the_openai_fields_are_forwarded(self) -> None:
        """FR2 names the SDK path specifically; it needs its own coverage."""
        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="k", max_concurrent_requests=1)
        tools = [{"type": "function", "function": {"name": "read_file"}}]

        captured = _capture_sdk_call(
            client, [{"role": "user", "content": "hi"}], tools=tools, tool_choice="auto"
        )

        self.assertEqual(captured["model"], "gpt-4o")
        self.assertEqual(captured["max_tokens"], 100)
        self.assertEqual(captured["tools"], tools)
        self.assertEqual(captured["tool_choice"], "auto")

    def test_reasoning_content_is_stripped_here_too(self) -> None:
        """FR3 says both paths, and the SDK forwards unknown message keys verbatim."""
        messages = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "x", "reasoning_content": "hidden thoughts"},
        ]

        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="k", max_concurrent_requests=1)
        captured = _capture_sdk_call(client, messages)

        self.assertNotIn("reasoning_content", json.dumps(captured["messages"]))
        self.assertIn("reasoning_content", messages[1], "the caller's list must not be mutated")

    def test_an_sdk_failure_becomes_this_clients_error(self) -> None:
        """Otherwise the failure escapes provider fallback wearing the wrong label."""
        import asyncio

        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="k", max_concurrent_requests=1)

        with mock.patch.object(client, "_get_client", side_effect=RuntimeError("SDK refused to build")):
            with self.assertRaises(OpenAiCompatClientError) as caught:
                asyncio.run(
                    client.chat_completion(
                        model="gpt-4o",
                        messages=[{"role": "user", "content": "hi"}],
                        timeout_seconds=5,
                        max_output_tokens=10,
                    )
                )

        self.assertIn("SDK refused to build", str(caught.exception))


class TestCredentialIsolation(unittest.TestCase):
    """No credential may ever come from the ambient environment.

    `AsyncOpenAI(api_key=None)` falls back to `OPENAI_API_KEY`, which is set in most
    Claude Code / Codex environments. Since this client's base URL is user-supplied,
    that would send an operator's personal OpenAI key to their corporate gateway.
    """

    def test_no_authorization_header_when_no_key_is_configured(self) -> None:
        """The stdlib path omits the header rather than sending `Bearer `."""
        client = OpenAiCompatClient(base_url=_BASE_URL, api_key=None, max_concurrent_requests=1)

        req = _capture_request(client, [{"role": "user", "content": "hi"}])

        self.assertIsNone(req.headers.get("Authorization"))
        self.assertNotIn("Bearer", json.dumps(dict(req.headers)))

    def test_authorization_header_uses_the_configured_key(self) -> None:
        """With a key, it is sent as a bearer token."""
        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="secret-token", max_concurrent_requests=1)

        req = _capture_request(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(req.headers.get("Authorization"), "Bearer secret-token")

    def test_a_keyless_client_never_constructs_an_sdk_client(self) -> None:
        """No value of `api_key` is safe across the `openai` versions we accept.

        Measured on `AsyncOpenAI(base_url=..., api_key=X)`: `None` sends the ambient
        `OPENAI_API_KEY` on every version, and `""` sent no header until 2.34.0 but
        raises `OpenAIError` from 2.34.0 onward. `pyproject.toml` bounds `openai` only
        from below, so a keyless gateway — explicitly supported — has to avoid the SDK
        altogether and take the stdlib path, which omits the header.
        """
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-operators-personal-key"}):
            client = OpenAiCompatClient(base_url=_BASE_URL, api_key=None, max_concurrent_requests=1)

            self.assertEqual(client._get_client(), "stdlib")

    def test_a_keyless_client_can_still_complete_a_call(self) -> None:
        """The regression that matters to the user: a keyless gateway must work.

        `_get_client` raising would surface as a message telling the user to set
        `OPENAI_API_KEY` — the exact wrong-provider advice this feature removes.
        """
        client = OpenAiCompatClient(base_url=_BASE_URL, api_key=None, max_concurrent_requests=1)

        req = _capture_request(client, [{"role": "user", "content": "hi"}])

        self.assertEqual(req.full_url, f"{_BASE_URL}/chat/completions")
        self.assertIsNone(req.headers.get("Authorization"))

    def test_sdk_client_sends_the_configured_key(self) -> None:
        """With a key configured, the SDK sends exactly that key, not the ambient one."""
        with mock.patch.dict("os.environ", {"OPENAI_API_KEY": "sk-operators-personal-key"}):
            client = OpenAiCompatClient(base_url=_BASE_URL, api_key="gateway-key", max_concurrent_requests=1)
            sdk = client._get_client()
            if sdk == "stdlib":  # pragma: no cover - only without the openai package
                self.skipTest("openai SDK not installed")

            self.assertEqual(sdk.auth_headers, {"Authorization": "Bearer gateway-key"})

    def test_no_ambient_openai_setting_reaches_the_gateway(self) -> None:
        """The key is not the only thing the SDK reads from the environment.

        Measured on 2.16 and 2.53: `OPENAI_ORG_ID` and `OPENAI_PROJECT_ID` become
        request headers, and 2.53 merges arbitrary ones from `OPENAI_CUSTOM_HEADERS`.
        All of it would go to a base URL the user supplied. Asserted on the built
        request rather than on constructor arguments, and as an allowlist, so a future
        release that adds a fourth ambient source fails here instead of leaking.
        """
        # Checked as values rather than as the variables' literal text: 2.53 splits
        # OPENAI_CUSTOM_HEADERS on ":" and lowercases the name, so asserting the raw
        # JSON is absent passes while the secret inside it still goes on the wire.
        secrets = (
            "sk-operators-personal-key",
            "org-operators-personal-org",
            "proj-operators-personal-project",
            "must-not-leak",
        )
        ambient = {
            "OPENAI_API_KEY": secrets[0],
            "OPENAI_ORG_ID": secrets[1],
            "OPENAI_PROJECT_ID": secrets[2],
            "OPENAI_CUSTOM_HEADERS": '{"X-Internal-Secret": "%s"}' % secrets[3],
            "OPENAI_BASE_URL": "https://api.openai.com/v1",
        }
        with mock.patch.dict("os.environ", ambient):
            client = OpenAiCompatClient(base_url=_BASE_URL, api_key="gateway-key", max_concurrent_requests=1)
            sdk = client._get_client()
            if sdk == "stdlib":  # pragma: no cover - only without the openai package
                self.skipTest("openai SDK not installed")

            import openai

            request = sdk._build_request(
                openai._models.FinalRequestOptions.construct(
                    method="post", url="/chat/completions", json_data={}
                )
            )
            sent = "\n".join(f"{name}: {value}" for name, value in request.headers.items()).lower()

        for secret in secrets:
            with self.subTest(leaked=secret):
                self.assertNotIn(secret.lower(), sent)
        self.assertEqual(str(sdk.base_url).rstrip("/"), _BASE_URL, "the ambient base URL must not win")
        self.assertIn("bearer gateway-key", sent, "the configured key must still be sent")

    def test_the_ambient_environment_is_restored_after_construction(self) -> None:
        """Suppression is a process-wide swap, so leaving it in place would break others."""
        ambient = {"OPENAI_API_KEY": "sk-operators-personal-key", "OPENAI_ORG_ID": "org-x"}
        with mock.patch.dict("os.environ", ambient):
            OpenAiCompatClient(
                base_url=_BASE_URL, api_key="gateway-key", max_concurrent_requests=1
            )._get_client()

            for name, value in ambient.items():
                with self.subTest(variable=name):
                    self.assertEqual(os.environ.get(name), value)


class TestErrorSurface(unittest.TestCase):
    """Failures arrive as this client's error type, with the endpoint named."""

    def test_an_error_payload_becomes_a_client_error(self) -> None:
        """A JSON `error` field is raised rather than returned as a result."""
        import asyncio

        client = OpenAiCompatClient(base_url=_BASE_URL, api_key="k", max_concurrent_requests=1)

        def _fake_urlopen(req: Any, timeout: int | None = None) -> _FakeResponse:
            return _FakeResponse({"error": {"message": "model not found"}})

        with mock.patch.object(client, "_get_client", return_value="stdlib"):
            with mock.patch("urllib.request.urlopen", _fake_urlopen):
                with self.assertRaises(OpenAiCompatClientError) as caught:
                    asyncio.run(
                        client.chat_completion(
                            model="nope", messages=[{"role": "user", "content": "hi"}],
                            timeout_seconds=5, max_output_tokens=10,
                        )
                    )

        self.assertIn("model not found", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
