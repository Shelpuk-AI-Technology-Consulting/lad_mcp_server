"""Which credentials are required to start, and which are never sourced ambiently.

Two properties are pinned here. Startup must accept a deployment that has no
OpenRouter account at all (issue #2's whole point) while still refusing one that has
no provider whatsoever. And no client may ever pick a credential up from the
environment: `AsyncOpenAI(api_key=None)` falls back to `OPENAI_API_KEY`, which is set
in most Claude Code / Codex environments, so an unset provider key would send the
operator's personal OpenAI key upstream.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import unittest
from typing import Any
from unittest import mock

from lad_mcp_server.config import Settings
from lad_mcp_server.model_metadata import ModelMetadataError, OpenRouterModelsClient
from lad_mcp_server.openrouter_client import OpenRouterClient
from lad_mcp_server.review_service import ReviewService

# Every variable `Settings.from_env` consults for credentials. Cleared wholesale so a
# developer's real environment cannot make these pass or fail.
_CREDENTIAL_VARS = (
    "OPENROUTER_API_KEY",
    "ZAI_CODING_PLAN_KEY",
    "KIMI_CODE_API_KEY",
    "DEEPSEEK_API_KEY",
    "OLLAMA_API_KEY",
    "OPENAI_COMPAT_BASE_URL",
    "OPENAI_COMPAT_API_KEY",
    "LAD_ENV_FILE",
)


def _capture_openrouter_request(client: OpenRouterClient) -> Any:
    """Run a stdlib-path chat call and return the request that was built.

    Args:
        client: The OpenRouter chat client.

    Returns:
        The captured ``urllib.request.Request``.
    """
    captured: list[Any] = []

    def _fake_urlopen(req: Any, timeout: int | None = None) -> Any:
        captured.append(req)
        raise RuntimeError("stop here; the request itself is what is under test")

    with mock.patch("urllib.request.urlopen", _fake_urlopen):
        with contextlib.suppress(Exception):
            asyncio.run(
                client.chat_completion(
                    model="test/model",
                    messages=[{"role": "user", "content": "hi"}],
                    timeout_seconds=5,
                    max_output_tokens=10,
                    tools=None,
                    tool_choice=None,
                    extra_body=None,
                )
            )
    return captured[0]


def _capture_models_request(client: OpenRouterModelsClient) -> Any:
    """Run a metadata fetch and return the request that was built.

    Args:
        client: The OpenRouter metadata client.

    Returns:
        The captured ``urllib.request.Request``.
    """
    captured: list[Any] = []

    def _fake_urlopen(req: Any, timeout: int | None = None) -> Any:
        captured.append(req)
        raise RuntimeError("stop here; the request itself is what is under test")

    with mock.patch("urllib.request.urlopen", _fake_urlopen):
        with contextlib.suppress(ModelMetadataError):
            client._fetch_models_payload()
    return captured[0]


class _CleanEnvironment(unittest.TestCase):
    """Base class that runs each test with no credentials in the environment."""

    def setUp(self) -> None:
        """Remove every credential variable for the duration of the test."""
        patcher = mock.patch.dict(os.environ, {}, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for name in _CREDENTIAL_VARS:
            os.environ.pop(name, None)


class TestStartupCredentialRequirements(_CleanEnvironment):
    """At least one provider must be configured — but it need not be OpenRouter."""

    def test_a_base_url_alone_is_enough_to_start(self) -> None:
        """The reported blocker: no OpenRouter account, so no OPENROUTER_API_KEY."""
        os.environ["OPENAI_COMPAT_BASE_URL"] = "https://gateway.internal.example.com/v1"

        settings = Settings.from_env()

        self.assertIsNone(settings.openrouter_api_key)
        self.assertEqual(settings.openai_compat_base_url, "https://gateway.internal.example.com/v1")

    def test_a_keyless_gateway_is_allowed(self) -> None:
        """A local vLLM or dev LiteLLM needs no key; demanding a dummy one is a papercut."""
        os.environ["OPENAI_COMPAT_BASE_URL"] = "http://localhost:4000/v1"

        settings = Settings.from_env()

        self.assertIsNone(settings.openai_compat_api_key)

    def test_each_direct_provider_alone_is_enough(self) -> None:
        """The requirement is "some provider", not "this provider"."""
        for variable in ("ZAI_CODING_PLAN_KEY", "KIMI_CODE_API_KEY", "DEEPSEEK_API_KEY", "OLLAMA_API_KEY"):
            with self.subTest(variable=variable):
                os.environ[variable] = "a-key"
                try:
                    self.assertIsNone(Settings.from_env().openrouter_api_key)
                finally:
                    os.environ.pop(variable)

    def test_a_whitespace_only_base_url_does_not_count_as_configured(self) -> None:
        """It is truthy, so it would enable the provider and then fail as a bad URL."""
        os.environ["OPENAI_COMPAT_BASE_URL"] = "   "

        with self.assertRaises(ValueError):
            Settings.from_env()

    def test_no_provider_at_all_still_fails_closed(self) -> None:
        """Relaxing the requirement must not turn into having no requirement."""
        with self.assertRaises(ValueError) as caught:
            Settings.from_env()

        message = str(caught.exception)
        self.assertIn("OPENROUTER_API_KEY", message)
        self.assertIn("OPENAI_COMPAT_BASE_URL", message, "the message must offer the alternative")

    def test_the_failure_message_lists_every_accepted_alternative(self) -> None:
        """A user who has one of these should not have to read the source to find out.

        Enumerated rather than spot-checked: the failure mode this guards against is
        adding a provider and forgetting the message, which leaves the user believing
        their credential is unsupported.
        """
        with self.assertRaises(ValueError) as caught:
            Settings.from_env()

        for variable in _CREDENTIAL_VARS[:-2]:
            with self.subTest(variable=variable):
                self.assertIn(variable, str(caught.exception))


class TestNoCredentialComesFromTheEnvironment(_CleanEnvironment):
    """`api_key=None` is never passed to the OpenAI SDK, by any client.

    Asserted against the SDK's own `auth_headers` rather than against the argument we
    pass, because the property that matters is what goes on the wire — and it belongs
    to the `openai` package, which `pyproject.toml` bounds only from below.
    """

    def _settings(self) -> Settings:
        """Build settings for a gateway-only deployment.

        Returns:
            Settings with no OpenRouter key and a configured gateway.
        """
        os.environ["OPENAI_COMPAT_BASE_URL"] = "https://gateway.internal.example.com/v1"
        return Settings.from_env()

    def test_no_openrouter_request_carries_an_authorization_header(self) -> None:
        """Asserted on the wire, since `""` and `None` fail differently by SDK version.

        Both OpenRouter clients are checked at the point the request is built: the
        metadata client has only a stdlib path, and the chat client falls back to one
        whenever no credential is configured.
        """
        settings = self._settings()
        os.environ["OPENAI_API_KEY"] = "sk-operators-personal-key"
        service = ReviewService(repo_root=None, settings=settings)

        self.assertEqual(
            service._openrouter._get_client(), "stdlib", "no credential means no SDK client"
        )
        for label, request in (
            ("chat", _capture_openrouter_request(service._openrouter)),
            ("metadata", _capture_models_request(service._models)),
        ):
            with self.subTest(client=label):
                self.assertIsNone(request.headers.get("Authorization"))
                self.assertNotIn("sk-operators-personal-key", json.dumps(dict(request.headers)))

    def test_a_configured_openrouter_key_is_still_used(self) -> None:
        """Normalising the credential must not stop a real one from being sent."""
        os.environ["OPENROUTER_API_KEY"] = "sk-real-openrouter-key"
        os.environ["OPENAI_API_KEY"] = "sk-operators-personal-key"

        service = ReviewService(repo_root=None, settings=Settings.from_env())
        sdk = service._openrouter._get_client()
        if sdk == "stdlib":  # pragma: no cover
            self.skipTest("openai SDK not installed")

        self.assertEqual(sdk.auth_headers, {"Authorization": "Bearer sk-real-openrouter-key"})

    def test_the_gateway_client_is_built_from_the_configured_key_only(self) -> None:
        """End to end from environment to client: the ambient key must not survive."""
        settings = self._settings()
        os.environ["OPENAI_API_KEY"] = "sk-operators-personal-key"

        service = ReviewService(repo_root=None, settings=settings)

        self.assertEqual(service._openai_compat._get_client(), "stdlib")

    def test_a_configured_gateway_key_reaches_the_sdk(self) -> None:
        """The isolation must not come at the cost of dropping a real credential."""
        os.environ["OPENAI_COMPAT_BASE_URL"] = "https://gateway.internal.example.com/v1"
        os.environ["OPENAI_COMPAT_API_KEY"] = "gateway-key"
        os.environ["OPENAI_API_KEY"] = "sk-operators-personal-key"

        service = ReviewService(repo_root=None, settings=Settings.from_env())
        sdk = service._openai_compat._get_client()
        if sdk == "stdlib":  # pragma: no cover - only when the openai package is absent
            self.skipTest("openai SDK not installed")

        self.assertEqual(sdk.auth_headers, {"Authorization": "Bearer gateway-key"})


class TestConstructedClientsMatchTheirSettings(_CleanEnvironment):
    """The gateway client exists exactly when a base URL is configured."""

    def test_no_base_url_means_no_client(self) -> None:
        """Routing keys off the client's existence, so this decides FR11's diagnostic."""
        os.environ["DEEPSEEK_API_KEY"] = "a-key"

        service = ReviewService(repo_root=None, settings=Settings.from_env())

        self.assertIsNone(service._openai_compat)

    def test_a_base_url_means_a_client_pointed_at_it(self) -> None:
        """The base URL reaches the client verbatim."""
        os.environ["OPENAI_COMPAT_BASE_URL"] = "https://gateway.internal.example.com/v1"

        service = ReviewService(repo_root=None, settings=Settings.from_env())

        self.assertIsNotNone(service._openai_compat)
        self.assertEqual(
            service._openai_compat._base_url, "https://gateway.internal.example.com/v1"
        )

    def test_an_injected_client_is_not_replaced(self) -> None:
        """Tests and embedders inject their own; construction must not override it."""
        os.environ["OPENAI_COMPAT_BASE_URL"] = "https://gateway.internal.example.com/v1"
        injected: Any = object()

        service = ReviewService(
            repo_root=None, settings=Settings.from_env(), openai_compat_client=injected
        )

        self.assertIs(service._openai_compat, injected)


if __name__ == "__main__":
    unittest.main()
