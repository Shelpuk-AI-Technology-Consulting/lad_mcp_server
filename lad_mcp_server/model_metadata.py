from __future__ import annotations

import json
import time
import threading
import urllib.request
from dataclasses import dataclass
from typing import Any


class ModelMetadataError(RuntimeError):
    pass


@dataclass(frozen=True)
class ProviderLimits:
    context_length: int | None = None
    max_completion_tokens: int | None = None


@dataclass(frozen=True)
class ModelMetadata:
    model_id: str
    context_length: int
    supported_parameters: tuple[str, ...]
    provider_limits: ProviderLimits

    def supports_tools(self) -> bool:
        # Missing/empty list must be treated as "no tool calling support".
        return "tools" in self.supported_parameters

    def effective_context_length(self) -> int:
        if self.provider_limits.context_length is None:
            return self.context_length
        return min(self.context_length, self.provider_limits.context_length)

    def effective_output_budget(self, fixed_output_tokens: int) -> int:
        if self.provider_limits.max_completion_tokens is None:
            return fixed_output_tokens
        return min(fixed_output_tokens, self.provider_limits.max_completion_tokens)


def _require_int(value: Any, field: str) -> int:
    if not isinstance(value, int):
        raise ModelMetadataError(f"Invalid model metadata: {field} must be an integer")
    return value


def parse_models_payload(payload: dict[str, Any]) -> dict[str, ModelMetadata]:
    """
    Parse OpenRouter Models API response into a mapping of model_id -> ModelMetadata.
    """
    data = payload.get("data")
    if not isinstance(data, list):
        raise ModelMetadataError("Invalid models payload: missing 'data' list")

    out: dict[str, ModelMetadata] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = item.get("id")
        if not isinstance(model_id, str) or not model_id:
            continue

        context_length = _require_int(item.get("context_length"), f"data[].context_length for {model_id}")

        supported_parameters_raw = item.get("supported_parameters")
        if isinstance(supported_parameters_raw, list) and all(isinstance(x, str) for x in supported_parameters_raw):
            supported_parameters: tuple[str, ...] = tuple(supported_parameters_raw)
        else:
            supported_parameters = ()

        top_provider = item.get("top_provider")
        provider_limits = ProviderLimits()
        if isinstance(top_provider, dict):
            ctx = top_provider.get("context_length")
            if isinstance(ctx, int):
                provider_limits = ProviderLimits(
                    context_length=ctx,
                    max_completion_tokens=top_provider.get("max_completion_tokens")
                    if isinstance(top_provider.get("max_completion_tokens"), int)
                    else None,
                )

        out[model_id] = ModelMetadata(
            model_id=model_id,
            context_length=context_length,
            supported_parameters=supported_parameters,
            provider_limits=provider_limits,
        )

    return out


class OpenRouterModelsClient:
    def __init__(self, *, api_key: str, ttl_seconds: int = 3600) -> None:
        self._api_key = api_key
        self._ttl_seconds = ttl_seconds
        self._cache_at: float | None = None
        self._cache_models: dict[str, ModelMetadata] | None = None
        self._lock = threading.Lock()

    def get_model(self, model_id: str) -> ModelMetadata:
        models = self.list_models()
        try:
            return models[model_id]
        except KeyError as exc:
            raise ModelMetadataError(f"Model '{model_id}' not found in OpenRouter models list") from exc

    def list_models(self) -> dict[str, ModelMetadata]:
        with self._lock:
            now = time.time()
            if (
                self._cache_models is not None
                and self._cache_at is not None
                and (now - self._cache_at) < self._ttl_seconds
            ):
                return self._cache_models

            payload = self._fetch_models_payload()
            models = parse_models_payload(payload)

            self._cache_at = now
            self._cache_models = models
            return models

    def _fetch_models_payload(self) -> dict[str, Any]:
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/models",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Accept": "application/json",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        except Exception as exc:
            raise ModelMetadataError(f"Failed to fetch OpenRouter models metadata: {exc}") from exc

        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ModelMetadataError("Failed to parse OpenRouter models response as JSON") from exc

        if not isinstance(parsed, dict):
            raise ModelMetadataError("Invalid OpenRouter models response: expected JSON object")
        return parsed
