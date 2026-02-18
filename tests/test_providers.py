import pytest
from unittest.mock import MagicMock, AsyncMock

from tokemon.providers.openai import OpenAIProvider
from tokemon.providers.anthropic_ai import AnthropicProvider, AsyncAnthropicProvider
from tokemon.providers.google_ai import (
    GoogleProvider,
    AsyncGoogleProvider,
    _strip_models_prefix,
)
from tokemon.providers.xai import XaiProvider, AsyncXaiProvider


def test_strip_models_prefix_with_prefix():
    assert _strip_models_prefix("models/gemini-2.5-pro") == "gemini-2.5-pro"


def test_strip_models_prefix_without_prefix():
    assert _strip_models_prefix("gemini-2.5-pro") == "gemini-2.5-pro"


def test_strip_models_prefix_empty_string():
    assert _strip_models_prefix("") == ""


def test_openai_provider_models(monkeypatch):
    fake_model_map = {"gpt-4": "cl100k_base", "gpt-3.5-turbo": "cl100k_base"}
    monkeypatch.setattr(
        "tiktoken.model.MODEL_TO_ENCODING",
        fake_model_map,
    )

    provider = OpenAIProvider()
    result = provider.models()

    assert set(result) == {"gpt-4", "gpt-3.5-turbo"}


def test_anthropic_sync_provider_models(monkeypatch):
    mock_client = MagicMock()
    mock_model_1 = MagicMock()
    mock_model_1.id = "claude-sonnet-4-5"
    mock_model_2 = MagicMock()
    mock_model_2.id = "claude-haiku-4-5"
    mock_client.models.list.return_value = MagicMock(
        data=[mock_model_1, mock_model_2]
    )

    monkeypatch.setattr(
        "tokemon.providers.anthropic_ai.Anthropic",
        lambda: mock_client,
    )

    provider = AnthropicProvider()
    result = provider.models()

    assert result == ["claude-sonnet-4-5", "claude-haiku-4-5"]


@pytest.mark.asyncio
async def test_anthropic_async_provider_models(monkeypatch):
    mock_client = MagicMock()
    mock_model_1 = MagicMock()
    mock_model_1.id = "claude-sonnet-4-5"
    mock_model_2 = MagicMock()
    mock_model_2.id = "claude-opus-4-5"
    mock_client.models.list = AsyncMock(
        return_value=MagicMock(data=[mock_model_1, mock_model_2])
    )

    monkeypatch.setattr(
        "tokemon.providers.anthropic_ai.AsyncAnthropic",
        lambda: mock_client,
    )

    provider = AsyncAnthropicProvider()
    result = await provider.models()

    assert result == ["claude-sonnet-4-5", "claude-opus-4-5"]


def test_google_sync_provider_models(monkeypatch):
    mock_client = MagicMock()
    mock_model_1 = MagicMock()
    mock_model_1.name = "models/gemini-2.5-pro"
    mock_model_2 = MagicMock()
    mock_model_2.name = "models/gemini-2.0-flash"
    mock_client.models.list.return_value = [mock_model_1, mock_model_2]

    monkeypatch.setattr(
        "tokemon.providers.google_ai.genai.Client",
        lambda: mock_client,
    )

    provider = GoogleProvider()
    result = provider.models()

    assert result == ["gemini-2.5-pro", "gemini-2.0-flash"]


@pytest.mark.asyncio
async def test_google_async_provider_models(monkeypatch):
    mock_client = MagicMock()
    mock_model_1 = MagicMock()
    mock_model_1.name = "models/gemini-2.5-flash"
    mock_model_2 = MagicMock()
    mock_model_2.name = "gemini-2.0-flash-lite"

    mock_client.aio = MagicMock()
    mock_client.aio.models.list = AsyncMock(
        return_value=[mock_model_1, mock_model_2]
    )

    monkeypatch.setattr(
        "tokemon.providers.google_ai.genai.Client",
        lambda: mock_client,
    )

    provider = AsyncGoogleProvider()
    result = await provider.models()

    assert result == ["gemini-2.5-flash", "gemini-2.0-flash-lite"]


def test_xai_sync_provider_models(monkeypatch):
    mock_client = MagicMock()
    mock_model_1 = MagicMock()
    mock_model_1.name = "grok-3"
    mock_model_2 = MagicMock()
    mock_model_2.name = "grok-3-mini"
    mock_client.models.list_language_models.return_value = [
        mock_model_1, mock_model_2
    ]

    monkeypatch.setattr(
        "tokemon.providers.xai.Client",
        lambda: mock_client,
    )

    provider = XaiProvider()
    result = provider.models()

    assert result == ["grok-3", "grok-3-mini"]


@pytest.mark.asyncio
async def test_xai_async_provider_models(monkeypatch):
    mock_client = MagicMock()
    mock_model_1 = MagicMock()
    mock_model_1.name = "grok-3"
    mock_model_2 = MagicMock()
    mock_model_2.name = "grok-2-vision-1212"

    mock_client.models.list_language_models = AsyncMock(
        return_value=[mock_model_1, mock_model_2]
    )

    monkeypatch.setattr(
        "tokemon.providers.xai.AsyncClient",
        lambda: mock_client,
    )

    provider = AsyncXaiProvider()
    result = await provider.models()

    assert result == ["grok-3", "grok-2-vision-1212"]
