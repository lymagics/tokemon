import pytest
from unittest.mock import MagicMock, AsyncMock

from tokemon.tokenizers.anthropic_ai import (
    AnthropicTokenizer,
    AsyncAnthropicTokenizer,
)
from tokemon.model import ProviderName, TokenizerResponse


FAKE_MODELS = ["claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-5"]


@pytest.fixture
def valid_model():
    return FAKE_MODELS[0]


@pytest.fixture
def mock_sync_provider(monkeypatch):
    mock_prov = MagicMock()
    mock_prov.models.return_value = FAKE_MODELS

    monkeypatch.setattr(
        "tokemon.tokenizers.anthropic_ai.AnthropicProvider",
        lambda: mock_prov,
    )

    return mock_prov


@pytest.fixture
def mock_async_provider(monkeypatch):
    mock_prov = MagicMock()
    mock_prov.models = AsyncMock(return_value=FAKE_MODELS)

    monkeypatch.setattr(
        "tokemon.tokenizers.anthropic_ai.AsyncAnthropicProvider",
        lambda: mock_prov,
    )

    return mock_prov


@pytest.fixture
def mock_sync_anthropic(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.count_tokens.return_value = MagicMock(
        input_tokens=5
    )

    monkeypatch.setattr(
        "tokemon.tokenizers.anthropic_ai.Anthropic",
        lambda: mock_client,
    )

    return mock_client


@pytest.fixture
def mock_async_anthropic(monkeypatch):
    mock_client = MagicMock()
    mock_client.messages.count_tokens = AsyncMock(
        return_value=MagicMock(input_tokens=7)
    )

    monkeypatch.setattr(
        "tokemon.tokenizers.anthropic_ai.AsyncAnthropic",
        lambda: mock_client,
    )

    return mock_client


def test_sync_init_with_valid_model(
    valid_model, mock_sync_provider, mock_sync_anthropic
):
    tokenizer = AnthropicTokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_sync_anthropic


def test_async_init_with_valid_model(
    valid_model, mock_async_provider, mock_async_anthropic
):
    tokenizer = AsyncAnthropicTokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_async_anthropic


def test_sync_count_tokens_with_invalid_model(
    mock_sync_provider, mock_sync_anthropic
):
    tokenizer = AnthropicTokenizer("invalid-model")

    with pytest.raises(ValueError, match="Unsupported model"):
        tokenizer.count_tokens("hello")


@pytest.mark.asyncio
async def test_async_count_tokens_with_invalid_model(
    mock_async_provider, mock_async_anthropic
):
    tokenizer = AsyncAnthropicTokenizer("invalid-model")

    with pytest.raises(ValueError, match="Unsupported model"):
        await tokenizer.count_tokens("hello")


def test_sync_count_tokens_normal_input(
    valid_model, mock_sync_provider, mock_sync_anthropic
):
    tokenizer = AnthropicTokenizer(valid_model)

    response = tokenizer.count_tokens("hello")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 5
    assert response.model == valid_model
    assert response.provider == ProviderName.ANTHROPIC.value

    mock_sync_anthropic.messages.count_tokens.assert_called_once_with(
        model=valid_model,
        messages=[{"role": "user", "content": "hello"}],
    )


def test_sync_count_tokens_empty_string(
    valid_model, mock_sync_provider, mock_sync_anthropic
):
    mock_sync_anthropic.messages.count_tokens.return_value.input_tokens = 0

    tokenizer = AnthropicTokenizer(valid_model)
    response = tokenizer.count_tokens("")

    assert response.input_tokens == 0


def test_sync_count_tokens_large_input(
    valid_model, mock_sync_provider, mock_sync_anthropic
):
    mock_sync_anthropic.messages.count_tokens.return_value.input_tokens = 10_000

    tokenizer = AnthropicTokenizer(valid_model)
    response = tokenizer.count_tokens("large input")

    assert response.input_tokens == 10_000


@pytest.mark.asyncio
async def test_async_count_tokens_normal_input(
    valid_model, mock_async_provider, mock_async_anthropic
):
    tokenizer = AsyncAnthropicTokenizer(valid_model)

    response = await tokenizer.count_tokens("hello async")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 7
    assert response.model == valid_model
    assert response.provider == ProviderName.ANTHROPIC.value

    mock_async_anthropic.messages.count_tokens.assert_awaited_once_with(
        model=valid_model,
        messages=[{"role": "user", "content": "hello async"}],
    )


@pytest.mark.asyncio
async def test_async_count_tokens_empty_string(
    valid_model, mock_async_provider, mock_async_anthropic
):
    mock_async_anthropic.messages.count_tokens.return_value.input_tokens = 0

    tokenizer = AsyncAnthropicTokenizer(valid_model)
    response = await tokenizer.count_tokens("")

    assert response.input_tokens == 0
