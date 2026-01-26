import pytest
from unittest.mock import MagicMock, AsyncMock

from tokemon.tokenizers.anthropic_ai import (
    AnthropicTokenizer,
    AsyncAnthropicTokenizer,
)
from tokemon.model import Provider, TokenizerResponse, SUPPORTED_PROVIDERS


@pytest.fixture
def valid_model():
    return SUPPORTED_PROVIDERS[Provider.ANTHROPIC.value][0]


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


def test_sync_init_with_valid_model(valid_model, mock_sync_anthropic):
    tokenizer = AnthropicTokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_sync_anthropic


def test_async_init_with_valid_model(valid_model, mock_async_anthropic):
    tokenizer = AsyncAnthropicTokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_async_anthropic


def test_sync_init_with_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        AnthropicTokenizer("invalid-model")


def test_async_init_with_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        AsyncAnthropicTokenizer("invalid-model")


def test_sync_count_tokens_normal_input(valid_model, mock_sync_anthropic):
    tokenizer = AnthropicTokenizer(valid_model)

    response = tokenizer.count_tokens("hello")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 5
    assert response.model == valid_model
    assert response.provider == Provider.ANTHROPIC.value

    mock_sync_anthropic.messages.count_tokens.assert_called_once_with(
        model=valid_model,
        messages=[{"role": "user", "content": "hello"}],
    )


def test_sync_count_tokens_empty_string(valid_model, mock_sync_anthropic):
    mock_sync_anthropic.messages.count_tokens.return_value.input_tokens = 0

    tokenizer = AnthropicTokenizer(valid_model)
    response = tokenizer.count_tokens("")

    assert response.input_tokens == 0


def test_sync_count_tokens_large_input(valid_model, mock_sync_anthropic):
    mock_sync_anthropic.messages.count_tokens.return_value.input_tokens = 10_000

    tokenizer = AnthropicTokenizer(valid_model)
    response = tokenizer.count_tokens("large input")

    assert response.input_tokens == 10_000


@pytest.mark.asyncio
async def test_async_count_tokens_normal_input(
    valid_model, mock_async_anthropic
):
    tokenizer = AsyncAnthropicTokenizer(valid_model)

    response = await tokenizer.count_tokens("hello async")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 7
    assert response.model == valid_model
    assert response.provider == Provider.ANTHROPIC.value

    mock_async_anthropic.messages.count_tokens.assert_awaited_once_with(
        model=valid_model,
        messages=[{"role": "user", "content": "hello async"}],
    )


@pytest.mark.asyncio
async def test_async_count_tokens_empty_string(
    valid_model, mock_async_anthropic
):
    mock_async_anthropic.messages.count_tokens.return_value.input_tokens = 0

    tokenizer = AsyncAnthropicTokenizer(valid_model)
    response = await tokenizer.count_tokens("")

    assert response.input_tokens == 0
