import pytest
from unittest.mock import MagicMock, AsyncMock

from tokemon.tokenizers.google_ai import (
    GoogleAITokenizer,
    AsyncGoogleAITokenizer,
)
from tokemon.model import ProviderName, TokenizerResponse


FAKE_MODELS = ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.0-flash"]


@pytest.fixture
def valid_model():
    return FAKE_MODELS[0]


@pytest.fixture
def mock_sync_provider(monkeypatch):
    mock_prov = MagicMock()
    mock_prov.models.return_value = FAKE_MODELS

    monkeypatch.setattr(
        "tokemon.tokenizers.google_ai.GoogleProvider",
        lambda: mock_prov,
    )

    return mock_prov


@pytest.fixture
def mock_async_provider(monkeypatch):
    mock_prov = MagicMock()
    mock_prov.models = AsyncMock(return_value=FAKE_MODELS)

    monkeypatch.setattr(
        "tokemon.tokenizers.google_ai.AsyncGoogleProvider",
        lambda: mock_prov,
    )

    return mock_prov


@pytest.fixture
def mock_google_client(monkeypatch):
    mock_client = MagicMock()

    # Sync path
    mock_client.models.count_tokens.return_value = MagicMock(
        total_tokens=5
    )

    # Async path
    mock_client.aio = MagicMock()
    mock_client.aio.models.count_tokens = AsyncMock(
        return_value=MagicMock(total_tokens=7)
    )

    monkeypatch.setattr(
        "tokemon.tokenizers.google_ai.genai.Client",
        lambda: mock_client,
    )

    return mock_client


def test_sync_init_with_valid_model(
    valid_model, mock_sync_provider, mock_google_client
):
    tokenizer = GoogleAITokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_google_client


def test_async_init_with_valid_model(
    valid_model, mock_async_provider, mock_google_client
):
    tokenizer = AsyncGoogleAITokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_google_client


def test_sync_count_tokens_with_invalid_model(
    mock_sync_provider, mock_google_client
):
    tokenizer = GoogleAITokenizer("invalid-model")

    with pytest.raises(ValueError, match="Unsupported model"):
        tokenizer.count_tokens("hello")


@pytest.mark.asyncio
async def test_async_count_tokens_with_invalid_model(
    mock_async_provider, mock_google_client
):
    tokenizer = AsyncGoogleAITokenizer("invalid-model")

    with pytest.raises(ValueError, match="Unsupported model"):
        await tokenizer.count_tokens("hello")


def test_sync_count_tokens_normal_input(
    valid_model, mock_sync_provider, mock_google_client
):
    tokenizer = GoogleAITokenizer(valid_model)

    response = tokenizer.count_tokens("hello")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 5
    assert response.model == valid_model
    assert response.provider == ProviderName.GOOGLE.value

    mock_google_client.models.count_tokens.assert_called_once_with(
        model=valid_model,
        contents="hello",
    )


def test_sync_count_tokens_empty_string(
    valid_model, mock_sync_provider, mock_google_client
):
    mock_google_client.models.count_tokens.return_value.total_tokens = 0

    tokenizer = GoogleAITokenizer(valid_model)
    response = tokenizer.count_tokens("")

    assert response.input_tokens == 0


def test_sync_count_tokens_large_input(
    valid_model, mock_sync_provider, mock_google_client
):
    mock_google_client.models.count_tokens.return_value.total_tokens = 10_000

    tokenizer = GoogleAITokenizer(valid_model)
    response = tokenizer.count_tokens("large input")

    assert response.input_tokens == 10_000


@pytest.mark.asyncio
async def test_async_count_tokens_normal_input(
    valid_model, mock_async_provider, mock_google_client
):
    tokenizer = AsyncGoogleAITokenizer(valid_model)

    response = await tokenizer.count_tokens("hello async")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 7
    assert response.model == valid_model
    assert response.provider == ProviderName.GOOGLE.value

    mock_google_client.aio.models.count_tokens.assert_awaited_once_with(
        model=valid_model,
        contents="hello async",
    )


@pytest.mark.asyncio
async def test_async_count_tokens_empty_string(
    valid_model, mock_async_provider, mock_google_client
):
    mock_google_client.aio.models.count_tokens.return_value.total_tokens = 0

    tokenizer = AsyncGoogleAITokenizer(valid_model)
    response = await tokenizer.count_tokens("")

    assert response.input_tokens == 0
