import pytest
from unittest.mock import MagicMock, AsyncMock

from tokemon.tokenizers.xai import (
    XaiTokenizer,
    AsyncXaiTokenizer,
)
from tokemon.model import Provider, TokenizerResponse, SUPPORTED_PROVIDERS


@pytest.fixture
def valid_model():
    return SUPPORTED_PROVIDERS[Provider.XAI.value][0]


@pytest.fixture
def mock_sync_xai_client(monkeypatch):
    mock_client = MagicMock()

    mock_client.tokenize.tokenize_text.return_value = ["t1", "t2", "t3"]

    monkeypatch.setattr(
        "tokemon.tokenizers.xai.Client",
        lambda: mock_client,
    )

    return mock_client


@pytest.fixture
def mock_async_xai_client(monkeypatch):
    mock_client = MagicMock()

    mock_client.tokenize.tokenize_text = AsyncMock(
        return_value=["t1", "t2", "t3", "t4"]
    )

    monkeypatch.setattr(
        "tokemon.tokenizers.xai.AsyncClient",
        lambda: mock_client,
    )

    return mock_client


def test_sync_init_with_valid_model(valid_model, mock_sync_xai_client):
    tokenizer = XaiTokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_sync_xai_client


def test_async_init_with_valid_model(valid_model, mock_async_xai_client):
    tokenizer = AsyncXaiTokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.client is mock_async_xai_client


def test_sync_init_with_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        XaiTokenizer("invalid-model")


def test_async_init_with_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        AsyncXaiTokenizer("invalid-model")


def test_sync_count_tokens_normal_input(valid_model, mock_sync_xai_client):
    tokenizer = XaiTokenizer(valid_model)

    response = tokenizer.count_tokens("hello")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 3
    assert response.model == valid_model
    assert response.provider == Provider.XAI.value

    mock_sync_xai_client.tokenize.tokenize_text.assert_called_once_with(
        model=valid_model,
        text="hello",
    )


def test_sync_count_tokens_empty_string(valid_model, mock_sync_xai_client):
    mock_sync_xai_client.tokenize.tokenize_text.return_value = []

    tokenizer = XaiTokenizer(valid_model)
    response = tokenizer.count_tokens("")

    assert response.input_tokens == 0


def test_sync_count_tokens_large_input(valid_model, mock_sync_xai_client):
    mock_sync_xai_client.tokenize.tokenize_text.return_value = ["t"] * 10_000

    tokenizer = XaiTokenizer(valid_model)
    response = tokenizer.count_tokens("large input")

    assert response.input_tokens == 10_000


@pytest.mark.asyncio
async def test_async_count_tokens_normal_input(
    valid_model, mock_async_xai_client
):
    tokenizer = AsyncXaiTokenizer(valid_model)

    response = await tokenizer.count_tokens("hello async")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 4
    assert response.model == valid_model
    assert response.provider == Provider.XAI.value

    mock_async_xai_client.tokenize.tokenize_text.assert_awaited_once_with(
        model=valid_model,
        text="hello async",
    )


@pytest.mark.asyncio
async def test_async_count_tokens_empty_string(
    valid_model, mock_async_xai_client
):
    mock_async_xai_client.tokenize.tokenize_text.return_value = []

    tokenizer = AsyncXaiTokenizer(valid_model)
    response = await tokenizer.count_tokens("")

    assert response.input_tokens == 0
