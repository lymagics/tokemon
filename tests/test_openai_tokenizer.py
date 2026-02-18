import pytest
from unittest.mock import MagicMock, patch

from tokemon.tokenizers.openai import OpenAITokenizer
from tokemon.model import ProviderName, TokenizerResponse


FAKE_MODELS = ["gpt-4", "gpt-3.5-turbo", "gpt-4o"]


@pytest.fixture
def mock_provider(monkeypatch):
    mock_prov = MagicMock()
    mock_prov.models.return_value = FAKE_MODELS

    monkeypatch.setattr(
        "tokemon.tokenizers.openai.OpenAIProvider",
        lambda: mock_prov,
    )

    return mock_prov


@pytest.fixture
def mock_encoding(monkeypatch):
    fake_encoding = MagicMock()
    monkeypatch.setattr(
        "tiktoken.encoding_for_model",
        lambda model: fake_encoding,
    )
    return fake_encoding


@pytest.fixture
def valid_model():
    return FAKE_MODELS[0]


def test_init_with_valid_model(valid_model, mock_provider):
    tokenizer = OpenAITokenizer(valid_model)

    assert tokenizer.model == valid_model


def test_count_tokens_with_invalid_model(mock_provider, mock_encoding):
    tokenizer = OpenAITokenizer("not-a-real-model")

    with pytest.raises(ValueError, match="Unsupported model"):
        tokenizer.count_tokens("hello")


def test_count_tokens_normal_text(valid_model, mock_provider, mock_encoding):
    mock_encoding.encode.return_value = [1, 2, 3, 4]

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("hello world")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 4
    assert response.model == valid_model
    assert response.provider == ProviderName.OPENAI.value

    mock_encoding.encode.assert_called_once_with("hello world")


def test_count_tokens_empty_string(valid_model, mock_provider, mock_encoding):
    mock_encoding.encode.return_value = []

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("")

    assert response.input_tokens == 0


def test_count_tokens_whitespace(valid_model, mock_provider, mock_encoding):
    mock_encoding.encode.return_value = [42]

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("   ")

    assert response.input_tokens == 1


def test_count_tokens_large_input(valid_model, mock_provider, mock_encoding):
    mock_encoding.encode.return_value = list(range(10_000))

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("large input")

    assert response.input_tokens == 10_000


def test_encode_called_exactly_once(valid_model, mock_provider, mock_encoding):
    mock_encoding.encode.return_value = [1, 2]

    tokenizer = OpenAITokenizer(valid_model)
    tokenizer.count_tokens("test")

    mock_encoding.encode.assert_called_once()
