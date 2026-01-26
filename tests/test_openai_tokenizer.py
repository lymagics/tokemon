import pytest
from unittest.mock import MagicMock

from tokemon.tokenizers.openai import OpenAITokenizer
from tokemon.model import Provider, TokenizerResponse, SUPPORTED_PROVIDERS


@pytest.fixture
def mock_encoding(monkeypatch):
    fake_encoding = MagicMock()
    monkeypatch.setattr(
        "tiktoken.encoding_for_model",
        lambda model: fake_encoding
    )
    return fake_encoding


@pytest.fixture
def valid_model():
    return SUPPORTED_PROVIDERS[Provider.OPENAI.value][0]


def test_init_with_valid_model(valid_model, mock_encoding):
    tokenizer = OpenAITokenizer(valid_model)

    assert tokenizer.model == valid_model
    assert tokenizer.encoding is mock_encoding


def test_init_with_invalid_model():
    with pytest.raises(ValueError, match="Unsupported model"):
        OpenAITokenizer("not-a-real-model")


def test_count_tokens_normal_text(valid_model, mock_encoding):
    mock_encoding.encode.return_value = [1, 2, 3, 4]

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("hello world")

    assert isinstance(response, TokenizerResponse)
    assert response.input_tokens == 4
    assert response.model == valid_model
    assert response.provider == Provider.OPENAI.value

    mock_encoding.encode.assert_called_once_with("hello world")


def test_count_tokens_empty_string(valid_model, mock_encoding):
    mock_encoding.encode.return_value = []

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("")

    assert response.input_tokens == 0


def test_count_tokens_whitespace(valid_model, mock_encoding):
    mock_encoding.encode.return_value = [42]

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("   ")

    assert response.input_tokens == 1


def test_count_tokens_large_input(valid_model, mock_encoding):
    mock_encoding.encode.return_value = list(range(10_000))

    tokenizer = OpenAITokenizer(valid_model)
    response = tokenizer.count_tokens("large input")

    assert response.input_tokens == 10_000


def test_encode_called_exactly_once(valid_model, mock_encoding):
    mock_encoding.encode.return_value = [1, 2]

    tokenizer = OpenAITokenizer(valid_model)
    tokenizer.count_tokens("test")

    mock_encoding.encode.assert_called_once()
