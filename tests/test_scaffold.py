import pytest
from unittest.mock import MagicMock

from tokemon import tokemon
from tokemon.model import Provider, Mode


@pytest.fixture
def mock_tokenizers(monkeypatch):
    mocks = {}

    def make_mock(name):
        mock_cls = MagicMock(name=name)
        mock_instance = MagicMock(name=f"{name}Instance")
        mock_cls.return_value = mock_instance
        mocks[name] = mock_cls
        return mock_cls

    monkeypatch.setattr(
        "tokemon.scaffold.OpenAITokenizer",
        make_mock("OpenAITokenizer"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AnthropicTokenizer",
        make_mock("AnthropicTokenizer"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AsyncAnthropicTokenizer",
        make_mock("AsyncAnthropicTokenizer"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.GoogleAITokenizer",
        make_mock("GoogleAITokenizer"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AsyncGoogleAITokenizer",
        make_mock("AsyncGoogleAITokenizer"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.XaiTokenizer",
        make_mock("XaiTokenizer"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AsyncXaiTokenizer",
        make_mock("AsyncXaiTokenizer"),
    )

    return mocks


def test_openai_sync_factory(mock_tokenizers):
    result = tokemon(
        model="gpt-4",
        provider=Provider.OPENAI.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["OpenAITokenizer"].assert_called_once_with(model="gpt-4")
    assert result is mock_tokenizers["OpenAITokenizer"].return_value


def test_anthropic_sync_factory(mock_tokenizers):
    result = tokemon(
        model="claude-3",
        provider=Provider.ANTHROPIC.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["AnthropicTokenizer"].assert_called_once_with(model="claude-3")
    assert result is mock_tokenizers["AnthropicTokenizer"].return_value


def test_google_sync_factory(mock_tokenizers):
    result = tokemon(
        model="gemini-pro",
        provider=Provider.GOOGLE.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["GoogleAITokenizer"].assert_called_once_with(model="gemini-pro")
    assert result is mock_tokenizers["GoogleAITokenizer"].return_value


def test_xai_sync_factory(mock_tokenizers):
    result = tokemon(
        model="grok-2",
        provider=Provider.XAI.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["XaiTokenizer"].assert_called_once_with(model="grok-2")
    assert result is mock_tokenizers["XaiTokenizer"].return_value


def test_anthropic_async_factory(mock_tokenizers):
    result = tokemon(
        model="claude-3",
        provider=Provider.ANTHROPIC.value,
        mode=Mode.ASYNC,
    )

    mock_tokenizers["AsyncAnthropicTokenizer"].assert_called_once_with(model="claude-3")
    assert result is mock_tokenizers["AsyncAnthropicTokenizer"].return_value


def test_google_async_factory(mock_tokenizers):
    result = tokemon(
        model="gemini-pro",
        provider=Provider.GOOGLE.value,
        mode=Mode.ASYNC,
    )

    mock_tokenizers["AsyncGoogleAITokenizer"].assert_called_once_with(model="gemini-pro")
    assert result is mock_tokenizers["AsyncGoogleAITokenizer"].return_value


def test_xai_async_factory(mock_tokenizers):
    result = tokemon(
        model="grok-2",
        provider=Provider.XAI.value,
        mode=Mode.ASYNC,
    )

    mock_tokenizers["AsyncXaiTokenizer"].assert_called_once_with(model="grok-2")
    assert result is mock_tokenizers["AsyncXaiTokenizer"].return_value


def test_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        tokemon(
            model="some-model",
            provider="not-a-provider",
            mode=Mode.SYNC,
        )


def test_unsupported_async_openai_raises(mock_tokenizers):
    with pytest.raises(ValueError, match="Unsupported provider"):
        tokemon(
            model="gpt-4",
            provider=Provider.OPENAI.value,
            mode=Mode.ASYNC,
        )


def test_invalid_mode_treated_as_sync(mock_tokenizers):
    result = tokemon(
        model="gpt-4",
        provider=Provider.OPENAI.value,
        mode="not-a-real-mode",
    )

    mock_tokenizers["OpenAITokenizer"].assert_called_once_with(model="gpt-4")
    assert result is mock_tokenizers["OpenAITokenizer"].return_value
