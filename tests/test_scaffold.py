import pytest
from unittest.mock import MagicMock

from tokemon import tokemon, tokemon_models
from tokemon.model import ProviderName, Mode


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


@pytest.fixture
def mock_providers(monkeypatch):
    mocks = {}

    def make_mock(name):
        mock_cls = MagicMock(name=name)
        mock_instance = MagicMock(name=f"{name}Instance")
        mock_cls.return_value = mock_instance
        mocks[name] = mock_cls
        return mock_cls

    monkeypatch.setattr(
        "tokemon.scaffold.OpenAIProvider",
        make_mock("OpenAIProvider"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AnthropicProvider",
        make_mock("AnthropicProvider"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AsyncAnthropicProvider",
        make_mock("AsyncAnthropicProvider"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.GoogleProvider",
        make_mock("GoogleProvider"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AsyncGoogleProvider",
        make_mock("AsyncGoogleProvider"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.XaiProvider",
        make_mock("XaiProvider"),
    )
    monkeypatch.setattr(
        "tokemon.scaffold.AsyncXaiProvider",
        make_mock("AsyncXaiProvider"),
    )

    return mocks


def test_openai_sync_factory(mock_tokenizers):
    result = tokemon(
        model="gpt-4",
        provider=ProviderName.OPENAI.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["OpenAITokenizer"].assert_called_once_with(model="gpt-4")
    assert result is mock_tokenizers["OpenAITokenizer"].return_value


def test_anthropic_sync_factory(mock_tokenizers):
    result = tokemon(
        model="claude-3",
        provider=ProviderName.ANTHROPIC.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["AnthropicTokenizer"].assert_called_once_with(model="claude-3")
    assert result is mock_tokenizers["AnthropicTokenizer"].return_value


def test_google_sync_factory(mock_tokenizers):
    result = tokemon(
        model="gemini-pro",
        provider=ProviderName.GOOGLE.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["GoogleAITokenizer"].assert_called_once_with(model="gemini-pro")
    assert result is mock_tokenizers["GoogleAITokenizer"].return_value


def test_xai_sync_factory(mock_tokenizers):
    result = tokemon(
        model="grok-2",
        provider=ProviderName.XAI.value,
        mode=Mode.SYNC,
    )

    mock_tokenizers["XaiTokenizer"].assert_called_once_with(model="grok-2")
    assert result is mock_tokenizers["XaiTokenizer"].return_value


def test_anthropic_async_factory(mock_tokenizers):
    result = tokemon(
        model="claude-3",
        provider=ProviderName.ANTHROPIC.value,
        mode=Mode.ASYNC,
    )

    mock_tokenizers["AsyncAnthropicTokenizer"].assert_called_once_with(model="claude-3")
    assert result is mock_tokenizers["AsyncAnthropicTokenizer"].return_value


def test_google_async_factory(mock_tokenizers):
    result = tokemon(
        model="gemini-pro",
        provider=ProviderName.GOOGLE.value,
        mode=Mode.ASYNC,
    )

    mock_tokenizers["AsyncGoogleAITokenizer"].assert_called_once_with(model="gemini-pro")
    assert result is mock_tokenizers["AsyncGoogleAITokenizer"].return_value


def test_xai_async_factory(mock_tokenizers):
    result = tokemon(
        model="grok-2",
        provider=ProviderName.XAI.value,
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
            provider=ProviderName.OPENAI.value,
            mode=Mode.ASYNC,
        )


def test_invalid_mode_treated_as_sync(mock_tokenizers):
    result = tokemon(
        model="gpt-4",
        provider=ProviderName.OPENAI.value,
        mode="not-a-real-mode",
    )

    mock_tokenizers["OpenAITokenizer"].assert_called_once_with(model="gpt-4")
    assert result is mock_tokenizers["OpenAITokenizer"].return_value


def test_tokemon_models_openai_sync(mock_providers):
    result = tokemon_models(
        provider=ProviderName.OPENAI.value,
        mode=Mode.SYNC,
    )

    mock_providers["OpenAIProvider"].assert_called_once_with()
    assert result is mock_providers["OpenAIProvider"].return_value


def test_tokemon_models_anthropic_sync(mock_providers):
    result = tokemon_models(
        provider=ProviderName.ANTHROPIC.value,
        mode=Mode.SYNC,
    )

    mock_providers["AnthropicProvider"].assert_called_once_with()
    assert result is mock_providers["AnthropicProvider"].return_value


def test_tokemon_models_anthropic_async(mock_providers):
    result = tokemon_models(
        provider=ProviderName.ANTHROPIC.value,
        mode=Mode.ASYNC,
    )

    mock_providers["AsyncAnthropicProvider"].assert_called_once_with()
    assert result is mock_providers["AsyncAnthropicProvider"].return_value


def test_tokemon_models_google_sync(mock_providers):
    result = tokemon_models(
        provider=ProviderName.GOOGLE.value,
        mode=Mode.SYNC,
    )

    mock_providers["GoogleProvider"].assert_called_once_with()
    assert result is mock_providers["GoogleProvider"].return_value


def test_tokemon_models_google_async(mock_providers):
    result = tokemon_models(
        provider=ProviderName.GOOGLE.value,
        mode=Mode.ASYNC,
    )

    mock_providers["AsyncGoogleProvider"].assert_called_once_with()
    assert result is mock_providers["AsyncGoogleProvider"].return_value


def test_tokemon_models_xai_sync(mock_providers):
    result = tokemon_models(
        provider=ProviderName.XAI.value,
        mode=Mode.SYNC,
    )

    mock_providers["XaiProvider"].assert_called_once_with()
    assert result is mock_providers["XaiProvider"].return_value


def test_tokemon_models_xai_async(mock_providers):
    result = tokemon_models(
        provider=ProviderName.XAI.value,
        mode=Mode.ASYNC,
    )

    mock_providers["AsyncXaiProvider"].assert_called_once_with()
    assert result is mock_providers["AsyncXaiProvider"].return_value


def test_tokemon_models_unsupported_provider_raises():
    with pytest.raises(ValueError, match="Unsupported provider"):
        tokemon_models(
            provider="not-a-provider",
            mode=Mode.SYNC,
        )


def test_tokemon_models_unsupported_async_openai_raises(mock_providers):
    with pytest.raises(ValueError, match="Unsupported provider"):
        tokemon_models(
            provider=ProviderName.OPENAI.value,
            mode=Mode.ASYNC,
        )
