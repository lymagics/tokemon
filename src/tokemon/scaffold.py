from .providers.base import AsyncProvider, Provider
from .providers.anthropic_ai import AsyncAnthropicProvider, AnthropicProvider
from .providers.google_ai import AsyncGoogleProvider, GoogleProvider
from .providers.openai import OpenAIProvider
from .providers.xai import AsyncXaiProvider, XaiProvider
from .tokenizers.base import AsyncTokenizer, Tokenizer
from .tokenizers.anthropic_ai import AsyncAnthropicTokenizer, AnthropicTokenizer
from .tokenizers.google_ai import AsyncGoogleAITokenizer, GoogleAITokenizer
from .tokenizers.openai import OpenAITokenizer
from .tokenizers.xai import AsyncXaiTokenizer, XaiTokenizer
from .model import Mode, ProviderName


def tokemon(
    model: str,
    provider: str,
    mode: str = Mode.SYNC,
) -> AsyncTokenizer | Tokenizer:
    if mode == Mode.ASYNC:
        provider = f"async-{provider}"

    tokenizers: dict[str, type[AsyncTokenizer | Tokenizer]] = {
        ProviderName.OPENAI.value: OpenAITokenizer,
        ProviderName.ANTHROPIC.value: AnthropicTokenizer,
        f"async-{ProviderName.ANTHROPIC.value}": AsyncAnthropicTokenizer,
        ProviderName.XAI.value: XaiTokenizer,
        f"async-{ProviderName.XAI.value}": AsyncXaiTokenizer,
        ProviderName.GOOGLE.value: GoogleAITokenizer,
        f"async-{ProviderName.GOOGLE.value}": AsyncGoogleAITokenizer,
    }

    if provider not in tokenizers:
        raise ValueError(f"Unsupported provider: {provider}")

    return tokenizers[provider](model=model)


def tokemon_models(
    provider: str,
    mode: str = Mode.SYNC,
) -> AsyncProvider | Provider:
    if mode == Mode.ASYNC:
        provider = f'async-{provider}'

    models: dict[str, type] = {
        ProviderName.OPENAI.value: OpenAIProvider,
        ProviderName.ANTHROPIC.value: AnthropicProvider,
        f'async-{ProviderName.ANTHROPIC.value}': AsyncAnthropicProvider,
        ProviderName.XAI.value: XaiProvider,
        f'async-{ProviderName.XAI.value}': AsyncXaiProvider,
        ProviderName.GOOGLE.value: GoogleProvider,
        f'async-{ProviderName.GOOGLE.value}': AsyncGoogleProvider,
    }
    if provider not in models:
        raise ValueError(f'Unsupported provider: {provider}')

    return models[provider]()
