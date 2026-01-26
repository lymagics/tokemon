from .tokenizers.base import AsyncTokenizer, Tokenizer
from .tokenizers.anthropic_ai import AsyncAnthropicTokenizer, AnthropicTokenizer
from .tokenizers.google_ai import AsyncGoogleAITokenizer, GoogleAITokenizer
from .tokenizers.openai import OpenAITokenizer
from .tokenizers.xai import AsyncXaiTokenizer, XaiTokenizer
from .model import Mode, Provider


def tokemon(
    model: str,
    provider: str,
    mode: str = Mode.SYNC,
) -> AsyncTokenizer | Tokenizer:
    if mode == Mode.ASYNC:
        provider = f"async-{provider}"

    tokenizers: dict[str, type[AsyncTokenizer | Tokenizer]] = {
        Provider.OPENAI.value: OpenAITokenizer,
        Provider.ANTHROPIC.value: AnthropicTokenizer,
        f"async-{Provider.ANTHROPIC.value}": AsyncAnthropicTokenizer,
        Provider.XAI.value: XaiTokenizer,
        f"async-{Provider.XAI.value}": AsyncXaiTokenizer,
        Provider.GOOGLE.value: GoogleAITokenizer,
        f"async-{Provider.GOOGLE.value}": AsyncGoogleAITokenizer,
    }

    if provider not in tokenizers:
        raise ValueError(f"Unsupported provider: {provider}")

    return tokenizers[provider](model=model)
