# Tokemon

A unified Python library for counting tokens across multiple LLM providers. Tokemon provides a simple, consistent interface to count tokens for OpenAI, Anthropic, Google AI, and xAI models.

## Features

- Unified API for token counting across multiple providers
- Support for both synchronous and asynchronous operations
- Dynamic model discovery via provider APIs
- Type-safe responses with dataclass

## Installation

```bash
pip install tokemon
```

## Quick Start

```python
from tokemon import tokemon, ProviderName, Mode

# Create a tokenizer for your preferred provider
tokenizer = tokemon(
    model="gpt-4o",
    provider=ProviderName.OPENAI.value,
    mode=Mode.SYNC,
)

# Count tokens
response = tokenizer.count_tokens("Hello, world!")
print(response.input_tokens)  # Number of tokens
print(response.model)         # Model name
print(response.provider)      # Provider name
```

## Usage by Provider

### OpenAI

OpenAI tokenization uses [tiktoken](https://github.com/openai/tiktoken) and works offline without an API key.

```python
from tokemon import tokemon, ProviderName, Mode

tokenizer = tokemon(
    model="gpt-4o",
    provider=ProviderName.OPENAI.value,
    mode=Mode.SYNC,
)

response = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {response.input_tokens}")
```

### Anthropic

> **Note:** Set the `ANTHROPIC_API_KEY` environment variable before using the Anthropic provider.

```bash
export ANTHROPIC_API_KEY="your-api-key"
```

```python
from tokemon import tokemon, ProviderName, Mode

tokenizer = tokemon(
    model="claude-sonnet-4-5",
    provider=ProviderName.ANTHROPIC.value,
    mode=Mode.SYNC,
)

response = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {response.input_tokens}")
```

**Async example:**

```python
import asyncio
from tokemon import tokemon, ProviderName, Mode

async def main():
    tokenizer = tokemon(
        model="claude-sonnet-4-5",
        provider=ProviderName.ANTHROPIC.value,
        mode=Mode.ASYNC,
    )
    response = await tokenizer.count_tokens("Hello, world!")
    print(f"Token count: {response.input_tokens}")

asyncio.run(main())
```

### Google AI (Gemini)

> **Note:** Set the `GEMINI_API_KEY` environment variable before using the Google AI provider.

```bash
export GEMINI_API_KEY="your-api-key"
```

```python
from tokemon import tokemon, ProviderName, Mode

tokenizer = tokemon(
    model="gemini-2.5-flash",
    provider=ProviderName.GOOGLE.value,
    mode=Mode.SYNC,
)

response = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {response.input_tokens}")
```

**Async example:**

```python
import asyncio
from tokemon import tokemon, ProviderName, Mode

async def main():
    tokenizer = tokemon(
        model="gemini-2.5-flash",
        provider=ProviderName.GOOGLE.value,
        mode=Mode.ASYNC,
    )
    response = await tokenizer.count_tokens("Hello, world!")
    print(f"Token count: {response.input_tokens}")

asyncio.run(main())
```

### xAI (Grok)

> **Note:** Set the `XAI_API_KEY` environment variable before using the xAI provider.

```bash
export XAI_API_KEY="your-api-key"
```

```python
from tokemon import tokemon, ProviderName, Mode

tokenizer = tokemon(
    model="grok-3",
    provider=ProviderName.XAI.value,
    mode=Mode.SYNC,
)

response = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {response.input_tokens}")
```

**Async example:**

```python
import asyncio
from tokemon import tokemon, ProviderName, Mode

async def main():
    tokenizer = tokemon(
        model="grok-3",
        provider=ProviderName.XAI.value,
        mode=Mode.ASYNC,
    )
    response = await tokenizer.count_tokens("Hello, world!")
    print(f"Token count: {response.input_tokens}")

asyncio.run(main())
```

## Listing Available Models

Use `tokemon_models()` to discover models supported by each provider at runtime:

```python
from tokemon import tokemon_models, ProviderName, Mode

# Get a provider instance
provider = tokemon_models(
    provider=ProviderName.OPENAI.value,
    mode=Mode.SYNC,
)

# List available models
models = provider.models()
print(models)
```

**Async example:**

```python
import asyncio
from tokemon import tokemon_models, ProviderName, Mode

async def main():
    provider = tokemon_models(
        provider=ProviderName.ANTHROPIC.value,
        mode=Mode.ASYNC,
    )
    models = await provider.models()
    print(models)

asyncio.run(main())
```

## Response Object

The `count_tokens` method returns a `TokenizerResponse` dataclass:

```python
@dataclass
class TokenizerResponse:
    input_tokens: int | None  # Number of tokens in the input
    model: str                # Model name used for tokenization
    provider: str             # Provider name (openai, anthropic, google, xai)
```

## Requirements

- Python >= 3.10

## License

MIT License
