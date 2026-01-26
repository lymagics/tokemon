# Tokemon

A unified Python library for counting tokens across multiple LLM providers. Tokemon provides a simple, consistent interface to count tokens for OpenAI, Anthropic, Google AI, and xAI models.

## Features

- Unified API for token counting across multiple providers
- Support for both synchronous and asynchronous operations
- Built-in model validation
- Type-safe responses with dataclass

## Installation

```bash
pip install tokemon
```

## Quick Start

```python
from tokemon import tokemon, Provider, Mode

# Create a tokenizer for your preferred provider
tokenizer = tokemon(
    model="gpt-4o",
    provider=Provider.OPENAI.value,
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
from tokemon import tokemon, Provider, Mode

tokenizer = tokemon(
    model="gpt-4o",
    provider=Provider.OPENAI.value,
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
from tokemon import tokemon, Provider, Mode

tokenizer = tokemon(
    model="claude-sonnet-4-5",
    provider=Provider.ANTHROPIC.value,
    mode=Mode.SYNC,
)

response = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {response.input_tokens}")
```

**Async example:**

```python
import asyncio
from tokemon import tokemon, Provider, Mode

async def main():
    tokenizer = tokemon(
        model="claude-sonnet-4-5",
        provider=Provider.ANTHROPIC.value,
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
from tokemon import tokemon, Provider, Mode

tokenizer = tokemon(
    model="gemini-2.5-flash",
    provider=Provider.GOOGLE.value,
    mode=Mode.SYNC,
)

response = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {response.input_tokens}")
```

**Async example:**

```python
import asyncio
from tokemon import tokemon, Provider, Mode

async def main():
    tokenizer = tokemon(
        model="gemini-2.5-flash",
        provider=Provider.GOOGLE.value,
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
from tokemon import tokemon, Provider, Mode

tokenizer = tokemon(
    model="grok-3",
    provider=Provider.XAI.value,
    mode=Mode.SYNC,
)

response = tokenizer.count_tokens("Hello, world!")
print(f"Token count: {response.input_tokens}")
```

**Async example:**

```python
import asyncio
from tokemon import tokemon, Provider, Mode

async def main():
    tokenizer = tokemon(
        model="grok-3",
        provider=Provider.XAI.value,
        mode=Mode.ASYNC,
    )
    response = await tokenizer.count_tokens("Hello, world!")
    print(f"Token count: {response.input_tokens}")

asyncio.run(main())
```

## Supported Models

### OpenAI
- GPT-5, GPT-4.1, GPT-4o, GPT-4, GPT-3.5-turbo
- o1, o3, o4-mini
- Text embedding models (text-embedding-ada-002, text-embedding-3-small, text-embedding-3-large)
- Legacy models (davinci, curie, babbage, ada)

### Anthropic
- claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5
- claude-opus-4-1, claude-opus-4-0, claude-sonnet-4-0
- claude-3-7-sonnet-latest, claude-3-haiku-20240307

### Google AI
- gemini-3-pro-preview, gemini-3-flash-preview
- gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite
- gemini-2.0-flash, gemini-2.0-flash-lite

### xAI
- grok-4-1-fast-reasoning, grok-4-1-fast-non-reasoning
- grok-4-fast-reasoning, grok-4-fast-non-reasoning, grok-4-0709
- grok-3, grok-3-mini
- grok-code-fast-1, grok-2-vision-1212

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
