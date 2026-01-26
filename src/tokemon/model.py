from dataclasses import dataclass
from enum import Enum


class Mode:
    SYNC = "sync"
    ASYNC = "async"


class Provider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    XAI = "xai"
    GOOGLE = "google"


SUPPORTED_PROVIDERS = {
    Provider.ANTHROPIC.value: [
        "claude-sonnet-4-5",
        "claude-haiku-4-5",
        "claude-opus-4-5",
        "claude-opus-4-1",
        "claude-sonnet-4-0",
        "claude-3-7-sonnet-latest",
        "claude-opus-4-0",
        "claude-3-haiku-20240307",
    ],
    Provider.GOOGLE.value: [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-preview-09-2025",
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash-lite-preview-09-2025",
        "gemini-2.0-flash",
        "gemini-2.0-flash-lite",
    ],
    Provider.OPENAI.value: [
        "o1",
        "o3",
        "o4-mini",
        "gpt-5",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4",
        "gpt-3.5-turbo",
        "gpt-3.5",
        "gpt-35-turbo",
        "davinci-002",
        "babbage-002",
        "text-embedding-ada-002",
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-davinci-003",
        "text-davinci-002",
        "text-davinci-001",
        "text-curie-001",
        "text-babbage-001",
        "text-ada-001",
        "davinci",
        "curie",
        "babbage",
        "ada",
        "code-davinci-002",
        "code-davinci-001",
        "code-cushman-002",
        "code-cushman-001",
        "davinci-codex",
        "cushman-codex",
        "text-davinci-edit-001",
        "code-davinci-edit-001",
        "text-similarity-davinci-001",
        "text-similarity-curie-001",
        "text-similarity-babbage-001",
        "text-similarity-ada-001",
        "text-search-davinci-doc-001",
        "text-search-curie-doc-001",
        "text-search-babbage-doc-001",
        "text-search-ada-doc-001",
        "code-search-babbage-code-001",
        "code-search-ada-code-001",
        "gpt2",
        "gpt-2",
    ],
    Provider.XAI.value: [
        "grok-4-1-fast-reasoning",
        "grok-4-1-fast-non-reasoning",
        "grok-code-fast-1",
        "grok-4-fast-reasoning",
        "grok-4-fast-non-reasoning",
        "grok-4-0709",
        "grok-3-mini",
        "grok-3",
        "grok-2-vision-1212",
    ],
}


@dataclass
class TokenizerResponse:
    input_tokens: int | None
    model: str
    provider: str
