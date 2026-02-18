from dataclasses import dataclass
from enum import Enum


class Mode:
    SYNC = 'sync'
    ASYNC = 'async'


class ProviderName(Enum):
    OPENAI = 'openai'
    ANTHROPIC = 'anthropic'
    XAI = 'xai'
    GOOGLE = 'google'


@dataclass
class TokenizerResponse:
    input_tokens: int | None
    model: str
    provider: str
