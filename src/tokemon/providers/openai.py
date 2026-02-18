from functools import lru_cache

import tiktoken

from .base import Provider


class OpenAIProvider(Provider):
    @lru_cache(maxsize=1)
    def models(self) -> list[str]:
        return list(tiktoken.model.MODEL_TO_ENCODING.keys())
