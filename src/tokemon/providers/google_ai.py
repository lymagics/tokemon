from functools import lru_cache

from async_lru import alru_cache
from google import genai

from .base import Provider, AsyncProvider


def _strip_models_prefix(name: str) -> str:
    if name.startswith('models/'):
        return name[len('models/'):]
    return name


class GoogleProvider(Provider):
    def __init__(self):
        self.client = genai.Client()

    @lru_cache(maxsize=1)
    def models(self) -> list[str]:
        client = genai.Client()
        return [_strip_models_prefix(m.name) for m in client.models.list()]


class AsyncGoogleProvider(AsyncProvider):
    def __init__(self):
        self.client = genai.Client()

    @alru_cache(maxsize=1, ttl=300)
    async def models(self) -> list[str]:
        response = await self.client.aio.models.list()
        return [
            _strip_models_prefix(m.name) for m in response
        ]
