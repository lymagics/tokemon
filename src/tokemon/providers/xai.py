from functools import lru_cache

from async_lru import alru_cache
from xai_sdk import AsyncClient, Client

from .base import Provider, AsyncProvider


class XaiProvider(Provider):
    def __init__(self):
        self.client = Client()

    @lru_cache(maxsize=1)
    def models(self) -> list[str]:
        return [m.name for m in self.client.models.list_language_models()]


class AsyncXaiProvider(AsyncProvider):
    def __init__(self):
        self.client = AsyncClient()

    @alru_cache(maxsize=1, ttl=300)
    async def models(self) -> list[str]:
        response = await self.client.models.list_language_models()
        return [m.name for m in response]
