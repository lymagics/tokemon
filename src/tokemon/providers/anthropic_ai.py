from functools import lru_cache

from anthropic import Anthropic, AsyncAnthropic
from async_lru import alru_cache

from .base import Provider, AsyncProvider


class AnthropicProvider(Provider):
    def __init__(self):
        self.client = Anthropic()

    @lru_cache(maxsize=1)
    def models(self) -> list[str]:
        client = Anthropic()
        return [m.id for m in client.models.list().data]


class AsyncAnthropicProvider(AsyncProvider):
    def __init__(self):
        self.client = AsyncAnthropic()

    @alru_cache(maxsize=1, ttl=300)
    async def models(self) -> list[str]:
        response = await self.client.models.list()
        return [m.id for m in response.data]
