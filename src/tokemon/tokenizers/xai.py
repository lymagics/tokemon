from xai_sdk import AsyncClient, Client

from .base import AsyncTokenizer, Tokenizer
from ..providers.xai import XaiProvider, AsyncXaiProvider
from ..model import ProviderName, TokenizerResponse


class XaiTokenizer(Tokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = Client()
        self.provider = XaiProvider()

    def count_tokens(self, text: str) -> TokenizerResponse:
        if self.model not in self.provider.models():
            raise ValueError(f'Unsupported model: {self.model}')
        response = self.client.tokenize.tokenize_text(
            model=self.model,
            text=text,
        )
        return TokenizerResponse(
            input_tokens=len(response),
            model=self.model,
            provider=ProviderName.XAI.value,
        )


class AsyncXaiTokenizer(AsyncTokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = AsyncClient()
        self.provider = AsyncXaiProvider()

    async def count_tokens(self, text: str) -> TokenizerResponse:
        if self.model not in await self.provider.models():
            raise ValueError(f'Unsupported model: {self.model}')
        response = await self.client.tokenize.tokenize_text(
            model=self.model,
            text=text,
        )
        return TokenizerResponse(
            input_tokens=len(response),
            model=self.model,
            provider=ProviderName.XAI.value,
        )
