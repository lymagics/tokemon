from google import genai

from .base import AsyncTokenizer, Tokenizer
from ..providers.google_ai import GoogleProvider, AsyncGoogleProvider
from ..model import ProviderName, TokenizerResponse


class GoogleAITokenizer(Tokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = genai.Client()
        self.provider = GoogleProvider()

    def count_tokens(self, text: str) -> TokenizerResponse:
        if self.model not in self.provider.models():
            raise ValueError(f'Unsupported model: {self.model}')
        response = self.client.models.count_tokens(
            model=self.model,
            contents=text,
        )
        return TokenizerResponse(
            input_tokens=response.total_tokens,
            model=self.model,
            provider=ProviderName.GOOGLE.value,
        )


class AsyncGoogleAITokenizer(AsyncTokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = genai.Client()
        self.provider = AsyncGoogleProvider()

    async def count_tokens(self, text: str) -> TokenizerResponse:
        if self.model not in await self.provider.models():
            raise ValueError(f'Unsupported model: {self.model}')
        response = await self.client.aio.models.count_tokens(
            model=self.model,
            contents=text,
        )
        return TokenizerResponse(
            input_tokens=response.total_tokens,
            model=self.model,
            provider=ProviderName.GOOGLE.value,
        )
