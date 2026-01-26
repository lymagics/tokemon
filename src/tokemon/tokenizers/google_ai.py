from google import genai

from .base import AsyncTokenizer, Tokenizer
from ..model import TokenizerResponse, Provider, SUPPORTED_PROVIDERS


class GoogleAITokenizer(Tokenizer):
    def __init__(self, model: str):
        if model not in SUPPORTED_PROVIDERS[Provider.GOOGLE.value]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.client = genai.Client()

    def count_tokens(self, text: str) -> TokenizerResponse:
        response = self.client.models.count_tokens(
            model=self.model,
            contents=text,
        )
        return TokenizerResponse(
            input_tokens=response.total_tokens,
            model=self.model,
            provider=Provider.GOOGLE.value,
        )


class AsyncGoogleAITokenizer(AsyncTokenizer):
    def __init__(self, model: str):
        if model not in SUPPORTED_PROVIDERS[Provider.GOOGLE.value]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.client = genai.Client()

    async def count_tokens(self, text: str) -> TokenizerResponse:
        response = await self.client.aio.models.count_tokens(
            model=self.model,
            contents=text,
        )
        return TokenizerResponse(
            input_tokens=response.total_tokens,
            model=self.model,
            provider=Provider.GOOGLE.value,
        )
