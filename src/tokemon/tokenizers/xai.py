from xai_sdk import AsyncClient, Client

from .base import AsyncTokenizer, Tokenizer
from ..model import TokenizerResponse, Provider, SUPPORTED_PROVIDERS


class XaiTokenizer(Tokenizer):
    def __init__(self, model: str):
        if model not in SUPPORTED_PROVIDERS[Provider.XAI.value]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.client = Client()

    def count_tokens(self, text: str) -> TokenizerResponse:
        response = self.client.tokenize.tokenize_text(
            model=self.model,
            text=text,
        )
        return TokenizerResponse(
            input_tokens=len(response),
            model=self.model,
            provider=Provider.XAI.value,
        )


class AsyncXaiTokenizer(AsyncTokenizer):
    def __init__(self, model: str):
        if model not in SUPPORTED_PROVIDERS[Provider.XAI.value]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.client = AsyncClient()

    async def count_tokens(self, text: str) -> TokenizerResponse:
        response = await self.client.tokenize.tokenize_text(
            model=self.model,
            text=text,
        )
        return TokenizerResponse(
            input_tokens=len(response),
            model=self.model,
            provider=Provider.XAI.value,
        )
