from anthropic import Anthropic, AsyncAnthropic

from .base import AsyncTokenizer, Tokenizer
from ..model import TokenizerResponse, Provider, SUPPORTED_PROVIDERS


class AnthropicTokenizer(Tokenizer):
    def __init__(self, model: str):
        if model not in SUPPORTED_PROVIDERS[Provider.ANTHROPIC.value]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.client = Anthropic()

    def count_tokens(self, text: str) -> TokenizerResponse:
        count = self.client.messages.count_tokens(
            model=self.model,
            messages=[
                {"role": "user", "content": text},
            ]
        )
        return TokenizerResponse(
            input_tokens=count.input_tokens,
            model=self.model,
            provider=Provider.ANTHROPIC.value,
        )


class AsyncAnthropicTokenizer(AsyncTokenizer):
    def __init__(self, model: str):
        if model not in SUPPORTED_PROVIDERS[Provider.ANTHROPIC.value]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.client = AsyncAnthropic()

    async def count_tokens(self, text: str) -> TokenizerResponse:
        count = await self.client.messages.count_tokens(
            model=self.model,
            messages=[
                {"role": "user", "content": text},
            ]
        )
        return TokenizerResponse(
            input_tokens=count.input_tokens,
            model=self.model,
            provider=Provider.ANTHROPIC.value,
        )
