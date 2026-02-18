from anthropic import Anthropic, AsyncAnthropic

from .base import AsyncTokenizer, Tokenizer
from ..providers.anthropic_ai import AnthropicProvider, AsyncAnthropicProvider
from ..model import ProviderName, TokenizerResponse


class AnthropicTokenizer(Tokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = Anthropic()
        self.provider = AnthropicProvider()

    def count_tokens(self, text: str) -> TokenizerResponse:
        if self.model not in self.provider.models():
            raise ValueError(f'Unsupported model: {self.model}')
        count = self.client.messages.count_tokens(
            model=self.model,
            messages=[
                {'role': 'user', 'content': text},
            ],
        )
        return TokenizerResponse(
            input_tokens=count.input_tokens,
            model=self.model,
            provider=ProviderName.ANTHROPIC.value,
        )


class AsyncAnthropicTokenizer(AsyncTokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.client = AsyncAnthropic()
        self.provider = AsyncAnthropicProvider()

    async def count_tokens(self, text: str) -> TokenizerResponse:
        if self.model not in await self.provider.models():
            raise ValueError(f'Unsupported model: {self.model}')
        count = await self.client.messages.count_tokens(
            model=self.model,
            messages=[
                {'role': 'user', 'content': text},
            ],
        )
        return TokenizerResponse(
            input_tokens=count.input_tokens,
            model=self.model,
            provider=ProviderName.ANTHROPIC.value,
        )
