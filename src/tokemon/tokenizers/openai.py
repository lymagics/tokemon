import tiktoken

from .base import Tokenizer
from ..providers.openai import OpenAIProvider
from ..model import ProviderName, TokenizerResponse


class OpenAITokenizer(Tokenizer):
    def __init__(self, model: str):
        super().__init__(model)
        self.provider = OpenAIProvider()

    def count_tokens(self, text: str) -> TokenizerResponse:
        if self.model not in self.provider.models():
            raise ValueError(f'Unsupported model: {self.model}')
        encoding = tiktoken.encoding_for_model(self.model)
        response = encoding.encode(text)
        return TokenizerResponse(
            input_tokens=len(response),
            model=self.model,
            provider=ProviderName.OPENAI.value,
        )
