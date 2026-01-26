import tiktoken

from .base import Tokenizer
from ..model import TokenizerResponse, Provider, SUPPORTED_PROVIDERS


class OpenAITokenizer(Tokenizer):
    def __init__(self, model: str):
        if model not in SUPPORTED_PROVIDERS[Provider.OPENAI.value]:
            raise ValueError(f"Unsupported model: {model}")
        self.model = model
        self.encoding = tiktoken.encoding_for_model(model)

    def count_tokens(self, text: str) -> TokenizerResponse:
        response = self.encoding.encode(text)
        return TokenizerResponse(
            input_tokens=len(response),
            model=self.model,
            provider=Provider.OPENAI.value,
        )
