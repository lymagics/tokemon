import abc

from ..model import TokenizerResponse


class Tokenizer(abc.ABC):
    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    def count_tokens(self, text: str) -> TokenizerResponse:
        pass


class AsyncTokenizer(abc.ABC):
    def __init__(self, model: str):
        self.model = model

    @abc.abstractmethod
    async def count_tokens(self, text: str) -> TokenizerResponse:
        pass
