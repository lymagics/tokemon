import abc


class Provider(abc.ABC):
    @abc.abstractmethod
    def models(self) -> list[str]:
        pass  # pragma: no cover


class AsyncProvider(abc.ABC):
    @abc.abstractmethod
    async def models(self) -> list[str]:
        pass  # pragma: no cover
