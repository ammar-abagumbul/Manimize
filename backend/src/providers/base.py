from abc import ABC, abstractmethod
from typing import Iterable
from ..types.chat_completions import ChatCompletionMessageType

class BaseLLM(ABC):
    def __init__(self, *, endpoint: str, credential: str, model: str | None = None):
        self.endpoint = endpoint
        self.credential = credential
        self.model = model

    @abstractmethod
    def complete(self, messages: Iterable[ChatCompletionMessageType]) -> str:
        pass
