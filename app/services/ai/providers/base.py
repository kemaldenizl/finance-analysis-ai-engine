from typing import Protocol, TypeVar

from pydantic import BaseModel


T = TypeVar("T", bound=BaseModel)


class LLMProvider(Protocol):
    def is_available(self) -> bool:
        ...

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
        num_predict: int | None = None,
    ) -> str | None:
        ...

    def generate_structured(
        self,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> T | None:
        ...