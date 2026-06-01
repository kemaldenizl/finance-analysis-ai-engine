import json
import logging
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.core.config import settings


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class OllamaProvider:
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
    ):
        self.base_url = (base_url or settings.LLM_BASE_URL).rstrip("/")
        self.model = model or settings.LLM_MODEL
        self.timeout = settings.LLM_TIMEOUT_SECONDS
        self.temperature = settings.LLM_TEMPERATURE

    def is_available(self) -> bool:
        if not settings.LLM_ENABLED:
            return False

        try:
            with httpx.Client(timeout=5.0) as client:
                response = client.get(f"{self.base_url}/api/tags")
                response.raise_for_status()

            return True

        except Exception as exc:
            logger.warning("Ollama unavailable: %s", exc)
            return False

    def generate_text(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str | None:
        if not settings.LLM_ENABLED:
            return None

        payload = {
            "model": self.model,
            "stream": False,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

            content = response.json()["message"]["content"]

            return str(content).strip()

        except Exception as exc:
            logger.exception("LLM text generation failed: %s", exc)

            return None

    def generate_structured(
        self,
        response_model: type[T],
        system_prompt: str,
        user_prompt: str,
    ) -> T | None:
        if not settings.LLM_ENABLED:
            return None

        schema = response_model.model_json_schema()

        grounded_prompt = (
            f"{user_prompt}\n\n"
            "Yalnızca verilen JSON schema ile uyumlu JSON döndür. "
            "Açıklama, markdown veya ek metin ekleme.\n"
            f"JSON schema:\n{json.dumps(schema, ensure_ascii=False)}"
        )

        payload = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "messages": [
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": grounded_prompt,
                },
            ],
            "options": {
                "temperature": self.temperature,
            },
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()

            content = response.json()["message"]["content"]

            return response_model.model_validate_json(content)

        except (httpx.HTTPError, KeyError, ValidationError, ValueError) as exc:
            logger.exception("LLM structured generation failed: %s", exc)

            return None