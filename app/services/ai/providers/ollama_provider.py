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

    def _build_options(self) -> dict:
        return {
            "temperature": self.temperature,
            "num_ctx": settings.LLM_NUM_CTX,
            "seed": settings.LLM_SEED,
            "top_p": settings.LLM_TOP_P,
            "top_k": settings.LLM_TOP_K,
            "repeat_penalty": settings.LLM_REPEAT_PENALTY,
            "num_predict": settings.LLM_NUM_PREDICT,
        }

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
        num_predict: int | None = None,
    ) -> str | None:
        if not settings.LLM_ENABLED:
            return None

        options = self._build_options()

        if num_predict is not None:
            options["num_predict"] = num_predict

        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "keep_alive": settings.LLM_KEEP_ALIVE,
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
            "options": options,
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
            "Yalnızca verilen JSON şemasıyla uyumlu geçerli JSON döndür. "
            "Açıklama, markdown veya ek metin ekleme."
        )

        max_attempts = max(1, settings.LLM_MAX_RETRIES + 1)
        last_error: Exception | None = None

        for attempt in range(max_attempts):
            prompt = grounded_prompt

            if attempt > 0:
                prompt = (
                    f"{grounded_prompt}\n\n"
                    "Önceki yanıt şemaya uymuyordu. "
                    "Sadece şemaya birebir uyan geçerli JSON üret."
                )

            payload = {
                "model": self.model,
                "stream": False,
                "format": schema,
                "think": False,
                "keep_alive": settings.LLM_KEEP_ALIVE,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "options": self._build_options(),
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
                last_error = exc
                logger.warning(
                    "LLM structured generation attempt %d/%d failed: %s",
                    attempt + 1,
                    max_attempts,
                    exc,
                )

        logger.exception(
            "LLM structured generation failed after %d attempts: %s",
            max_attempts,
            last_error,
        )

        return None