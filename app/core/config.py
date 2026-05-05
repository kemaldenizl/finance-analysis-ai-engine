from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Finance AI Input Pipeline"
    ENV: str = "dev"
    DEBUG: bool = True

    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    MAX_UPLOAD_SIZE_MB: int = 50

    DATABASE_URL: str = "postgresql+psycopg2://postgres:postgres@postgres:5432/finance_ai"
    REDIS_URL: str = "redis://redis:6379/0"

    LOCAL_STORAGE_ROOT: Path = Field(default=Path("/storage"))
    LOCAL_INPUT_STORAGE_DIR: Path = Field(default=Path("/storage/inputs"))

    CLASSIFICATION_MODEL_VERSION: str = "rules-v1"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
