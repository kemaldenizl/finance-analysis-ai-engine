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
    LOCAL_PROCESSED_STORAGE_DIR: Path = Field(default=Path("/storage/processed"))

    CLASSIFICATION_MODEL_VERSION: str = "rules-v1"
    PREPROCESSING_VERSION: str = "preprocessing-v2-no-crop"

    PDF_RENDER_DPI: int = 220

    PREPROCESSING_SAVE_DEBUG_VARIANTS: bool = True
    PREPROCESSING_MAX_OUTPUT_VARIANTS_PER_PAGE: int = 4

    AI_ANALYSIS_VERSION: str = "ai-analysis-v2"

    LLM_ENABLED: bool = True
    LLM_BASE_URL: str = "http://ollama:11434"
    LLM_MODEL: str = "qwen2.5:1.5b"
    LLM_TIMEOUT_SECONDS: int = 90
    LLM_TEMPERATURE: float = 0.1

    EMBEDDING_ENABLED: bool = True
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    EMBEDDING_SIMILARITY_THRESHOLD: float = 0.52

    ANOMALY_MIN_ROWS_FOR_PYOD: int = 8
    ANOMALY_CONTAMINATION: float = 0.15

    FORECAST_MIN_MONTHS_TRANSFORMER: int = 6
    FORECAST_LOOKBACK_MONTHS: int = 3
    FORECAST_TRAIN_EPOCHS: int = 120
    FORECAST_RANDOM_SEED: int = 42

    AI_STORE_ANALYSES: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()