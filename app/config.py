from functools import lru_cache
from typing import Optional, Union
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):   
    GEMINI_API_KEY: str = ""
    GEMINI_API_KEY_1: str = ""
    GEMINI_API_KEY_2: str = ""
    GEMINI_API_KEY_3: str = ""
    GEMINI_API_KEY_4: str = ""
    GEMINI_MODEL: str = "gemini-3.5-flash"
    VLM_PROVIDER: str = "gemini"
    OLLAMA_MODEL: str = "qwen2.5-vl:7b"
    MAX_FILE_SIZE_MB: int = 10
    QUEUE_MAX_SIZE: int = 50
    WEBHOOK_RETRY_COUNT: int = 3
    WEBHOOK_TIMEOUT_SECONDS: int = 10
    ALLOWED_EXTENSIONS: Union[list[str], str] = ["jpg", "jpeg", "png", "webp", "pdf"]
    DATABASE_URL: Optional[str] = "postgresql://postgres:2301420045@localhost:5432/pod_verification"

    @property
    def gemini_api_keys(self) -> list[str]:
        """Returns list of configured non-empty Gemini API keys in priority order."""
        keys = []
        for k in [
            self.GEMINI_API_KEY,
            self.GEMINI_API_KEY_1,
            self.GEMINI_API_KEY_2,
            self.GEMINI_API_KEY_3,
            self.GEMINI_API_KEY_4,
        ]:
            if k and k.strip() and k.strip() not in keys:
                keys.append(k.strip())
        return keys

    @field_validator("ALLOWED_EXTENSIONS", mode="before")
    @classmethod
    def parse_allowed_extensions(cls, v: Union[str, list[str]]) -> list[str]:
        if isinstance(v, str):
            return [ext.strip().lower().lstrip(".") for ext in v.split(",") if ext.strip()]
        if isinstance(v, list):
            return [ext.strip().lower().lstrip(".") if isinstance(ext, str) else str(ext) for ext in v]
        return v

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
