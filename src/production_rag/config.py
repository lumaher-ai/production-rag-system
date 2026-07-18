from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # App
    app_name: str = "production_rag"
    app_version: str = "0.1.0"
    environment: str = Field(default="development", description="development, staging, production")
    debug: bool = True

    # Server
    host: str = "127.0.0.1"
    port: int = 8000

    # Logging
    log_level: str = Field(default="INFO", description="DEBUG, INFO, WARNING, ERROR, CRITICAL")

    # Database
    database_url: str = Field(
        default="postgresql+asyncpg://paddington:paddington_dev_password@localhost:5432/paddington",
        description="Async PostgreSQL connection URL",
    )

    @property
    def checkpointer_url(self) -> str:
        """psycopg-compatible URL for the LangGraph checkpointer.

        SQLAlchemy selects its async driver via the ``+asyncpg`` tag, but
        AsyncPostgresSaver is psycopg-based and rejects that tag — strip it.
        """
        return self.database_url.replace("postgresql+asyncpg://", "postgresql://", 1)

    # Auth
    jwt_secret_key: str = Field(
        default="CHANGE-ME-IN-PRODUCTION-use-a-real-random-string",
        description="Secret key for signing JWTs",
    )
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 30
    refresh_token_expire_days: int = 7

    # Document uploads
    max_upload_bytes: int = Field(
        default=10_485_760,
        description="Maximum accepted upload size in bytes (10 MiB)",
    )

    openai_api_key: str = Field(default="", description="OpenAI API key")
    anthropic_api_key: str = Field(default="", description="Anthropic API key")

    # Booking PII (read internally by fill_checkout_form; never passed as tool params)
    booking_full_name: str = Field(default="", description="First/given name for checkout")
    booking_last_name: str = Field(default="", description="Last/family name for checkout")
    booking_email: str = Field(default="", description="Email for checkout")
    booking_dni: str = Field(default="", description="Document / cédula for checkout")

    # LLM for LiteLLM
    default_model: str = Field(
        default="gpt-4o-mini",
        description="Default LLM model for completions",
    )
    fallback_model: str = Field(
        default="claude-haiku-4-5-20251001",
        description="Fallback model if primary fails",
    )

    # Debug observability
    debug_dir: str = Field(
        default="debug_runs",
        description="Root folder for debug screenshot runs",
    )
    max_runs_per_thread: int = Field(
        default=10,
        description="Keep at most N debug run folders per thread (oldest pruned)",
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
