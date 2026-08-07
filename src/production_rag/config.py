from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root, derived from this file's location
# (<root>/src/production_rag/config.py). Used to resolve relative file paths in
# configuration so they do not depend on the process's working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_config_path(value: str) -> str:
    """Turn a configured file path into an absolute one, CWD-independently.

    A relative path in ``.env`` (``secrets/key.json``) silently means different
    files depending on where the process was launched from — it resolves under
    `uvicorn` started in the repo root, then fails under Docker, systemd, or a
    test runner with a different working directory, surfacing as a confusing
    "not configured" error rather than "wrong path".

    Resolution order, first hit wins:
      1. already absolute (``~`` expanded) — used as given;
      2. relative to the current working directory — preserves the behaviour of
         any existing deployment that relies on it;
      3. relative to the project root — what a developer writing
         ``secrets/key.json`` in ``.env`` actually means.

    If none exist, the CWD-relative absolute path is returned so the resulting
    error message names a concrete location instead of a bare relative string.
    Empty stays empty: that is "not configured", which is a valid state.
    """
    if not value:
        return value

    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    from_cwd = (Path.cwd() / path).resolve()
    if from_cwd.is_file():
        return str(from_cwd)

    from_root = (PROJECT_ROOT / path).resolve()
    if from_root.is_file():
        return str(from_root)

    return str(from_cwd)


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
        default=104_857_600,
        description="Maximum accepted document size in bytes (100 MiB). Caps both "
        "multipart uploads and connector fetches, so every ingestion path shares "
        "one limit. Sized for image-heavy PDFs: a 384-page book measured at 91 MiB "
        "on disk but only ~250K characters of extractable text, so the byte size "
        "of a PDF says little about its embedding cost.",
    )

    # Background ingestion (arq worker)
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL backing the ingestion job queue",
    )
    ingestion_batch_size: int = Field(
        default=100,
        description="Chunks embedded and committed per batch. Sets the resume "
        "granularity (a failure loses at most this many chunks' work) and bounds "
        "each embedding request, which the provider caps at 2048 inputs.",
    )
    ingestion_job_timeout_seconds: int = Field(
        default=3600,
        description="Maximum wall time for one ingestion job before the worker "
        "kills it. Must exceed the slowest realistic document.",
    )
    ingestion_max_attempts: int = Field(
        default=3,
        description="Worker retries per job. Retries resume from the last "
        "committed batch rather than restarting.",
    )
    ingestion_stale_after_seconds: int = Field(
        default=300,
        description="A 'running' job whose heartbeat is older than this is "
        "presumed orphaned (worker killed without recording a failure) and is "
        "re-queued. Must exceed the time one batch can take, or a slow but "
        "healthy job would be handed to a second worker and both would write "
        "the same chunks.",
    )

    # Source connectors (ingest-by-URI)
    source_fetch_timeout_seconds: float = Field(
        default=120.0,
        description="Per-operation timeout when a connector fetches a source URI. "
        "httpx applies this between reads rather than to the whole transfer, but "
        "it is sized against measured throughput: a 91 MiB Drive PDF took ~21s on "
        "a fast connection, so a slow link needs real headroom.",
    )
    source_fetch_max_redirects: int = Field(
        default=5,
        description="Maximum redirect hops followed when fetching an http(s) source. "
        "Each hop is re-checked against the SSRF guard.",
    )
    allow_insecure_http_sources: bool = Field(
        default=False,
        description="Allow ingesting plaintext http:// URLs. Off by default; "
        "intended for local development only.",
    )
    allow_private_network_sources: bool = Field(
        default=False,
        description="Disable the SSRF guard that blocks fetches resolving to "
        "private/loopback/link-local addresses. NEVER enable in production — it "
        "exposes cloud metadata endpoints and internal services. Exists so tests "
        "and local development can fetch from localhost.",
    )
    google_service_account_file: str = Field(
        default="",
        description="Path to a Google service-account JSON key with read access "
        "to the Drive files to ingest. Relative paths resolve against the project "
        "root, so they work regardless of the process's working directory. "
        "Empty disables the gdrive:// connector.",
    )

    @field_validator("google_service_account_file")
    @classmethod
    def _resolve_service_account_path(cls, value: str) -> str:
        """Normalize to an absolute path at load time.

        Done here rather than at the call site so every consumer — connector,
        health check, CLI — sees the same resolved path, and so the value shown
        in an error message is the location actually probed.
        """
        return resolve_config_path(value)

    # Ingestion / retrieval
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model — part of every idempotency key",
    )
    query_cache_ttl_seconds: int = Field(
        default=3600,
        description="Time-to-live for cached query answers (seconds)",
    )
    hnsw_ef_search: int = Field(
        default=100,
        description=(
            "Candidate-list size for HNSW queries (pgvector default is 40). Higher "
            "trades latency for recall; matters most when retrieval is filtered, "
            "since filtered-out rows still consume the candidate list. 0 leaves the "
            "server default in place."
        ),
    )
    hnsw_iterative_scan: str = Field(
        default="relaxed_order",
        description=(
            "pgvector iterative scan mode: 'relaxed_order', 'strict_order', or 'off'. "
            "Without it an HNSW scan applies WHERE *after* walking the graph and can "
            "return fewer rows than LIMIT with no error — the filtered-ANN under-return "
            "bug. Requires pgvector >= 0.8; on older servers the setting is skipped "
            "with a warning. Empty string disables setting it at all."
        ),
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
