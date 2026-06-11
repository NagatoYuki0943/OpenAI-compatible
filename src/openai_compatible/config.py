from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True, slots=True)
class Settings:
    host: str = field(default_factory=lambda: os.getenv("HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: int(os.getenv("PORT", "8000")))
    reload: bool = field(default_factory=lambda: _as_bool(os.getenv("RELOAD", "false")))
    api_key: str | None = field(default_factory=lambda: os.getenv("API_KEY"))

    model_id: str = field(default_factory=lambda: os.getenv("MODEL_ID", "demo-multimodal-model"))
    model_backend_class: str | None = field(
        default_factory=lambda: os.getenv("MODEL_BACKEND_CLASS")
    )
    model_max_concurrency: int = field(
        default_factory=lambda: int(os.getenv("MODEL_MAX_CONCURRENCY", "1"))
    )
    model_stream_chunk_size: int = field(
        default_factory=lambda: int(os.getenv("MODEL_STREAM_CHUNK_SIZE", "12"))
    )

    log_dir: Path = field(default_factory=lambda: Path(os.getenv("LOG_DIR", "logs")).resolve())
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO").upper())
    log_rotation: str = field(default_factory=lambda: os.getenv("LOG_ROTATION", "00:00"))
    log_retention: str = field(default_factory=lambda: os.getenv("LOG_RETENTION", "14 days"))
