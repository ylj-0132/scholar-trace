"""Configuration helpers for local CLI usage."""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: Path | None = None) -> None:
    """Load simple KEY=VALUE pairs from a local .env file without overriding env vars."""

    env_path = path or Path.cwd() / ".env"
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def openrouter_api_key() -> str | None:
    load_dotenv()
    return os.getenv("OPENROUTER_API_KEY") or None


def llm_api_base() -> str | None:
    load_dotenv()
    return os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_API_BASE") or None


def llm_request_timeout() -> float:
    load_dotenv()
    return float(os.getenv("LITELLM_REQUEST_TIMEOUT", "180"))


def llm_api_key() -> str | None:
    load_dotenv()
    return os.getenv("OPENAI_API_KEY") or None
