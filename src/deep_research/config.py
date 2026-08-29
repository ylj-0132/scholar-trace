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


def semantic_scholar_api_key() -> str | None:
    load_dotenv()
    return os.getenv("SEMANTIC_SCHOLAR_API_KEY") or None


def openalex_api_key() -> str | None:
    load_dotenv()
    return os.getenv("OPENALEX_API_KEY") or None


def llm_model() -> str:
    load_dotenv()
    return os.getenv("LITELLM_MODEL", "openai/gpt-5.6-terra")


def llm_api_base() -> str | None:
    load_dotenv()
    return os.getenv("LITELLM_API_BASE") or os.getenv("OPENAI_API_BASE") or None


def llm_request_timeout() -> float:
    load_dotenv()
    return float(os.getenv("LITELLM_REQUEST_TIMEOUT", "180"))


def llm_api_key_for_model(model: str | None = None) -> str | None:
    load_dotenv()
    selected_model = (model or llm_model()).lower()
    deepseek_key = os.getenv("DEEPSEEK_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    if selected_model.startswith("gemini/"):
        return gemini_key or openai_key or deepseek_key or anthropic_key
    if selected_model.startswith("openai/"):
        return openai_key or deepseek_key or anthropic_key
    if selected_model.startswith("anthropic/") or selected_model.startswith("claude"):
        return anthropic_key or openai_key or deepseek_key
    return deepseek_key or gemini_key or openai_key or anthropic_key
