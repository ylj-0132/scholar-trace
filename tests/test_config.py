from __future__ import annotations

import os

from deep_research import config
from deep_research.config import llm_api_key, load_dotenv


def test_load_dotenv_sets_missing_values(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "SCHOLAR_TRACE_TEST_VALUE=test-value",
                "QUOTED='quoted value'",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("SCHOLAR_TRACE_TEST_VALUE", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    load_dotenv(env_path)

    assert os.getenv("SCHOLAR_TRACE_TEST_VALUE") == "test-value"
    assert os.getenv("QUOTED") == "quoted value"


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("SCHOLAR_TRACE_TEST_VALUE=from-file", encoding="utf-8")
    monkeypatch.setenv("SCHOLAR_TRACE_TEST_VALUE", "from-env")

    load_dotenv(env_path)

    assert os.getenv("SCHOLAR_TRACE_TEST_VALUE") == "from-env"


def test_llm_api_key_reads_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    assert llm_api_key() == "openai-key"


def test_openrouter_api_key_reads_standard_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "router-test")

    assert config.openrouter_api_key() == "router-test"
