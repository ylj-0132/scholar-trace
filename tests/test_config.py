from __future__ import annotations

import os

from deep_research.config import llm_api_key_for_model, llm_model, load_dotenv


def test_load_dotenv_sets_missing_values(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# comment",
                "SEMANTIC_SCHOLAR_API_KEY=test-key",
                "OPENALEX_API_KEY=openalex-key",
                "QUOTED='quoted value'",
            ]
        ),
        encoding="utf-8",
    )

    monkeypatch.delenv("SEMANTIC_SCHOLAR_API_KEY", raising=False)
    monkeypatch.delenv("OPENALEX_API_KEY", raising=False)
    monkeypatch.delenv("QUOTED", raising=False)

    load_dotenv(env_path)

    assert os.getenv("SEMANTIC_SCHOLAR_API_KEY") == "test-key"
    assert os.getenv("OPENALEX_API_KEY") == "openalex-key"
    assert os.getenv("QUOTED") == "quoted value"


def test_load_dotenv_does_not_override_existing_env(tmp_path, monkeypatch) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("SEMANTIC_SCHOLAR_API_KEY=from-file", encoding="utf-8")
    monkeypatch.setenv("SEMANTIC_SCHOLAR_API_KEY", "from-env")

    load_dotenv(env_path)

    assert os.getenv("SEMANTIC_SCHOLAR_API_KEY") == "from-env"


def test_llm_api_key_prefers_gemini_key_for_gemini_model(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    assert llm_api_key_for_model("gemini/gemini-3.1-pro-preview") == "gemini-key"


def test_llm_model_default_is_gpt_5_6_terra(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("LITELLM_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)

    assert llm_model() == "openai/gpt-5.6-terra"
