from __future__ import annotations

import json
from pathlib import Path

import pytest


def _pdf(path: Path) -> Path:
    path.write_bytes(b"%PDF-1.4\n% fake fixture\n")
    return path


class _SafeTrace:
    assessment = "The claim is bounded by the paper evidence."
    outcome = "DECIDE"
    error = None

    def to_json(self) -> str:
        return json.dumps(
            {
                "source_document": "paper.pdf",
                "assessment": self.assessment,
                "outcome": self.outcome,
                "model_calls": [],
            }
        )


class _UnsafeTrace(_SafeTrace):
    def __init__(self, unsafe_value: str) -> None:
        self.unsafe_value = unsafe_value

    def to_json(self) -> str:
        return json.dumps(
            {
                "source_document": "paper.pdf",
                "assessment": self.assessment,
                "outcome": self.outcome,
                "model_calls": [],
                "unsafe_value": self.unsafe_value,
            }
        )


def test_run_audit_rejects_missing_or_non_pdf_before_client_construction(
    monkeypatch, tmp_path: Path
) -> None:
    from deep_research import experiment

    constructed: list[str] = []
    monkeypatch.setattr(
        experiment,
        "LLMClient",
        lambda model, **_kwargs: constructed.append(model),
    )

    assert experiment.run_audit(tmp_path / "missing.pdf", tmp_path / "missing-output") == 1
    non_pdf = tmp_path / "paper.txt"
    non_pdf.write_text("not a PDF", encoding="utf-8")
    assert experiment.run_audit(non_pdf, tmp_path / "text-output") == 1
    assert constructed == []


def test_run_audit_refuses_existing_output_before_client_construction(
    monkeypatch, tmp_path: Path
) -> None:
    from deep_research import experiment

    constructed: list[str] = []
    monkeypatch.setattr(
        experiment,
        "LLMClient",
        lambda model, **_kwargs: constructed.append(model),
    )
    output_dir = tmp_path / "existing-output"
    output_dir.mkdir()

    assert experiment.run_audit(_pdf(tmp_path / "paper.pdf"), output_dir) == 1
    assert constructed == []


@pytest.mark.parametrize("mode", ["unknown", "rubric-focused", "rubric-routing"])
def test_invalid_reflection_context_is_rejected_before_output_creation(tmp_path: Path, monkeypatch, mode: str) -> None:
    from deep_research import experiment
    monkeypatch.setattr(experiment, "_role_clients", lambda *_: pytest.fail("must not construct clients"))
    output = tmp_path / "invalid-mode"
    assert experiment.run_audit(
        _pdf(tmp_path / "paper.pdf"), output,
        config=experiment.AuditConfig(reflection_context_mode=mode),
    ) == 1
    assert not output.exists()


def test_run_audit_fake_execution_writes_safe_create_once_outputs(
    monkeypatch, tmp_path: Path
) -> None:
    from deep_research import experiment

    constructed: list[str] = []
    captured: dict[str, object] = {}

    class _FakeClient:
        def __init__(self, model: str, **_kwargs: object) -> None:
            self.model = model
            constructed.append(model)

    def fake_run_local_paper_agent(**kwargs: object) -> _SafeTrace:
        captured.update(kwargs)
        return _SafeTrace()

    monkeypatch.setattr(experiment, "LLMClient", _FakeClient)
    monkeypatch.setattr(experiment, "run_local_paper_agent", fake_run_local_paper_agent)
    monkeypatch.setattr(experiment, "llm_api_key", lambda: "secret-key")
    output_dir = tmp_path / "audit-output"

    assert experiment.run_audit(_pdf(tmp_path / "paper.pdf"), output_dir) == 0
    assert constructed == list(experiment.ROLE_MODELS.values())
    assert captured["max_rounds"] == 5
    assert captured["max_reflections"] == 2
    assert captured["reflection_context_mode"] == "rubric-union"
    assert captured["worker_parallelism"] == 2
    assert captured["worker_context_mode"] == "selected-context"
    assert captured["master_context_mode"] == "incremental-no-raw-evidence"
    assert captured["paper_context_mode"] == "master-overview-history-only"
    assert captured["worker_role_mode"] == "legacy"
    assert set(captured["role_llms"]) == set(experiment.ROLE_MODELS)

    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    progress = json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))
    assert manifest["role_models"] == experiment.ROLE_MODELS
    assert manifest["prompt_versions"]["master"] == "master-v20-restored-convergence"
    assert manifest["prompt_versions"]["reflection"] == "reflection-v10-independent-request"
    assert manifest["investigation_target"] == captured["investigation_target"] == experiment.INVESTIGATION_TARGET
    assert manifest["reflection_policy"] == "post-method-model-and-master-requested"
    assert manifest["action_policy"] == "read-reflect-independent-batch-v1"
    assert manifest["prompt_versions"]["synthesis"] == "synthesis-v11-rubric-reconciliation"
    assert manifest["reflection_context_mode"] == "rubric-union"
    assert manifest["finding_provenance"] == "worker-id-dispositions-v1"
    assert "synthesis_comparison_attribution_revision" not in manifest
    assert manifest["paper_sha256"] == experiment._sha256(tmp_path / "paper.pdf")
    assert {
        "src/deep_research/experiment.py",
        "src/deep_research/paper_agent_runtime.py",
        "src/deep_research/paper_agent.py",
    } <= set(manifest["source_sha256"])
    assert manifest["status"] == "completed"
    assert result["status"] == "completed"
    assert progress["status"] == "completed"
    serialized = json.dumps({"manifest": manifest, "result": result, "progress": progress})
    assert "secret-key" not in serialized
    assert "data:image" not in serialized
    assert str(tmp_path.resolve()) not in serialized
    assert experiment.run_audit(_pdf(tmp_path / "paper.pdf"), output_dir) == 1


def test_safety_scan_allows_public_urls_but_rejects_posix_absolute_paths() -> None:
    from deep_research import experiment

    assert not experiment._has_unsafe_text("see https://github.com/org/repo for code")
    assert experiment._has_unsafe_text("read /home/researcher/private/paper.pdf")


def test_run_audit_failure_does_not_mutate_shared_runtime_configuration(
    monkeypatch, tmp_path: Path
) -> None:
    from deep_research import experiment
    from deep_research import paper_agent_runtime

    original_config = experiment.DEFAULT_AUDIT_CONFIG
    original_role_models = dict(experiment.ROLE_MODELS)
    original_runtime_modes = (
        set(paper_agent_runtime.MASTER_CONTEXT_MODES),
        set(paper_agent_runtime.PAPER_CONTEXT_MODES),
    )

    class _FakeClient:
        def __init__(self, model: str, **_kwargs: object) -> None:
            self.model = model

    def fake_run_local_paper_agent(**_kwargs: object) -> _SafeTrace:
        raise RuntimeError("fake runtime failure")

    monkeypatch.setattr(experiment, "LLMClient", _FakeClient)
    monkeypatch.setattr(experiment, "run_local_paper_agent", fake_run_local_paper_agent)

    output_dir = tmp_path / "failed-output"
    assert experiment.run_audit(_pdf(tmp_path / "paper.pdf"), output_dir) == 1
    assert experiment.DEFAULT_AUDIT_CONFIG == original_config
    assert experiment.ROLE_MODELS == original_role_models
    assert (
        set(paper_agent_runtime.MASTER_CONTEXT_MODES),
        set(paper_agent_runtime.PAPER_CONTEXT_MODES),
    ) == original_runtime_modes
    assert json.loads((output_dir / "result.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert json.loads((output_dir / "progress.json").read_text(encoding="utf-8"))["status"] == "failed"


@pytest.mark.parametrize(
    "benign_value",
    (
        "A" * 256,
        "/workspace/ReMe/config.json",
        r"C:\published\ReMe\config.json",
    ),
)
def test_run_audit_preserves_benign_paths_and_long_strings(
    monkeypatch, tmp_path: Path, benign_value: str
) -> None:
    from deep_research import experiment

    class _FakeClient:
        def __init__(self, model: str, **_kwargs: object) -> None:
            self.model = model

    monkeypatch.setattr(experiment, "LLMClient", _FakeClient)
    monkeypatch.setattr(experiment, "llm_api_key", lambda: None)
    monkeypatch.setattr(
        experiment,
        "run_local_paper_agent",
        lambda **_kwargs: _UnsafeTrace(benign_value),
    )
    output_dir = tmp_path / "benign-output"

    assert experiment.run_audit(_pdf(tmp_path / "paper.pdf"), output_dir) == 0
    result = json.loads((output_dir / "result.json").read_text(encoding="utf-8"))
    assert result["status"] == "completed"
    assert result["trace"]["unsafe_value"] == benign_value
    assert result["serialization_warnings"] == []


def test_run_audit_redacts_known_sensitive_output_and_records_warnings(
    monkeypatch, tmp_path: Path
) -> None:
    from deep_research import experiment

    class _FakeClient:
        def __init__(self, model: str, **_kwargs: object) -> None:
            self.model = model

    secret = "configured-secret"
    sensitive_value = (
        f"{tmp_path.resolve()}/private.json {secret} "
        "data:image/png;base64,iVBORw0KGgo" + "A" * 256
    )
    monkeypatch.setattr(experiment, "LLMClient", _FakeClient)
    monkeypatch.setattr(experiment, "llm_api_key", lambda: secret)
    monkeypatch.setattr(
        experiment,
        "run_local_paper_agent",
        lambda **_kwargs: _UnsafeTrace(sensitive_value),
    )
    output_dir = tmp_path / "sanitized-output"

    assert experiment.run_audit(_pdf(tmp_path / "paper.pdf"), output_dir) == 0
    result_text = (output_dir / "result.json").read_text(encoding="utf-8")
    result = json.loads(result_text)
    assert result["status"] == "completed"
    assert str(tmp_path.resolve()) not in result_text
    assert secret not in result_text
    assert "data:image" not in result_text
    assert result["trace"]["unsafe_value"] == (
        "<PAPER_ROOT>/private.json [REDACTED] [REDACTED_IMAGE_DATA]"
    )
    assert result["serialization_warnings"] == [
        "api_key_redacted",
        "image_data_redacted",
        "paper_root_redacted",
    ]
