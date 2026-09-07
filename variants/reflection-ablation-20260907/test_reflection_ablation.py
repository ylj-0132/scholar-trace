"""Offline checks for the current two-policy entry point; no local audit data required."""
import copy
import json
from pathlib import Path

import pytest

import run_reflection_ablation as comparison
from deep_research import paper_agent_runtime as runtime
from deep_research.paper_agent import AgentState, EvidenceTask, MasterAction, TraceStep, WorkerResult


def state():
    results = tuple(WorkerResult(
        EvidenceTask(label, rubric_ids=keys), finding=label, evidence="quote " + label,
        evidence_type="text", evidence_locator="p. 1",
    ) for label, keys in (("anchor", ()), ("related A", comparison.RUBRICS[:1]),
                          ("related B", comparison.RUBRICS[1:]), ("outside", ())))
    return runtime._state_payload(AgentState(findings=results, steps=(
        TraceStep(1, MasterAction("READ_PAPER", tuple(r.task for r in results)), results),
    ), remaining_rounds=3))


def test_only_current_native_policies_preserve_source(monkeypatch):
    monkeypatch.setattr(comparison, "ANCHORS", ("r1-t1-f1",))
    source = state()
    before = copy.deepcopy(source)
    inputs = comparison.build_inputs(source)
    assert set(inputs) == {"full-history", "rubric-union"}
    assert source == before
    for mode, payload in inputs.items():
        assert payload == comparison.native_prompt(source, mode)
        assert payload["context_mode"] == mode
        assert "overview_pages" not in payload
    assert len(inputs["full-history"]["state"]["findings"]) == 4
    union = inputs["rubric-union"]
    assert union["findings_to_reflect"] == ["r1-t1-f1", "r1-t2-f1", "r1-t3-f1"]
    assert union["context_selection"]["strategy"] == "rubric-union-plus-anchors"
    assert Path(runtime.__file__).resolve().is_relative_to(comparison.ROOT / "src")


def test_old_four_group_manifest_is_rejected_before_clients(tmp_path, monkeypatch):
    from deep_research import experiment
    monkeypatch.setattr(experiment, "_role_clients", lambda *_: pytest.fail("must not construct clients"))
    (tmp_path / "manifest.json").write_text(json.dumps({
        "schema": "reflection-context-ablation-v1",
        "groups": ["full-history", "rubric-focused", "rubric-routing", "current-v20"],
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="historical|unsupported"):
        comparison.run(tmp_path)


def test_prepare_and_fake_run_record_actual_context_and_refuse_reuse(tmp_path, monkeypatch):
    from deep_research import experiment
    monkeypatch.setattr(comparison, "ANCHORS", ("r1-t1-f1",))
    monkeypatch.setattr(comparison, "load_state", state)
    source = tmp_path / "source.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(comparison, "SOURCE", source)
    output = tmp_path / "comparison"
    comparison.prepare(output)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == "reflection-context-comparison-v2"
    assert manifest["groups"] == ["full-history", "rubric-union"]
    with pytest.raises(FileExistsError):
        comparison.prepare(output)

    class Client:
        model = "openai/gpt-5.6-terra"
        temperature = 1.0
        max_retries = 1
        allow_json_repair = False
        def complete_json(self, prompt, **kwargs):
            return {"reflection_memo": "Supplied evidence supports only a bounded conclusion."}
    monkeypatch.setattr(experiment, "_role_clients", lambda *_: ({"reflection": Client()}, ()))
    comparison.run(output)
    for mode, count in (("full-history", 4), ("rubric-union", 3)):
        result = json.loads((output / mode / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == "completed"
        assert result["report"]["context_mode"] == mode
        assert len(result["report"]["context_finding_ids"]) == count
        assert len(result["calls"]) == 1
    assert json.loads((output / "finished.json").read_text(encoding="utf-8"))["source_verified_after"]
    monkeypatch.setattr(experiment, "_role_clients", lambda *_: pytest.fail("must not replay"))
    with pytest.raises(ValueError, match="already started"):
        comparison.run(output)


def test_entry_requires_explicit_output():
    with pytest.raises(SystemExit) as exc:
        comparison.main(["prepare"])
    assert exc.value.code == 2


def test_load_state_excludes_final_decision_and_answers(tmp_path, monkeypatch):
    history = state()
    history["findings"] = [
        {**history["findings"][0], "finding_id": f"r1-t1-f{i}"} for i in range(1, 34)
    ]
    history["reflection_reports"] = [{"reflection_memo": "Earlier mechanism check"}]
    before = copy.deepcopy(history)
    before["provisional_assessment"] = "Before final Master"
    history["history"].append({"kind": "DECIDE", "assessment": "FINAL_DECISION_SECRET"})
    history["provisional_assessment"] = "FINAL_ASSESSMENT_SECRET"
    source = {"trace": {"model_calls": [
        {"role": "master", "user_prompt": json.dumps({"state": before}),
         "parsed_response": {"assessment": "FINAL_MASTER_ANSWER_SECRET"}},
        {"role": "synthesis", "user_prompt": json.dumps({"history": history}),
         "parsed_response": {"assessment": "FINAL_SYNTHESIS_SECRET"}},
    ]}}
    path = tmp_path / "source.json"
    path.write_text(json.dumps(source), encoding="utf-8")
    original = path.read_bytes()
    monkeypatch.setattr(comparison, "SOURCE", path)
    restored = comparison.load_state()
    assert restored["provisional_assessment"] == "Before final Master"
    assert len(restored["findings"]) == 33
    assert "SECRET" not in json.dumps(restored)
    assert path.read_bytes() == original
