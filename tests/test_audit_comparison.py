from __future__ import annotations

import json

import pytest


def _call(**updates):
    return {"role": "master", "model": "model-a", "prompt_tokens": 100,
            "cached_prompt_tokens": 60, "cache_write_prompt_tokens": 20,
            "completion_tokens": 10, "total_tokens": 110,
            "latency_seconds": 4, "attempt_count": 1, **updates}


def _result(calls, wall=5):
    return {"status": "completed", "judgment_status": "decided", "wall_seconds": wall,
            "trace": {"model_calls": calls, "outcome": "DECIDE", "steps": [],
                      "final_judgment": {"key_findings": [], "provenance_warnings": []}}}


def _rates():
    return {"currency": "USD", "models": {
        "model-a": {"input": 2, "cached_input": 0.2, "cache_write": 2.5, "output": 12},
        "model-b": {"input": 1, "cached_input": 0.1, "cache_write": 1, "output": 2},
    }}


def test_cost_includes_cache_writes_and_routes_prices_by_actual_model():
    from deep_research.audit_comparison import summarize_run
    report = summarize_run(_result([_call(), _call(role="evidence", model="model-b")]), _rates())
    # a: 20*2 + 60*.2 + 20*2.5 + 10*12; b: 20*1 + 60*.1 + 20*1 + 10*2.
    assert report["estimated_recorded_cost"] == pytest.approx(0.000288)
    assert report["cached_prompt_tokens"] == 120
    assert report["cache_hit_fraction"] == 0.6
    assert report["wall_seconds"] == 5
    assert report["summed_call_seconds"] == 8
    assert report["by_role"]["master"]["estimated_recorded_cost"] == pytest.approx(0.000222)


def test_missing_usage_is_unknown_and_not_zero_or_a_partial_run_total():
    from deep_research.audit_comparison import summarize_run
    report = summarize_run(_result([_call(), _call(cached_prompt_tokens=None)]), _rates())
    assert report["cached_prompt_tokens"] is None
    assert report["cache_hit_fraction"] is None
    assert report["estimated_recorded_cost"] is None
    assert report["missing_usage_calls"]["cached_prompt_tokens"] == 1
    assert report["calls_with_cache_hits"] == 1
    assert report["calls_without_cache_usage"] == 1


def test_missing_writes_only_allow_cost_when_write_and_ordinary_prices_match():
    from deep_research.audit_comparison import summarize_run
    assert summarize_run(_result([_call(cache_write_prompt_tokens=None)]), _rates())["estimated_recorded_cost"] is None
    report = summarize_run(_result([_call(model="model-b", cache_write_prompt_tokens=None)]), _rates())
    assert report["cache_write_prompt_tokens"] is None
    assert report["estimated_recorded_cost"] == pytest.approx(0.000066)


def test_invalid_per_call_cache_counts_cannot_be_hidden_by_aggregation():
    from deep_research.audit_comparison import summarize_run
    report = summarize_run(_result([_call(cached_prompt_tokens=101), _call(cached_prompt_tokens=0)]), _rates())
    assert report["cache_hit_fraction"] is None
    assert report["invalid_cache_usage_calls"] == 1
    assert "inconsistent_cache_usage" in report["warnings"]


@pytest.mark.parametrize("updates", [
    {"model": "unknown"}, {"cached_prompt_tokens": 101},
    {"cache_write_prompt_tokens": 50}, {"prompt_tokens": None},
    {"cached_prompt_tokens": -1}, {"cached_prompt_tokens": True},
])
def test_unpriceable_usage_has_no_cost(updates):
    from deep_research.audit_comparison import summarize_run
    assert summarize_run(_result([_call(**updates)]), _rates())["estimated_recorded_cost"] is None


def test_retries_and_failed_calls_cannot_look_like_complete_billing():
    from deep_research.audit_comparison import summarize_run
    report = summarize_run(_result([
        _call(attempt_count=2),
        _call(error="timeout", prompt_tokens=None, completion_tokens=None, cached_prompt_tokens=None,
              cache_write_prompt_tokens=None, total_tokens=None),
    ]), _rates())
    assert report["extra_transport_attempts"] == 1
    assert report["failed_calls"] == 1
    assert report["estimated_recorded_cost"] is None
    assert "provider_billing_may_include_unobserved_attempts" in report["warnings"]


def test_offline_comparison_reports_observational_deltas_and_manifest_mismatches(tmp_path, capsys):
    from deep_research.audit_comparison import main
    for label, wall, layout, paper in [("baseline", 10, "standard", "a"), ("candidate", 8, "cache-friendly", "b")]:
        folder = tmp_path / label
        folder.mkdir()
        (folder / "result.json").write_text(json.dumps(_result([_call()], wall)), encoding="utf-8")
        (folder / "manifest.json").write_text(json.dumps({"paper_sha256": paper, "prompt_layout": layout}), encoding="utf-8")
    assert main([str(tmp_path / "baseline/result.json"), str(tmp_path / "candidate/result.json")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["observed_change_percent"]["wall_seconds"] == -20
    assert report["observed_change_percent"]["estimated_recorded_cost"] is None
    assert "manifest_mismatch:paper_sha256" in report["warnings"]
    assert report["baseline"]["prompt_layout"] == "standard"
    assert report["candidate"]["prompt_layout"] == "cache-friendly"
    assert report["quality_review_required"] is True


def test_historical_result_does_not_invent_wall_time_or_cache_counters():
    from deep_research.audit_comparison import summarize_run
    historical = _result([_call()])
    del historical["wall_seconds"]
    del historical["trace"]["model_calls"][0]["cached_prompt_tokens"]
    report = summarize_run(historical)
    assert report["wall_seconds"] is None
    assert report["cached_prompt_tokens"] is None


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True, "2"])
def test_invalid_price_is_rejected(value):
    from deep_research.audit_comparison import summarize_run
    rates = _rates()
    rates["models"]["model-a"]["input"] = value
    with pytest.raises(ValueError, match="rate"):
        summarize_run(_result([_call()]), rates)
