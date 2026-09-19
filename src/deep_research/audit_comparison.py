"""Offline observations from two saved audits; never calls a model or loads credentials."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

TOKEN_FIELDS = (
    "prompt_tokens", "completion_tokens", "total_tokens",
    "cached_prompt_tokens", "cache_write_prompt_tokens",
)
CONTROL_FIELDS = (
    "paper_sha256", "role_models", "prompt_versions", "source_sha256",
    "worker_parallelism", "reflection_context_mode", "worker_context_mode",
    "worker_role_mode", "master_context_mode", "paper_context_mode",
    "adaptive_max_rounds", "max_reflections", "investigation_target",
    "reflection_policy", "action_policy", "prompt_layout_version",
)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _tokens(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _validate_rates(rates: dict[str, Any] | None) -> None:
    if rates is None:
        return
    if not isinstance(rates, dict) or not rates.get("currency") or not isinstance(rates.get("models"), dict):
        raise ValueError("rates require currency and a models object")
    for prices in rates["models"].values():
        if not isinstance(prices, dict) or any(
            not _number(prices.get(key)) for key in ("input", "cached_input", "cache_write", "output")
        ):
            raise ValueError("each model rate must provide four finite nonnegative per-million prices")


def _cost(call: dict[str, Any], rates: dict[str, Any] | None) -> float | None:
    prices = rates["models"].get(call.get("model")) if rates else None
    total, output, cached, writes = (call.get(key) for key in (
        "prompt_tokens", "completion_tokens", "cached_prompt_tokens", "cache_write_prompt_tokens",
    ))
    if not prices or not all(_tokens(value) for value in (total, output, cached)):
        return None
    if writes is None and prices["cache_write"] == prices["input"]:
        # The unknown split has no effect on this explicitly supplied tariff.
        writes = 0
    if not _tokens(writes) or cached + writes > total:
        return None
    return ((total - cached - writes) * prices["input"] + cached * prices["cached_input"]
            + writes * prices["cache_write"] + output * prices["output"]) / 1_000_000


def _metrics(calls: list[dict[str, Any]], rates: dict[str, Any] | None) -> dict[str, Any]:
    missing = {key: sum(not _tokens(call.get(key)) for call in calls) for key in TOKEN_FIELDS}
    totals = {key: sum(call[key] for call in calls) if calls and not missing[key] else None
              for key in TOKEN_FIELDS}
    cached, inputs = totals["cached_prompt_tokens"], totals["prompt_tokens"]
    invalid_cache = sum(
        _tokens(call.get("prompt_tokens")) and sum(
            call[key] for key in ("cached_prompt_tokens", "cache_write_prompt_tokens") if _tokens(call.get(key))
        ) > call["prompt_tokens"] for call in calls
    )
    costs = [_cost(call, rates) for call in calls]
    latencies = [call.get("latency_seconds") for call in calls]
    return {
        "logical_calls": len(calls), **totals,
        "missing_usage_calls": missing,
        "cache_hit_fraction": cached / inputs if inputs and cached is not None and not invalid_cache else None,
        "invalid_cache_usage_calls": invalid_cache,
        "calls_with_cache_hits": sum(_tokens(call.get("cached_prompt_tokens")) and call["cached_prompt_tokens"] > 0 for call in calls),
        "calls_without_cache_usage": missing["cached_prompt_tokens"],
        "extra_transport_attempts": sum(max(0, call["attempt_count"] - 1) for call in calls if _tokens(call.get("attempt_count"))),
        "failed_calls": sum(bool(call.get("error")) for call in calls),
        "validation_error_calls": sum(bool(call.get("validation_error")) for call in calls),
        "summed_call_seconds": sum(latencies) if calls and all(_number(value) for value in latencies) else None,
        "estimated_recorded_cost": sum(costs) if calls and all(value is not None for value in costs) else None,
        "unpriced_calls": sum(value is None for value in costs),
    }


def summarize_run(result: dict[str, Any], rates: dict[str, Any] | None = None) -> dict[str, Any]:
    _validate_rates(rates)
    trace = result.get("trace", {})
    calls = trace.get("model_calls", [])
    if not isinstance(calls, list) or any(not isinstance(call, dict) for call in calls):
        raise ValueError("trace.model_calls must be a list of objects")
    summary = _metrics(calls, rates)
    warnings = []
    if summary["extra_transport_attempts"] or summary["failed_calls"]:
        warnings.append("provider_billing_may_include_unobserved_attempts")
    if summary["calls_without_cache_usage"]:
        warnings.append("cache_usage_unknown_for_some_calls")
    if summary["invalid_cache_usage_calls"]:
        warnings.append("inconsistent_cache_usage")
    if rates and summary["unpriced_calls"]:
        warnings.append("cost_unknown_due_to_missing_rates_or_missing_or_invalid_usage")
    if not calls:
        warnings.append("no_recorded_calls")
    judgment = trace.get("final_judgment") or {}
    wall = result.get("wall_seconds")
    return {
        **summary, "wall_seconds": wall if _number(wall) else None,
        "status": result.get("status"), "judgment_status": result.get("judgment_status"),
        "outcome": trace.get("outcome"), "rounds": len(trace.get("steps", [])),
        "reflections": len(trace.get("reflection_reports", [])),
        "final_finding_count": len(judgment.get("key_findings", [])),
        "provenance_status": judgment.get("provenance_status"),
        "provenance_warning_count": len(judgment.get("provenance_warnings", [])),
        "warnings": warnings,
        "by_role": {role: _metrics([call for call in calls if call.get("role", "unknown") == role], rates)
                    for role in sorted({call.get("role", "unknown") for call in calls})},
    }


def _read_json(path: Path) -> tuple[dict[str, Any], str]:
    content = path.read_bytes()
    payload = json.loads(content.decode("utf-8-sig"))
    if not isinstance(payload, dict):
        raise ValueError("expected a JSON object")
    return payload, hashlib.sha256(content).hexdigest()


def compare_runs(baseline: Path, candidate: Path, rates: dict[str, Any] | None = None) -> dict[str, Any]:
    summaries = []
    manifests = []
    settings = []
    for path in (baseline, candidate):
        result, digest = _read_json(path)
        manifest_path = path.parent / "manifest.json"
        manifest, manifest_digest = _read_json(manifest_path) if manifest_path.is_file() else ({}, None)
        manifests.append(manifest)
        settings.append(sorted({(call.get("role") or "unknown", call.get("model") or "unknown", str(call.get("temperature")))
                                for call in result.get("trace", {}).get("model_calls", [])}))
        summaries.append({**summarize_run(result, rates), "prompt_layout": manifest.get("prompt_layout"),
                          "source_result_sha256": digest, "source_manifest_sha256": manifest_digest})
    warnings = [f"manifest_mismatch:{key}" for key in CONTROL_FIELDS
                if manifests[0].get(key) != manifests[1].get(key)]
    warnings.extend(f"missing_control_metadata:{key}" for key in CONTROL_FIELDS
                    if any(manifest.get(key) is None for manifest in manifests))
    if settings[0] != settings[1]:
        warnings.append("actual_role_model_temperature_sets_differ")
    deltas = {}
    for key in ("wall_seconds", "summed_call_seconds", "prompt_tokens", "completion_tokens",
                "total_tokens", "estimated_recorded_cost"):
        before, after = summaries[0][key], summaries[1][key]
        deltas[key] = 100 * (after - before) / before if before and after is not None else None
    return {
        "schema_version": "audit-layout-comparison-v1",
        "baseline": summaries[0], "candidate": summaries[1],
        "rates_per_million": rates, "observed_change_percent": deltas,
        "warnings": warnings, "quality_review_required": True,
        "interpretation": "Observational layout comparison; both runs may use provider caching. Costs cover recorded response usage at supplied rates, not the provider invoice. Parallel call times are not wall time. Review final findings and evidence manually.",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two saved paper-only audits offline (no model calls).")
    parser.add_argument("baseline", type=Path, metavar="BASELINE_RESULT.json")
    parser.add_argument("candidate", type=Path, metavar="CANDIDATE_RESULT.json")
    parser.add_argument("--rates", type=Path, help="JSON with currency and per-model input/cached_input/cache_write/output rates per million tokens")
    args = parser.parse_args(argv)
    try:
        rates = _read_json(args.rates)[0] if args.rates else None
        report = compare_runs(args.baseline, args.candidate, rates)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
