"""Supported create-once local-PDF paper audit entry point."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import llm_api_base, llm_api_key, llm_request_timeout
from .llm import LLMClient
from .paper_agent_runtime import REFLECTION_CONTEXT_MODES, run_local_paper_agent


@dataclass(frozen=True)
class AuditConfig:
    worker_parallelism: int = 2
    reflection_context_mode: str = "rubric-union"


DEFAULT_AUDIT_CONFIG = AuditConfig()

ROLE_MODELS = {
    "master": "openai/gpt-5.6-terra",
    "reflection": "openai/gpt-5.6-terra",
    "locator": "openai/gpt-5.6-luna",
    "evidence": "openai/gpt-5.6-luna",
    "synthesis": "openai/gpt-5.6-luna",
}
PROMPT_VERSIONS = {
    "master": "master-v20-restored-convergence",
    "locator": "locator-v7-rubric-routing",
    "reflection": "reflection-v10-independent-request",
    "evidence": "evidence-v10-rubric-scope",
    "synthesis": "synthesis-v11-rubric-reconciliation",
}
INVESTIGATION_TARGET = (
    "Audit the paper's own research problem, claimed contributions, mechanisms and "
    "experimental evidence. Reconstruct the information flow, component roles, runtime "
    "access paths, operational bounds and costs relevant to those claims. Determine which claims are "
    "directly supported by the method, appendices, ablations, and held-out experiments; "
    "identify missing implementation details, experimental confounders, and inferences "
    "beyond the evidence. Distinguish paper facts, reasonable inferences, and information "
    "the paper does not specify. Investigate cross-task learning or continual improvement "
    "only when the paper explicitly claims those capabilities; do not require persistent "
    "learning, credit assignment or other unrelated engineering features merely to fill a rubric."
)

_UNSAFE_PATH_RE = re.compile(r"(?:(?<![A-Za-z0-9])[A-Za-z]:[\\/]+|file://)", re.IGNORECASE)
_POSIX_ABSOLUTE_PATH_RE = re.compile(r"(?<![A-Za-z0-9:/])/[^/\s]+(?:/[^/\s]+)+")
_UNC_PATH_RE = re.compile(r"\\\\[^\\/\s]+\\")
_BARE_BASE64_RE = re.compile(r"(?<![A-Za-z0-9+/])[A-Za-z0-9+/]{256,}={0,2}(?![A-Za-z0-9+/=])")
_IMAGE_BASE64_RE = re.compile(
    r"(?:iVBORw0KGgo|/9j/|R0lGOD|Qk|UklGR|SUkq|TU0AK)[A-Za-z0-9+/]{8,}={0,2}"
)
_DATA_IMAGE_RE = re.compile(
    r"data:image/[^;,\s]+;base64,[A-Za-z0-9+/]+={0,2}", re.IGNORECASE
)


def run_audit(
    paper_path: Path,
    output_dir: Path,
    *,
    config: AuditConfig = DEFAULT_AUDIT_CONFIG,
    client: object | None = None,
) -> int:
    """Run one create-once adaptive local-PDF audit."""

    paper_path = Path(paper_path)
    output_dir = Path(output_dir)
    if not paper_path.is_file() or paper_path.suffix.casefold() != ".pdf":
        return 1
    if not 1 <= config.worker_parallelism <= 4 or output_dir.exists():
        return 1
    if config.reflection_context_mode not in REFLECTION_CONTEXT_MODES:
        return 1

    try:
        output_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return 1

    api_keys: tuple[str, ...] = ()
    try:
        role_llms, api_keys = _role_clients(client)
        manifest = _manifest(paper_path, config)
        _write_new_json(output_dir / "manifest.json", manifest)
        _write_json(output_dir / "progress.json", _progress("running"))
        trace = run_local_paper_agent(
            pdf_path=paper_path,
            llm=role_llms["master"],
            role_llms=role_llms,
            max_rounds=5,
            worker_parallelism=config.worker_parallelism,
            max_reflections=2,
            reflection_context_mode=config.reflection_context_mode,
            worker_context_mode="selected-context",
            worker_role_mode="legacy",
            master_context_mode="incremental-no-raw-evidence",
            paper_context_mode="master-overview-history-only",
            investigation_target=INVESTIGATION_TARGET,
        )
        trace_payload = json.loads(trace.to_json())
        result = {
            "status": "completed",
            "judgment_status": _judgment_status(trace),
            "paper": paper_path.name,
            "trace": trace_payload,
        }
        sanitized_result, serialization_warnings = _sanitize_audit_payload(
            result, paper_path.parent, api_keys
        )
        assert isinstance(sanitized_result, dict)
        result = sanitized_result
        result["serialization_warnings"] = serialization_warnings
    except Exception as exc:
        error = _safe_error(str(exc), paper_path.parent, api_keys)
        _write_new_json(
            output_dir / "result.json",
            {"status": "failed", "paper": paper_path.name, "error": error},
        )
        _write_json(output_dir / "progress.json", _progress("failed"))
        _write_json(output_dir / "manifest.json", {**_manifest(paper_path, config), "status": "failed"})
        return 1

    _write_new_json(output_dir / "result.json", result)
    _write_json(output_dir / "progress.json", _progress("completed"))
    _write_json(output_dir / "manifest.json", {**manifest, "status": "completed"})
    return 0


def _role_clients(client: object | None) -> tuple[dict[str, object], tuple[str, ...]]:
    if client is not None:
        return {role: client for role in ROLE_MODELS}, ()
    clients: dict[str, object] = {}
    keys: list[str] = []
    for role, model in ROLE_MODELS.items():
        api_key = llm_api_key()
        if api_key:
            keys.append(api_key)
        clients[role] = LLMClient(
            model,
            api_key=api_key,
            api_base=llm_api_base(),
            request_timeout=llm_request_timeout(),
            max_retries=1,
            temperature=1.0,
            allow_json_repair=False,
        )
    return clients, tuple(keys)


def _manifest(paper_path: Path, config: AuditConfig) -> dict[str, object]:
    return {
        "schema_version": "local-audit-v1",
        "status": "pending",
        "paper": paper_path.name,
        "role_models": dict(ROLE_MODELS),
        "prompt_versions": dict(PROMPT_VERSIONS),
        "paper_sha256": _sha256(paper_path),
        "adaptive_max_rounds": 5,
        "max_reflections": 2,
        "reflection_policy": "post-method-model-and-master-requested",
        "action_policy": "read-reflect-independent-batch-v1",
        "investigation_target": INVESTIGATION_TARGET,
        "reflection_context_mode": config.reflection_context_mode,
        "finding_provenance": "worker-id-dispositions-v1",
        "worker_parallelism": config.worker_parallelism,
        "worker_context_mode": "selected-context",
        "worker_role_mode": "legacy",
        "master_context_mode": "incremental-no-raw-evidence",
        "paper_context_mode": "master-overview-history-only",
        "source_sha256": {
            relative_path: _sha256(path)
            for relative_path, path in _reproducibility_sources().items()
        },
    }


def _progress(status: str) -> dict[str, str]:
    return {"schema_version": "local-audit-progress-v1", "status": status, "updated_at": datetime.now(timezone.utc).isoformat()}


def _judgment_status(trace: object) -> str:
    if getattr(trace, "error", None):
        return "unavailable"
    if getattr(trace, "outcome", None) == "DECIDE" and getattr(trace, "assessment", None):
        return "decided"
    if getattr(trace, "outcome", None) == "NEEDS_HUMAN":
        return "needs_human"
    return "unavailable"


def _sanitize_audit_payload(
    payload: object, project_root: Path, api_keys: tuple[str, ...]
) -> tuple[object, list[str]]:
    warnings: set[str] = set()
    root = str(project_root.resolve())

    def sanitize_text(text: str) -> str:
        for root_variant in {root, root.replace("\\", "/")}:
            redacted = re.sub(
                re.escape(root_variant), "<PAPER_ROOT>", text, flags=re.IGNORECASE
            )
            if redacted != text:
                text = redacted
                warnings.add("paper_root_redacted")
        for api_key in api_keys:
            if api_key and api_key in text:
                text = text.replace(api_key, "[REDACTED]")
                warnings.add("api_key_redacted")
        redacted = _DATA_IMAGE_RE.sub("[REDACTED_IMAGE_DATA]", text)
        redacted = _IMAGE_BASE64_RE.sub("[REDACTED_IMAGE_DATA]", redacted)
        if redacted != text:
            warnings.add("image_data_redacted")
        return redacted

    def sanitize(value: object) -> object:
        if isinstance(value, str):
            return sanitize_text(value)
        if isinstance(value, dict):
            return {sanitize_text(str(key)): sanitize(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [sanitize(child) for child in value]
        return value

    return sanitize(payload), sorted(warnings)


def _safe_error(text: str, project_root: Path, api_keys: tuple[str, ...]) -> str:
    root = str(project_root.resolve())
    if (
        "data:image" in text.casefold()
        or root in text
        or _has_unsafe_text(text)
        or any(key in text for key in api_keys)
    ):
        return "unsafe error content [REDACTED]"
    if text == "unsafe trace serialization":
        return "unsafe error content [REDACTED]"
    return text or "audit failed"


def _has_unsafe_content(value: object) -> bool:
    if isinstance(value, dict):
        return any(_has_unsafe_text(key) or _has_unsafe_content(child) for key, child in value.items())
    if isinstance(value, (list, tuple)):
        return any(_has_unsafe_content(child) for child in value)
    return isinstance(value, str) and _has_unsafe_text(value)


def _has_unsafe_text(text: str) -> bool:
    return bool(
        _UNSAFE_PATH_RE.search(text)
        or _POSIX_ABSOLUTE_PATH_RE.search(text)
        or _UNC_PATH_RE.search(text)
        or _BARE_BASE64_RE.search(text)
        or _IMAGE_BASE64_RE.search(text)
    )


def _reproducibility_sources() -> dict[str, Path]:
    package_dir = Path(__file__).parent
    return {
        "src/deep_research/experiment.py": package_dir / "experiment.py",
        "src/deep_research/paper_agent_runtime.py": package_dir / "paper_agent_runtime.py",
        "src/deep_research/paper_agent.py": package_dir / "paper_agent.py",
        "src/deep_research/paper_reading.py": package_dir / "paper_reading.py",
        "src/deep_research/llm.py": package_dir / "llm.py",
        "src/deep_research/config.py": package_dir / "config.py",
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_new_json(path: Path, payload: dict[str, object]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    os.replace(temporary, path)
