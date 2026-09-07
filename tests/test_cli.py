from __future__ import annotations

from pathlib import Path

import pytest

from deep_research import cli


def test_help_exposes_only_supported_commands(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith("usage: scholar-trace ")
    assert "{audit,external-audit}" in output
    assert "Search papers" not in output
    assert "Initialize the local SQLite database" not in output


@pytest.mark.parametrize("argv", (["audit"], ["audit", "paper.pdf"]))
def test_audit_requires_paper_and_output(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(argv)

    assert exc_info.value.code == 2


def test_audit_returns_failure_for_missing_pdf(tmp_path: Path) -> None:
    assert cli.main(
        ["audit", str(tmp_path / "missing.pdf"), "--output", str(tmp_path / "output")]
    ) == 1


def test_audit_returns_failure_for_existing_output(tmp_path: Path) -> None:
    paper = tmp_path / "paper.pdf"
    paper.write_bytes(b"%PDF-1.4\n")
    output = tmp_path / "existing-output"
    output.mkdir()

    assert cli.main(["audit", str(paper), "--output", str(output)]) == 1


def test_audit_passes_arguments_to_run_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run_audit(
        paper_path: Path, output_dir: Path, *, config: object
    ) -> int:
        captured.update(
            paper_path=paper_path,
            output_dir=output_dir,
            config=config,
        )
        return 0

    monkeypatch.setattr(cli, "run_audit", fake_run_audit)

    paper = tmp_path / "paper.pdf"
    output = tmp_path / "audit-output"
    assert cli.main(
        [
            "audit",
            str(paper),
            "--output",
            str(output),
            "--worker-parallelism",
            "4",
        ]
    ) == 0
    assert captured == {
        "paper_path": paper,
        "output_dir": output,
        "config": cli.AuditConfig(worker_parallelism=4),
    }


def test_external_audit_passes_paths_to_runner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_run(source_result: Path, output_dir: Path) -> int:
        captured.update(source_result=source_result, output_dir=output_dir)
        return 0

    monkeypatch.setattr(cli, "run_external_audit", fake_run)
    source = tmp_path / "result.json"
    output = tmp_path / "external"

    assert cli.main(["external-audit", str(source), "--output", str(output)]) == 0
    assert captured == {
        "source_result": source,
        "output_dir": output,
    }


@pytest.mark.parametrize("mode", ["full-history", "rubric-union"])
def test_audit_reflection_context_reaches_runner(monkeypatch, mode: str) -> None:
    received = []
    def run(paper, output, *, config):
        received.append(config.reflection_context_mode)
        return 0
    monkeypatch.setattr(cli, "run_audit", run)
    assert cli.main(["audit", "paper.pdf", "--output", "out", "--reflection-context", mode]) == 0
    assert received == [mode]


@pytest.mark.parametrize("mode", ["rubric-focused", "rubric-routing"])
def test_audit_rejects_retired_context_names_before_running(monkeypatch, mode):
    monkeypatch.setattr(cli, "run_audit", lambda *a, **kw: pytest.fail("must not run"))
    with pytest.raises(SystemExit) as exc:
        cli.main(["audit", "paper.pdf", "--output", "out", "--reflection-context", mode])
    assert exc.value.code == 2


def test_audit_default_context_is_rubric_union():
    args = cli.build_parser().parse_args(["audit", "paper.pdf", "--output", "out"])
    assert args.reflection_context == cli.AuditConfig().reflection_context_mode == "rubric-union"


def test_external_audit_rejects_experimental_backend_flags() -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main([
            "external-audit", "result.json", "--output", "external", "--backend", "grok-native",
        ])
    assert exc_info.value.code == 2


@pytest.mark.parametrize("worker_parallelism", (0, 5))
def test_audit_rejects_worker_parallelism_outside_supported_range(
    worker_parallelism: int,
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(
            [
                "audit",
                "paper.pdf",
                "--output",
                "audit-output",
                "--worker-parallelism",
                str(worker_parallelism),
            ]
        )

    assert exc_info.value.code == 2
