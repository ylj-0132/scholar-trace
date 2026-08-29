from __future__ import annotations

from pathlib import Path

import pytest

from deep_research import cli


def test_help_exposes_only_audit_command(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli.main(["--help"])

    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    assert output.startswith("usage: scholar-trace ")
    assert "{audit}" in output
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
