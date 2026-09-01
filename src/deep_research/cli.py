"""Command line interface for the supported ScholarTrace workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .experiment import AuditConfig, run_audit
from .external_audit import run_external_audit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "external-audit":
        return run_external_audit(args.source_result, args.output)
    return run_audit(
        args.paper,
        args.output,
        config=AuditConfig(worker_parallelism=args.worker_parallelism),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="scholar-trace")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="Audit one local PDF")
    audit.add_argument("paper", type=Path, metavar="PAPER.pdf")
    audit.add_argument("--output", type=Path, required=True, metavar="OUTPUT_DIR")
    audit.add_argument(
        "--worker-parallelism",
        type=int,
        choices=range(1, 5),
        default=2,
        metavar="{1,2,3,4}",
    )
    external_audit = subparsers.add_parser(
        "external-audit", help="Audit external evidence after a completed paper-only result"
    )
    external_audit.add_argument("source_result", type=Path, metavar="RESULT.json")
    external_audit.add_argument("--output", type=Path, required=True, metavar="OUTPUT_DIR")
    return parser
