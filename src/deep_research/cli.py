"""Command line interface for the supported local-PDF audit."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from .experiment import AuditConfig, run_audit


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
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
    return parser
