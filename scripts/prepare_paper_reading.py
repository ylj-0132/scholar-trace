"""Prepare local paper PDF artifacts for manual ScholarTrace reading."""

from __future__ import annotations

import argparse
from pathlib import Path

from deep_research.paper_reading import prepare_reading_artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--title")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stem")
    parser.add_argument("--deepresearch-source")
    parser.add_argument("--translate", action="store_true")
    args = parser.parse_args()

    paths = prepare_reading_artifacts(
        args.pdf,
        title=args.title,
        output_dir=args.output_dir,
        stem=args.stem,
        translate=args.translate,
        deepresearch_source=args.deepresearch_source,
    )

    print(f"wrote {paths.text}")
    print(f"wrote {paths.reading_pack}")
    if args.translate:
        print(f"wrote {paths.chinese}")
    print(f"wrote {paths.judgment}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
