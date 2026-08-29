# ScholarTrace

ScholarTrace audits a local research paper as evidence for an implementable
mechanism. It is a paper-reading research prototype: it does not search the
web, rank papers for personal use, or replace peer review.

## What it does

The supported audit uses five bounded roles:

- **Master** decides which paper-internal questions still matter.
- **Locator** selects the smallest relevant page range.
- **Evidence** records findings and caveats from those pages.
- **Reflection** identifies a consequential unresolved mechanism question.
- **Synthesis** produces one evidence-grounded final judgment.

The default configuration uses Terra for Master and Reflection, and Luna for
Locator, Evidence, and Synthesis. See [Architecture](docs/ARCHITECTURE.md) for
the execution flow and context boundaries.

## Install

Requires Python 3.11 or newer.

```bash
python -m pip install -e ".[dev]"
```

Set `OPENAI_API_KEY` in a local `.env` file or in the environment before a real
audit. The sample variable names are in [.env.example](.env.example); do not
commit credentials.

## Run one audit

```bash
scholar-trace audit paper/example.pdf --output data/audits/example
```

`--output` is create-once: the command refuses an existing directory. The only
optional tuning flag is `--worker-parallelism {1,2,3,4}` (default `2`).

An audit writes:

- `manifest.json` — frozen models, prompt-version labels, input hash, and
  configuration;
- `progress.json` — current terminal-safe progress status;
- `result.json` — final trace or a safe failure record.

## Cost and limitations

This command makes paid model calls and can take many minutes for a long PDF.
Run it only with a deliberate output directory and budget. The result is an
auditable model-produced analysis, not an independent replication, a claim of
paper quality, or proof of causal mechanism. It is limited to extractable local
PDF content; missing reporting remains unresolved rather than being filled by
external search.

Historical paper PDFs and raw experiment outputs remain in the local development
workspace but are excluded from the public repository. The
[case study](docs/CASE_STUDY.md) follows the project from the first single-pass
baseline through adaptive routing, Reflection, context engineering, and the
HarnessBank transfer.
