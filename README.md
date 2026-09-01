# ScholarTrace

ScholarTrace audits a local research paper as evidence for an implementable
mechanism. It is a paper-reading research prototype: it does not search the
web during its internal audit, rank papers for personal use, or replace peer
review.

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
audit. Set `OPENAI_API_BASE` as well when using an OpenAI-compatible gateway.
The optional external audit also needs `OPENROUTER_API_KEY`. The sample variable
names are in [.env.example](.env.example); do not commit credentials.

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

## Run an external evidence audit

After a completed paper-only audit, an optional separate command can check
bounded external evidence without modifying the original result:

```bash
scholar-trace external-audit data/audits/paper/result.json --output data/audits/paper-external
```

This command uses Terra to select at most three consequential questions, sends
the queries concurrently as separate Grok 4.5 native web-search requests
through OpenRouter, and uses Terra to judge how the cited evidence changes the
paper-only assessment. Grok's cited answers are retained as model-generated
retrieval summaries, not represented as verbatim source text. The command
writes a separate create-once manifest, progress record, and result; the
paper-only source result is never modified.

## Repository layout

- `paper_agent.py` contains the deterministic adaptive loop and trace types.
- `paper_agent_runtime.py` contains role prompts, context construction, and LLM
  adapters.
- `experiment.py` is the supported local-PDF runner.
- `external_audit.py` is the optional post-hoc Planner → Grok → Auditor path.
- `tests/` uses fake clients and transports; it does not spend model or search
  credits.
- [CASE_STUDY.md](docs/CASE_STUDY.md) records why the current boundaries were
  chosen and what the canaries actually showed.

Run the offline suite with:

```bash
pytest -q
```

## Cost and limitations

Both commands make paid model calls. A long PDF audit can take many minutes,
and native search can return large model contexts. Run them only with a
deliberate output directory and budget. The result is an auditable
model-produced analysis, not an independent replication, a claim of paper
quality, or proof of causal mechanism.

The internal audit is limited to extractable local PDF content. The post-hoc
branch can check cited web evidence, but it does not execute experiments or
perform a full repository review. Missing evidence remains unresolved rather
than being converted into a negative fact.

Historical paper PDFs and raw experiment outputs remain in the local development
workspace but are excluded from the public repository. The
[case study](docs/CASE_STUDY.md) follows the project from the first single-pass
baseline through adaptive routing, Reflection, context engineering, and the
HarnessBank transfer.
