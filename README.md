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
  It also checks material cross-Worker contradictions and evidence-scope
  mismatches within the requested review scope; it does not replace evidence work.
- **Synthesis** produces one evidence-grounded final judgment.

The default configuration uses Terra for Master and Reflection, and Luna for
Locator, Evidence, and Synthesis. See [Architecture](docs/ARCHITECTURE.md) for
the execution flow and context boundaries.

## Architecture at a glance

```text
local PDF
  -> Master
  -> Locator -> Evidence (bounded questions, parallel when independent)
  -> automatic first Reflection
  -> Master: READ_PAPER / REFLECT / READ_PAPER_AND_REFLECT -> join -> Master
  -> DECIDE -> Synthesis -> FinalJudgment
                   -> optional external audit
                      -> Terra Planner -> Grok web search -> Terra Auditor
```

The controller is deterministic even though the role outputs are model
generated. Context ownership is asymmetric: the first Master call gets a
two-page overview, later Master calls get incremental structured state,
Evidence sees only selected pages, and Synthesis writes from the audit history
without rereading the full paper. Web evidence is isolated in a separate,
post-hoc workflow so it cannot silently change the paper-only trace.

## Engineering highlights

- Rubric IDs connect bounded tasks to their findings through an ID-only context
  index. Master uses it to select related evidence, Workers receive task-specific
  guidance, and Reflection/Synthesis compare reports without copying evidence into
  each rubric bucket. These are routing hints, not scores or proof of coverage.
- The adaptive trace records every question, selected page range, finding,
  caveat, stopping decision, model call, and token count for inspection.
- Role/model routing spends the stronger model on planning and Reflection while
  using the smaller model for navigation, local evidence work, and synthesis.
- In one ReMe context-ownership canary, prompt tokens fell **61.74%**, total
  tokens **59.69%**, logical calls **28.57%**, and wall time **45.22%** relative
  to the original transfer run. These are observed single-run results, not
  benchmark medians; the full experimental history and limitations are in the
  [case study](docs/CASE_STUDY.md#removing-the-final-full-paper-safety-net).
- Create-once outputs freeze input and source hashes. Known local paths,
  configured API keys, and image payloads are redacted from completed results
  with explicit serialization warnings instead of discarding the audit.

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

`--output` is create-once: the command refuses an existing directory.
`--worker-parallelism {1,2,3,4}` defaults to `2`. The optional second Reflection
uses `--reflection-context rubric-union` by default; `full-history` retains
the broader input for comparison. The first Reflection keeps its original
trigger, input scope and mechanism reasoning, with additive consistency guidance.
Master can request a valuable check of existing evidence without new findings or
a previously identified contradiction. The runtime assembles matching rubric
findings plus optional cross-rubric anchors; it does not force a second review.
Master can choose `REFLECT` independently or explicitly choose
`READ_PAPER_AND_REFLECT` to run independent reading and review in the same round.
Both branches use prior evidence and join before the next Master decision;
dependent work must be requested sequentially. `DECIDE` is exclusive. Each action
batch consumes a round, within the existing five-round/two-Reflection limits.
The default target follows the paper's own claims, with no presumed self-evolution
requirement. Old decision-attached reflection fields are no longer accepted.
Offline tests verify these contracts, not an improvement in judgment quality.

This is the formal **v20** mechanism. To select the full-history comparison explicitly:

```bash
scholar-trace audit paper/example.pdf --output data/audits/example-full --reflection-context full-history
```

Python callers use `AuditConfig(reflection_context_mode="full-history")`; omitting
the field selects `rubric-union`. Migrate the former v20 value `rubric-focused`
to `rubric-union` in CLI commands and Python configuration. The ambiguous old
name is rejected, not retained as an alias. This renames the existing union
behavior; the older anchor-only focused/routing implementations are removed.
Historical manifests and sent prompts keep their original names and hashes.
See [formalization and cleanup](docs/V20_FORMALIZATION.md).

The [HarnessBank comparison](docs/HARNESSBANK_VERSION_COMPARISON.md) includes a
separate four-input Reflection experiment: v20 used 46.48% fewer total tokens
than full history in that one fixed case while preserving the main judgment.
Full history also performed well. This is evidence of context efficiency in
one example, not a stable improvement in capability or end-to-end performance.

An audit writes:

- `manifest.json` — frozen models, prompt-version labels, input hash, and
  configuration;
- `progress.json` — current terminal-safe progress status;
- `result.json` — final trace or a safe failure record.

Final findings reference their upstream Worker finding IDs. A disposition ledger
records whether each finding was retained, merged, corrected, left unresolved,
or discarded with a reason. Missing or inconsistent links are preserved as
`provenance_warnings`; they do not discard the completed judgment. A `complete`
provenance status means structural accounting passed, not that the conclusions
have been independently verified. See [Architecture](docs/ARCHITECTURE.md).

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
