# ScholarTrace

ScholarTrace reads a local research paper to explain its whole method and assess
its contribution, evidence and limitations. It is a paper-reading research prototype: it does not search the
web during its internal audit, rank papers for personal use, or replace peer
review.

## Suggested reading

Start with this README for the project and execution flow, then read
[Architecture](docs/ARCHITECTURE.md) for role responsibilities and context boundaries.
The [paper-reading contract](docs/PAPER_READING.md) explains method reports,
Rubric content management and the latest documented ReMe observation.

For design evidence, the [case study](docs/CASE_STUDY.md) explains the earlier
iterations and tradeoffs; the [HarnessBank comparison](docs/HARNESSBANK_VERSION_COMPARISON.md)
summarizes the fixed-state Reflection experiment and its limits. The
[cache comparison](docs/PROMPT_CACHE_COMPARISON.md) separates observed prefix
reuse from unverified speed and billing benefits. Dated historical sections
describe the implementation at that time, not the current configuration.

## What it does

The supported audit uses five bounded roles:

- **Master** decides which paper-internal questions improve method understanding or evidence assessment.
- **Locator** selects the smallest relevant page range.
- **Evidence** records method details, findings and caveats from those pages.
- **Reflection** reports all distinct issues grounded in its supplied evidence
  that could affect method understanding or the paper's evaluation, including mechanism explanations,
  cross-Worker qualifications and evidence-scope mismatches. It orders issues
  by impact without limiting the memo to one issue or replacing evidence work.
- **Synthesis** produces a sourced method explanation alongside an evidence-grounded final judgment.

New supported runs explain the problem, core idea, representations and modules,
complete workflow, essential details and operating conditions, plus a worked
example when evidence allows. The twelve rubric directions guide reading and
evaluation; coverage is not a grade. The method explanation cites Worker findings
and distinguishes reported facts, inference, constructed illustrations and gaps.
See the [reading report contract](docs/PAPER_READING.md). Historical experiments
retain their original outputs; they are not retrofitted with method explanations.
The local interview showcase is not distributed in this repository.

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
two-page overview, later Master calls get incremental structured state including
each finding's complete Worker evidence items and source locators,
Evidence sees only selected pages, and Synthesis writes from the audit history
without rereading the full paper. Web evidence is isolated in a separate,
post-hoc workflow so it cannot silently change the paper-only trace.

## Engineering highlights

- Rubric IDs connect bounded tasks to their findings through an ID-only context
  index. Master uses it to select related evidence, Workers receive task-specific
  guidance, and Reflection/Synthesis compare reports without copying evidence into
  each rubric bucket. These are routing hints, not scores or proof of coverage.
- A separate traceable Rubric content layer organizes actual Worker findings,
  sourced Reflection notes and Master interpretations. Corrections preserve old
  entries and their reasons. Unassigned findings remain visible; final reports
  stay coherent rather than becoming twelve per-dimension reports.
- Master can select up to three relevant prior findings for complementary source
  reading. Workers retain the evidence actually used as traceable dependencies;
  independent rereading omits this background. Persistent key details help
  Synthesis account for acquired parameters, conditions and update rules.
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
the broader input for comparison. The first Reflection receives the complete
accumulated structured history, including Worker evidence. Every Reflection
reports all evaluation-relevant issues it identifies in its actual input.
Master's requested focus is a starting point, not an exclusive topic restriction;
the later union input still limits which evidence the Reflector can inspect.
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

The current reading revision uses `master-v30-complementary-evidence`,
`locator-v9-reading-rubric`, `evidence-v18-traceable-context`,
`reflection-v15-unconfirmed-scope` and `synthesis-v17-advisory-details` in new
manifests, with `rubric_version=paper-reading-12-v1` and
`rubric_content_version=rubric-content-v3-advisory`.
The v20 controller and Reflection input policies remain in use.
Master now receives Worker evidence as well as summaries and caveats. Low-level
callers should migrate `master_context_mode="incremental-no-raw-evidence"` to
`"incremental-with-evidence"`; the old name is rejected. The public audit selects
the updated mode automatically. Earlier experiments predate these revisions;
the [September 19 ReMe observation](docs/PAPER_READING.md#september-19-reme-observation)
documents one current run, not a controlled quality or cost improvement. Locator gets bounded
question-specific snippets and page-internal headings/captions as navigation
hints. Worker instructions prioritize author-reported qualifications alongside
the answer and preserve the inspected scope of missing-detail claims. Master
separately sees IDs added since the last successful Reflection and IDs never
supplied to a successful Reflection, then compares new evidence with memo leads.
These attention aids do not force a second Reflection or certify verification.

The [HarnessBank comparison](docs/HARNESSBANK_VERSION_COMPARISON.md) includes a
separate four-input Reflection experiment: v20 used 46.48% fewer total tokens
than full history in that one fixed case while preserving the main judgment.
Full history also performed well. This is evidence of context efficiency in
one example, not a stable improvement in capability or end-to-end performance.

For a prompt-cache experiment, keep the default `--prompt-layout standard` or
select `--prompt-layout cache-friendly`. The latter now sends explicit cache
parameters and marks the ends of the role system prompt and reusable JSON prefix
with cache breakpoints. The current `explicit-breakpoints-v3` layout keeps phase
and task-context instructions outside the reusable prefix; dynamic state and
images also follow those boundaries. Historical v2 experiments retain their
original prompts and measurements.
`standard` sends no cache controls; it is not a cache-off switch. Both make fresh
model requests. Gateway support and actual hits still require measurement. This is independent of
Reflection context and does not change the v20 controller. New results record
cache read/write counters when returned and audit wall time. An offline tool
compares saved runs with optional explicit prices; see
[prompt layout comparison](docs/PROMPT_CACHE_COMPARISON.md) for commands,
missing-data handling and quality checks. AiHubMix probes and one H-Mem run
reported prefix reuse; a reliable speed benefit and actual billed savings remain
unverified.

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
- `paper_understanding.py`, `rubric_content.py` and `rubric_details.py` manage
  method-review contracts, traceable interpretations and persistent detail accounting.
- `audit_comparison.py` compares saved runs offline, including cache usage and
  optional cost estimates.
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
