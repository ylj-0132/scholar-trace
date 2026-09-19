# Architecture

ScholarTrace performs one create-once, paper-only adaptive reading and audit. Its supported
entry points are `scholar-trace audit PAPER.pdf --output OUTPUT_DIR` and the
post-hoc `scholar-trace external-audit RESULT.json --output OUTPUT_DIR`.

The formal default is v20, with `reflection_context_mode="rubric-union"` in
`AuditConfig`, `PaperReflector`, and `run_local_paper_agent`. The only other
Reflection context mode is explicit `full-history`. Migrate the former v20
configuration value `rubric-focused` to `rubric-union`; it is not an alias.
This does not restore either retired anchor-only implementation. New reports
use `rubric-union` and `rubric_union_context_fallback:*`; historical context
names and diagnostics remain unchanged in saved traces. Current manifests record
`master-v30-complementary-evidence`, `locator-v9-reading-rubric`,
`evidence-v18-traceable-context`, `reflection-v15-unconfirmed-scope` and
`synthesis-v17-advisory-details`, with source hashes, the
`paper-reading-12-v1` rubric version and `rubric-content-v3-advisory` content version.
These revisions preserve the v20 controller and Reflection input policies,
while retaining Worker evidence in the incremental Master input and adding
an independent sourced method explanation to the final report.
Historical runs retain their original prompts and version labels. The
[September 19 ReMe run](PAPER_READING.md#september-19-reme-observation) exercised
this revision; it is not a controlled comparison of individual changes.
Other low-level experimental context defaults are unchanged.

## Execution flow

```text
local PDF -> Master
  -> READ_PAPER: Locator -> Evidence Worker (bounded questions in parallel)
     -> automatic first mechanism Reflection
  -> Master chooses READ_PAPER, REFLECT, or READ_PAPER_AND_REFLECT
     -> join completed results -> Master (repeat while valuable)
  -> exclusive DECIDE -> Synthesis -> FinalJudgment
                   -> external-audit (optional, separate create-once output)
                      -> Terra Planner -> Grok native web search -> Terra Auditor
                      -> external judgment delta
```

The Master owns the global investigation state and chooses new evidence,
reasoning over existing evidence, both, or a supported decision. A `READ_PAPER` action can dispatch independent
questions in parallel. Locator navigates a compact page index; Evidence reads
only the selected page text and corresponding page images. Evidence reports
both findings and caveats, so missing implementation details remain visible.

Reflection is not a source of paper facts. It reviews its supplied structured
state and reports all distinct evidence-grounded issues that could affect method
understanding or the paper's evaluation. Master accounts for each issue by adopting it, requesting a
bounded check, retaining uncertainty, or rejecting it with a reason.
Synthesis is the sole final-writing step and turns the structured history
into a `FinalJudgment`, including `method_understanding` alongside the assessment.
Method gaps can justify reading or reflection even when evaluation is unchanged;
they do not introduce a mandatory phase, additional rounds or a second synthesis.
The [reading contract](PAPER_READING.md) describes the twelve dimensions and output.

`REFLECT` is an independent action, not a field attached to `DECIDE`.
`READ_PAPER_AND_REFLECT` explicitly requests both actions in one round, with an
`independence_rationale`. Both branches use the same pre-round snapshot and run
concurrently; neither receives the other's results until they join and return to
Master. If Reflection needs new Worker evidence, or Workers need Reflection's
guidance, Master must request the prerequisite alone and reconsider afterward.
The rationale declares independence; code guarantees snapshot isolation, not the
semantic validity of that declaration. Worker parallelism still bounds Workers;
the combined batch may additionally run one Reflector call.

`DECIDE` and `NEEDS_HUMAN` cannot contain reading tasks or reflection requests.
Each explicit action batch, including standalone Reflection, consumes one of the
five rounds. Master must leave a turn to integrate results and decide. The automatic
first review and at most two total Reflection reports remain; a requested review
requires prior evidence, the first review and unused budget. Old `pre_decide_*`
model-output fields are rejected. Historical traces remain untouched; the legacy
`deferred_decisions` trace field remains present but is empty in new runs.

The default investigation target follows the paper's own problem, mechanism and
experimental claims. Cross-task learning and continual improvement are investigated
only when the paper explicitly claims them, not presumed of every harness.

The optional external branch starts only after `FinalJudgment` has been written
and reads that completed result as input. It never modifies the paper-only
result, and the Master never sees web evidence. Terra selects at most three
questions that could change the experimental interpretation. Grok 4.5 searches
the fixed questions concurrently in separate, non-schema OpenRouter requests
without seeing or judging the paper-only result. The runtime records URL
citations, bounded model-generated retrieval summaries, and OpenRouter's
server-tool routing metadata. A final Terra call receives that normalized
evidence and records the judgment delta. Retrieval summaries are explicitly
not treated as verbatim source text. Missing citations or a failed individual
query remain diagnostics rather than discarding the completed analysis.

## Role routing

| Role | Default model family | Responsibility |
| --- | --- | --- |
| Master | Terra | investigation and stopping decisions |
| Reflection | Terra | bounded reasoning over the audit state |
| Locator | Luna | page-range selection |
| Evidence | Luna | page-grounded findings and caveats |
| Synthesis | Luna | final structured judgment |

## Context ownership

The default profile is deliberately asymmetric to avoid repeatedly sending a
whole paper to every role.

- The first Master call receives the first two pages as overview text, a compact
  page index, and image inputs for those two pages.
- Later Master calls retain the compact index and an incremental state: task
  questions, page ranges, statuses, finding identifiers, findings, caveats, the
  latest Reflection, assessment, unresolved questions, checklist state, and
  remaining rounds. Each finding also includes all of its `evidence_items`,
  preserving each item's content, type and locator in order. Text items are
  Worker-supplied quotations; table/figure items are Worker descriptions, not
  direct image inspection by Master. The primary quote is not duplicated in a
  separate `evidence` field. Legacy findings retain their single evidence item.
  Later calls omit the overview and images, and do not receive full-paper text.
  Follow-up suggestions include all Workers from the latest reading batch in task order;
  earlier suggestions remain in the full trace rather than being repeated.
  A reflection-only round does not erase those suggestions. Requested reflection
  focus/selection and independence rationale are retained in action summaries and
  with the latest requested memo. Remaining Reflection budget is supplied explicitly.
- Master state starts with `finding_review_status`: `new_since_last_successful_reflection`
  compares current IDs with the last successful Reflection's observed pre-batch
  snapshot; `never_supplied_to_successful_reflection` excludes only IDs actually
  supplied across successful Reflection inputs. Failed or empty memos do not count.
  Legacy full-history reports without explicit context IDs use their snapshot IDs;
  rubric-union snapshots never substitute for the actual selected context.
  Thus an older finding omitted by rubric selection can remain never supplied,
  while findings from a parallel reading branch remain new after that Reflection.
  These lists do not certify reasoning quality, override budgets or trigger review.
  Master instructions inspect new evidence before memo suggestions and compare
  independent directions, all latest-batch Worker suggestions and memo-derived
  leads by their expected contribution to the assessment. Existing-evidence
  integration and missing-source reading remain separate choices.
- Locator receives the compact index, one bounded evidence question, and up to
  six lexical navigation snippets of at most 240 characters each. Whole-page
  heading/caption extraction recognizes numbered subsections and lettered table
  or figure labels, including headings below the page preview. Each page exposes
  at most 12 headings and 12 captions. Snippets rank distinct question-term matches
  deterministically; they are incomplete English lexical hints, not semantic
  retrieval, evidence, or proof of absence. Named sources and cross-page
  continuations remain part of the Locator's selection instructions.
- Evidence receives only the Locator-selected pages and their images, plus its
  bounded question and decision relevance. It does not receive navigation snippets.
  Its prompt prioritizes the direct answer together with author-reported controls,
  sensitivity results, qualifications and counterevidence, then other material
  local issues. Existing evidence arrays retain separate passages; `caveat` states
  the inspected scope and any unread reference or continuation. Explicit source
  cross-references can link component roles; proximity cannot establish shared
  models/settings, and an alias does not establish an exact revision.
- Master may select up to three relevant historical findings with evidence and
  caveats for complementary reading, without requiring a known contradiction.
  Content associations help find candidates but do not inject a whole dimension.
  Worker findings record only supplied prior IDs actually used in their reasoning.
  These dependencies do not expand Reflection's input slice. Independent reading
  omits historical findings and decision background from both Locator and Worker.
- The first Reflection receives complete structured state, including raw Worker
  evidence, but not the overview or compact index.
- For the optional second (`master_requested`) Reflection, Master names the valuable
  review purpose: proposed conclusions, unchecked evidence relationships,
  and how verification could change their boundaries. An already-identified conflict
  is not required. Master selects one or two rubric IDs and zero to six optional
  finding anchors. In default `rubric-union` mode, the runtime sends the union
  of all findings with matching task rubric labels and the explicit anchors,
  including their evidence and caveats, selected guidance/coverage and remaining rounds.
  Unrelated history, assessments, questions, and old Reflection memos are omitted
  from the actual payload. An omitted detail is not evidence of absence.
  The named focus motivates review and routes the input; it does not restrict
  which evaluation-relevant issues the Reflector may report from that input.
- Synthesis receives complete structured history and Worker evidence, but not
  full-paper text, the compact index, or images.

The execution trace retains the full recorded state even when a role receives a
smaller operational payload.

The public audit uses `master_context_mode="incremental-with-evidence"`.
Low-level callers must replace the former `incremental-no-raw-evidence` value;
it is rejected rather than silently acquiring different semantics. Master
`full-history` remains supported. The experimental main-text profile also uses
the evidence-bearing incremental projection after its first turn. Historical
manifests and recorded prompts remain unchanged. Evidence inclusion increases
Master input size; its effect on model judgment, tokens and latency is unmeasured.

`--reflection-context full-history` keeps the broader second-Reflection input
for comparisons. Both modes share the updated Master selection instructions and
the same stopping/Reflection triggers; neither forces a second Reflection.
First-round Reflection retains its trigger, full-state scope and mechanism
reasoning, but no longer selects only one issue or one follow-up investigation.
Both triggers request all distinct evaluation-relevant issues visible in their
input, with finding IDs, reasoning and judgment implications. The memo remains
one prose string with separate paragraphs as needed, without a fixed issue count.
Reports retain `context_mode`, `context_finding_ids`, `context_rubric_ids`, and
`context_diagnostics`, separately from the controller's `reflected_finding_ids`.
An empty assembled slice or IDs absent from the successful Worker history trigger an
explicit `rubric_union_context_fallback:*` diagnostic and a full-history call. Such
runs must not be counted as successful rubric-union trials. Malformed selector
types, duplicate IDs, unknown rubric keys, and over-limit selections are rejected
by the Master-output validation path, as are missing rubric selections. The low-level
Reflector also retains a missing-selection fallback for direct adapter callers.

## Rubric content management and routing

`rubric_content` is a deterministic projection of the investigation trace. Worker
findings explicitly propose content associations; Reflection supplies sourced
candidate notes; Master maintains current interpretations, qualifications and
questions, including explicit corrections and association revisions. Historical
entries are retained rather than overwritten, with source IDs and reasons.
Unassigned findings remain available alongside the complete original evidence.

Master and Synthesis receive the content layer. Synthesis still writes a coherent
method explanation and evaluation using the full structured history. The content
layer is not a source of paper facts or a semantic correctness certificate.
Reflection contributes notes without receiving this additional workspace; its
existing evidence selection remains based on the task index described below.
Locator and Worker keep their bounded inputs. See [the content contract](PAPER_READING.md#traceable-content-during-investigation)
for fields, update timing and diagnostics.

### Existing task-routing index

Master can attach one or two `rubric_ids` to each bounded EvidenceTask. Missing
labels default to an empty tuple for backward compatibility. The runtime carries
these as `task_rubric_ids` into finding records and selected Worker context; these
are inherited task routing hints, not validated classifications of every incidental
finding. A Worker may discover a relevant issue outside its assigned dimensions.

Full state and incremental Master state derive `rubric_context_index` from the
visible finding records. `by_rubric` maps dimensions to existing finding IDs;
`unassigned_finding_ids` keeps successful legacy findings discoverable. Evidence
and caveats remain in the existing records, not duplicated per dimension. Labels
never automatically mark coverage or remove findings. Cross-rubric selection is
allowed. Labels alone do not inject other Workers' reports into a Worker call;
the rubric-union Reflection uses the rubric-union routing described below.

Locator and Evidence receive only the assigned rubric guidance alongside their
existing bounded task inputs. Master uses the index to select related finding IDs.
The rubric-union second Reflection assembles all findings matching either selected
rubric plus explicit cross-rubric or unassigned anchors, in stable history order.
The six-ID limit applies to anchors, not the assembled slice; matching findings
are not silently truncated. It rebuilds the index from that slice only and records
`context_selection` (strategy, anchor IDs, total available findings and excluded
unassigned count) in the actual prompt. `context_finding_ids` records the exact
assembled input. Labels remain coarse task-origin hints: broad labels can create
a large slice, and unrelated or unassigned evidence outside it may still matter.
Synthesis receives the index with the complete structured history to reconcile
related reports. Phase 6 remains a separate external-evidence workflow.

Prompts ask Master to treat important cross-Worker contradictions and
evidence-scope mismatches as eligible named `reflection_focus` values.
Reflector compares support, contradiction and qualification across its supplied
evidence, including issues outside the named focus, alongside mechanism and
competing-explanation reasoning. It prioritizes issues without discarding them
because another is more consequential, and merges duplicates rather than filling
a checklist. Each issue must have an evidence-grounded possible evaluation impact.
It distinguishes a missing detail on selected pages from a paper-wide absence,
names relevant finding IDs, and separates evidence-supported corrections from
checks requiring a Worker to read more. It cannot supply new paper facts.

This does **not** force a second Reflection: Master must still request a named
focus, successful evidence must exist, and the reflection budget must remain.
New findings are not required. At every turn Master assesses further-reading value separately
from verification value and explains the strongest candidate's disposition in
its existing rationale. Prior exposure to finding IDs is not proof that their
relationships were verified. Adding these prompts does not establish that models
will reliably recognize valuable reviews or provide adequate explanations.
Requesting all relevant issues does not certify exhaustive discovery: either
review can miss relationships, and union review cannot inspect omitted evidence.
The first memo's supplied finding IDs do not certify that every relationship was
verified. No live-model quality or cost result is yet available for this revision.

## Finding provenance

In history-only Synthesis, each final key finding returns `source_finding_ids`
identifying the Worker findings used to form, qualify, or correct it. It also
returns exactly one `finding_dispositions` entry per successful Worker finding,
with a `finding_id`, `status`, and substantive `reason`. Statuses are `retained`,
`merged`, `corrected`, `unresolved`, and `discarded`. Discarding an unsupported or
irrelevant finding is allowed; silently omitting an issue is not.

The runtime derives each disposition's `final_finding_indexes` (one-based)
and `method_section_ids` from the final findings' and method sections' source IDs.
Retained, merged, and corrected findings must link to either kind of target;
a merged finding must share a target with another source. Unresolved findings
must have a linked target or at least an unresolved question in either output.
Discarded findings must not remain linked. The checker also
detects unknown IDs, duplicate dispositions, malformed metadata, and unaccounted
Worker findings. Invalid links are excluded from normalized metadata; the model
call record preserves the original response for inspection.

These checks are nonfatal: a usable judgment remains available with
`provenance_status=incomplete` and explicit `provenance_warnings`. `complete`
means structural accounting passed, not that an inference, merge, correction,
discard reason, or unresolved-question match is semantically valid. Human
evaluation must still check these. The legacy full-paper and single-pass paths
keep `provenance_status=not_checked`; they do not have this Worker-only contract.
Missing or malformed method reports in new history-only runs also produce
`method_understanding_warnings`; valid sections and the judgment are preserved.
An explicit unresolved section is structurally valid and does not establish
complete method understanding. Raw model output remains in the call record.

## Optional prompt layout and cache observations

`AuditConfig.prompt_layout` and `--prompt-layout` accept `standard` (default) or
`cache-friendly`. The shared internal `_invoke_model` path reorders existing
top-level JSON fields for the latter before recording and sending the prompt.
The LLM adapter uses the recorder's layout to send
`prompt_cache_options={"mode":"explicit"}` through LiteLLM's `extra_body`.
It marks the system text block and the reusable user JSON prefix with
`prompt_cache_breakpoint={"mode":"explicit"}`. The dynamic suffix and images
carry no breakpoints. Concatenated user text remains the recorded JSON prompt.
Common instructions stay in the reusable prefix; Master `phase_instructions`
and role/task-dependent `context_instructions` stay in the dynamic suffix.
Locator query snippets also stay outside the prefix. Both layouts receive these
same instruction fields and evidence-navigation/review changes; layout is not
a switch between different role responsibilities.
If no reusable user fields exist, only the system breakpoint is used; empty
blocks are omitted. An absent system prompt produces no system breakpoint.
Role instructions, payload values, context selection, images and controller
budgets are unchanged. The per-run recorder carries the selection to all roles,
including parallel Workers, Reflection and Synthesis. Each call records the
actual prompt, layout version and pre-redaction user-prefix character offset;
no model response is replayed. `standard` keeps the original request format and
sends no cache controls. It does not disable provider caching. Unsupported cache
parameters follow the existing retry/error path without falling back to standard.

The LLM adapter retains optional cache read/write input counters alongside normal
usage. Missing counters stay unknown. The audit records runtime wall seconds
separately from the overlapping call latencies. The standalone offline
`python -m deep_research.audit_comparison` module compares two saved results and
their manifests, with optional per-model rates. It reports partial observability,
retries, configuration differences and quality-review requirements. It neither
loads credentials nor changes the inputs. The cached strategy is versioned as
`explicit-breakpoints-v3`; historical `field-order-v1` and `explicit-breakpoints-v2` runs retain their original
meaning and hashes. AiHubMix has reported prefix reuse in text probes and a full
H-Mem run; reliable speed and actual billing benefits remain unverified. See
[comparison protocol](PROMPT_CACHE_COMPARISON.md).

The v3 navigation/review changes passed 368 offline tests (main suite plus the
retained Reflection ablation tests, 2026-09-08). Local PDF checks exposed Measure
p. 12's D.1 and Table C.5 plus an importance-configuration snippet, and also
checked HarnessBank navigation. Those checks verified payload construction and
control-flow contracts at that stage. Subsequent live observations are described
in the [cache comparison](PROMPT_CACHE_COMPARISON.md) and
[reading contract](PAPER_READING.md#september-19-reme-observation); they do not
isolate the causal effect of navigation, prompts, content management or caching.

## Safety and outputs

The audit rejects a non-PDF input, unsupported Worker parallelism, or an output
directory that already exists before it constructs model clients. It writes a
manifest, progress record, and result record under the requested output
directory. Serialized public outputs are checked for credentials, base64 image
payloads, and local absolute paths.

Completed internal results redact the known paper root, configured keys and
recognized image data, with `serialization_warnings`. Ordinary paths and long
strings do not cause a completed audit to be discarded. Error-message handling
remains separately conservative. These policies and the nonfatal provenance
diagnostics are preserved by formalization.

The internal audit is paper-only. It can report that information is unreported,
but cannot infer it from sources outside the PDF. External evidence is available
only through the separate post-hoc branch after `FinalJudgment`.
