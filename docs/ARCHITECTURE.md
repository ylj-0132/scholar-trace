# Architecture

ScholarTrace performs one create-once, paper-only adaptive audit. Its supported
entry points are `scholar-trace audit PAPER.pdf --output OUTPUT_DIR` and the
post-hoc `scholar-trace external-audit RESULT.json --output OUTPUT_DIR`.

The formal default is v20, with `reflection_context_mode="rubric-union"` in
`AuditConfig`, `PaperReflector`, and `run_local_paper_agent`. The only other
Reflection context mode is explicit `full-history`. Migrate the former v20
configuration value `rubric-focused` to `rubric-union`; it is not an alias.
This does not restore either retired anchor-only implementation. New reports
use `rubric-union` and `rubric_union_context_fallback:*`; historical context
names and diagnostics remain unchanged in saved traces. Prompt-version labels
continue to identify the v20 duties, while source hashes identify the naming
revision. Other low-level experimental context defaults are unchanged.

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

Reflection is not a source of paper facts. It reviews accumulated structured
state and can identify one consequential mechanism question for the Master to
check. Synthesis is the sole final-writing step and turns the structured history
into a `FinalJudgment`.

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
  remaining rounds. They omit the overview, images, and raw evidence excerpts.
  Follow-up suggestions include all Workers from the latest reading batch in task order;
  earlier suggestions remain in the full trace rather than being repeated.
  A reflection-only round does not erase those suggestions. Requested reflection
  focus/selection and independence rationale are retained in action summaries and
  with the latest requested memo. Remaining Reflection budget is supplied explicitly.
- Locator receives the compact index and one bounded evidence question.
- Evidence receives only the Locator-selected pages and their images, plus its
  bounded question and decision relevance.
- The first Reflection receives complete structured state, including raw Worker
  evidence, but not the overview or compact index.
- For the optional second (`master_requested`) Reflection, Master names one valuable
  audit question: the proposed conclusion, its unchecked evidence relationship,
  and how verification could change its boundary. An already-identified conflict
  is not required. Master selects one or two rubric IDs and zero to six optional
  finding anchors. In default `rubric-union` mode, the runtime sends the union
  of all findings with matching task rubric labels and the explicit anchors,
  including their evidence and caveats, selected guidance/coverage and remaining rounds.
  Unrelated history, assessments, questions, and old Reflection memos are omitted
  from the actual payload. An omitted detail is not evidence of absence.
- Synthesis receives complete structured history and Worker evidence, but not
  full-paper text, the compact index, or images.

The execution trace retains the full recorded state even when a role receives a
smaller operational payload.

`--reflection-context full-history` keeps the broader second-Reflection input
for comparisons. Both modes share the updated Master selection instructions and
the same stopping/Reflection triggers; neither forces a second Reflection.
First-round Reflection's trigger, full-state scope and original mechanism
instructions remain in place, as does the prose-only memo contract.
Reports retain `context_mode`, `context_finding_ids`, `context_rubric_ids`, and
`context_diagnostics`, separately from the controller's `reflected_finding_ids`.
An empty assembled slice or IDs absent from the successful Worker history trigger an
explicit `rubric_union_context_fallback:*` diagnostic and a full-history call. Such
runs must not be counted as successful rubric-union trials. Malformed selector
types, duplicate IDs, unknown rubric keys, and over-limit selections are rejected
by the Master-output validation path, as are missing rubric selections. The low-level
Reflector also retains a missing-selection fallback for direct adapter callers.

## Rubric as a context index

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
Reflector compares support, contradiction and qualification within its requested
scope, preserving the original mechanism/competing-explanation instructions.
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
The first review is still one consequential issue, not an exhaustive consistency
audit; focused review can still miss evidence outside its selected slice.

## Finding provenance

In history-only Synthesis, each final key finding returns `source_finding_ids`
identifying the Worker findings used to form, qualify, or correct it. It also
returns exactly one `finding_dispositions` entry per successful Worker finding,
with a `finding_id`, `status`, and substantive `reason`. Statuses are `retained`,
`merged`, `corrected`, `unresolved`, and `discarded`. Discarding an unsupported or
irrelevant finding is allowed; silently omitting an issue is not.

The runtime derives each disposition's `final_finding_indexes` (one-based)
from the final findings' source IDs. Retained, merged, and corrected findings
must link to a final finding; a merged finding must share one with another
source. Unresolved findings must have a linked final finding or at least an
unresolved question. Discarded findings must not remain linked. The checker also
detects unknown IDs, duplicate dispositions, malformed metadata, and unaccounted
Worker findings. Invalid links are excluded from normalized metadata; the model
call record preserves the original response for inspection.

These checks are nonfatal: a usable judgment remains available with
`provenance_status=incomplete` and explicit `provenance_warnings`. `complete`
means structural accounting passed, not that an inference, merge, correction,
discard reason, or unresolved-question match is semantically valid. Human
evaluation must still check these. The legacy full-paper and single-pass paths
keep `provenance_status=not_checked`; they do not have this Worker-only contract.

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
