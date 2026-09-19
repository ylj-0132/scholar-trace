# Paper reading contract (2026-09-19)

The supported `audit` command now targets two outcomes: explain the whole method
well enough to teach it to another reader, and assess the paper's value and
limitations using its own evidence. The command name and five-role v20 controller
remain unchanged. Offline contract tests and subsequent local H-Mem/ReMe runs
exercise the implementation. The observations below do not establish stable
quality, latency, token or cost improvements.

## Twelve rubric directions

| Direction | Routing ID | Purpose |
| --- | --- | --- |
| 问题定义与研究目标 | `problem_definition` | Problem, scope, inputs, outputs and objective |
| 核心思路与设计动机 | `core_contribution` | Core idea and why the authors propose it |
| 关键表示与模块职责 | `representations_and_components` | Objects, notation, states and component roles |
| 完整流程与信息传递 | `method_workflow` | End-to-end operations, transitions and dependencies |
| 关键细节与运行条件 | `key_details_and_assumptions` | Essential rules, equations, parameters and assumptions |
| 主要结果与指标含义 | `main_evidence` | Results and what the measurements actually observe |
| 评测协议与比较条件 | `evaluation_validity` | Splits, model roles, evaluation protocol and comparisons |
| 组件作用与机制证据 | `ablation_or_counterevidence` | Component contribution, controls and competing explanations |
| 资源投入与效率 | `matched_resource_efficiency` | Total resource use and matched-resource comparisons |
| 稳定性与适用范围 | `stability_and_scope` | Variability, robustness, conditions and generalization |
| 辅助模型可靠性 | `auxiliary_model_reliability` | Conditional: consequential model judgments |
| 生成变换忠实度 | `generative_transformation_fidelity` | Conditional: transformations affecting downstream information |

Examples support explanation and are not a thirteenth rubric direction. Metric
versus proxy distinctions are included in results/measurement. Costs and stability
are separated. Claim boundaries and consequential claim/result inconsistencies,
including arithmetic, remain cross-cutting obligations in the common principles.
Removing a standalone label does not remove its checks. The twelve common
mechanism-audit principles remain in force where relevant to the paper's claims.

These are preset investigation directions. Task labels route evidence; they are
not verified classifications of findings. `covered` means the model considers
a direction addressed, not that the paper passes it or evidence is sufficient.
The output does not promise twelve complete per-dimension evaluation reports.
The last two directions are conditional, not mandatory criticism. No paper must
demonstrate continual learning or self-evolution unless it claims those abilities.

## Roles and stopping

Master prioritizes both understanding gaps and evaluation questions. Existing
`conclusion_at_risk` and `expected_judgment_delta` fields can describe an incomplete
method explanation and the expected understanding gain. No new scheduling action
or score is introduced. Before stopping, Master considers whether Worker evidence
supports a connected explanation from problem to output; remaining gaps can stay
explicit when further paper reading is unproductive or the budget is exhausted.

Evidence Workers retain descriptive facts, notation, rules and paper examples,
as well as supported evaluations. Reflection checks connections among supplied
descriptions and evidence, including missing links and conflicting component
identities. Neither role acquires additional context automatically. A selected
Reflection slice cannot certify whole-method completeness.

### Selected historical evidence for Workers

Master can use the rubric content associations to find relevant prior findings,
then select at most three by their actual relevance to the new source question.
This supports complementary details and qualifications as well as explicit
cross-checks; a known contradiction is not required. Cross-rubric and unassigned
findings remain eligible. Sharing a dimension alone does not justify selection.
The existing `related_finding_ids` channel carries findings, evidence, locators
and caveats from the batch-before state, not all content in a dimension.

Each new Worker finding can record `prior_finding_ids` for the supplied reports
actually used in its reasoning. Its evidence still comes from selected pages;
the prose separates those observations from the cross-finding inference. The
runtime accepts dependency IDs only from that Worker's actual historical input.
Invalid references produce structural diagnostics without discarding the finding
or its current-page evidence. Dependency links propagate to later roles, but do
not automatically expand Reflection's selected evidence slice. They record model
declared dependencies, not proof that the inference is correct or independent.

Independent reading suppresses historical evidence and decision background in
both Locator and Worker. No extra call, mandatory reread, text-matching rejection,
or autonomous Worker scheduling is introduced. Historical prompts requiring an
explicit contradiction are superseded by this complementary-reading policy.

Synthesis still sees complete structured history and Worker evidence, with no
full paper, page index or images. It writes the method explanation and judgment
in the same call. The first automatic Reflection, optional second Reflection,
rubric-union/full-history options, five explicit rounds, two-Worker default
parallelism, and combined-action snapshot isolation are unchanged.

## Method output and traceability

New history-only Synthesis outputs `final_judgment.method_understanding` with
`sections` and `unresolved_questions`. Sections use the first five routing IDs
above plus `worked_example`, in that order. Each section contains:

- `explanation`: connected prose, with ordered steps and defined notation where useful.
- `basis`: `paper`, `inference`, `illustrative`, `unresolved` or `not_applicable`.
- `source_finding_ids`: existing Worker findings supporting or qualifying it.
- `caveat`: limitations and explanatory assumptions.

`paper` means reported by the paper, not independently verified. Substantive
inferences must be identified; a constructed example uses `illustrative` and
must state assumptions, cite the method it illustrates, and avoid invented rules
or empirical results. Insufficient evidence permits an `unresolved` example.
`not_applicable` is reserved for an example with an explained applicability limit,
not a way to omit core method sections. Mixed reported and inferred content must
be distinguished within the explanation and caveat as well as by `basis`.

Worker dispositions may point to method sections, evaluative findings, or both.
The runtime derives these destinations from citations. Missing sections, invalid
metadata and unknown IDs are reported as `method_understanding_warnings` and
included in provenance warnings; malformed output does not erase a usable
judgment. Known valid sections are retained and raw responses stay available.
Structural `complete` does not certify a correct explanation, valid inference,
appropriate disposition, semantic evidence coverage, or absence of known gaps.

## Traceable content during investigation

Rubric now also manages investigation content; it does not dictate the final
report's outline. New runs persist `trace.rubric_content`, and Master and Synthesis
receive the same trace-derived layer inside their structured state/history.
The runtime recomputes it deterministically from immutable role outputs.

Three forms of role contribution remain distinguishable:

- Worker findings carry `content_rubric_ids`, based on actual finding content.
  These may include multiple dimensions or be empty. They do not inherit task
  labels automatically and are model-proposed associations, not verified classes.
- Reflection returns its existing coherent memo plus concise `rubric_notes`.
  Notes identify a dimension, an understanding/assessment/qualification/question,
  and the supplied finding IDs they analyze. They remain candidate reasoning.
- Master returns sparse `rubric_updates` for substantive changes and `rubric_links`
  for association revisions. Empty lists mean no change, not deletion. Each update
  includes its text, dimension, kind, source IDs, status and reason.

The layer contains per-dimension finding IDs, current Master entry IDs and
candidate Reflection entry IDs. Entry text occurs once in `entries`; evidence
text is not copied into each dimension. `association_history` retains original
Worker associations and subsequent Master revisions. `unassigned_finding_ids`
retains findings with no current association. The complete Worker evidence stays
in the surrounding history, regardless of associations or entry status.

Master entries receive IDs such as `r2-master-1`; Reflection notes receive IDs
such as `reflection-1-note-1`. `source_finding_ids` refer to original evidence
reports, while `source_entry_ids` document prior reasoning being adopted,
qualified or rejected. Referencing a candidate does not convert it into evidence.
Interpretations require Worker sources; an initial open question may be unsourced.
These checks establish structural references, not whether the text follows from
the cited evidence.

To revise an interpretation, Master names the earlier current Master entry in
`supersedes`, supplies the corrected content and explains the reason. Superseded
entries remain in history but leave the current-entry list. A `resolved` or
`withdrawn` replacement records closure without creating an active claim. Closure
is itself a model judgment. Supersession is limited to current Master entries in
the same dimension; Reflection cannot overwrite them. Candidate notes remain
available with their downstream Master references rather than being silently
deleted after adoption or rejection.

Each `rubric_links` item names a finding, its full replacement list of content
dimensions and a reason. An empty list makes the finding unassigned; it never
deletes evidence or changes the task's original routing labels. Updates reference
only entries and findings that existed before the action. They cannot cite other
entries being created in the same action or the current batch's future findings.

Reflection notes may cite only the actual supplied evidence slice. The runtime
records the round after which a memo becomes available. A combined reading and
reflection batch therefore cannot let Master adopt its pending memo or let the
Reflector cite newly returned Worker evidence. On the next Master turn both
branches are available. The content layer is removed from Reflection input, so
neither full-history nor rubric-union silently gains extra candidate analyses or
changes its evidence selection. Content associations do not control that routing.

Malformed optional metadata and invalid references produce content warnings
without discarding an otherwise valid action, finding or memo. Rejected entries
do not enter current state; raw call records and role outputs retain the attempted
updates for inspection. The layer's warnings are separate from final judgment
provenance checks. Older outputs lacking metadata still work and their findings
remain unassigned rather than acquiring invented content classifications.

Synthesis receives this layer **and** full original structured history. It checks
current and superseded interpretations against original evidence, includes
unassigned material, and writes a connected explanation and evaluation. It does
not emit twelve rubric reports or replace Worker source citations with entry IDs.
The extra structured output/history has a context and token cost. The ReMe
observation below shows actual use, not an isolated or stable improvement.

## Historical boundary

### September 19 follow-up: method review and independent rereading

The initial isolated H-Mem reading run exposed incomplete method extraction, repeated
incorrect plot readings and loss of acquired detail during synthesis. The
following revision addressed these failure paths. At that point it had only
offline checks; subsequent H-Mem and ReMe observations do not retroactively
change the earlier results or establish that these failure paths are solved.

Master is prompted to maintain a concise `method_review`: its current connected
method explanation, Worker source IDs, essential method findings, six considered
aspects and explicit gaps. These aspects cover the problem, modules, workflow
and branches, details, example and evaluation support; they are not six additional
Rubric dimensions or scores. The latest review reaches subsequent Master turns
and Synthesis, while previous action outputs remain in the trace. Reflection
does not receive this additional state, retaining its existing evidence boundary.

The public audit runner requires a submitted review before DECIDE, with all six
aspects considered, a stop explanation, valid pre-action source IDs, and no gap
still designated `paper_check`. Remaining `bounded` or `external` gaps are allowed
with explicit reasons, including honest unread gaps when the budget is exhausted.
This checks structure, not semantic completeness. Invalid reviews follow the
existing invalid-Master/NEEDS_HUMAN path; no extra round, retry, closing Reflection
or model call is introduced. Low-level adapters retain an opt-in
`require_method_review=False` compatibility default; the public runner passes
True and records `method_review_policy=required-before-decide-v1`.

For disputed readings Master can select `independent_read=true`. Both Locator
and Worker then receive neither previous research context nor decision context,
and related finding selection must be empty. The question must be neutrally
worded; code cannot guarantee its semantic neutrality. Master compares returned
evidence on its next turn. Ordinary contextual cross-checking remains available.
This is source rereading within the existing two-call Worker path, not another
role or a guarantee that repeated visual estimates are accurate.

Synthesis is instructed to preserve essential acquired rules, parameters,
branches and favorable controls, and to carry method gaps into the final report.
That revision introduced a structural check when an essential source was absent
from method sections. The subsequent detail-ledger revision below replaces that
check in the supported runtime: a present citation cannot prove that an
associated detail was faithfully retained. Prompt improvements also ask
Workers to extract nearby qualifying facts and Reflection to examine method
coherence within its actual evidence slice. Budgets and snapshot isolation are
unchanged; Rubric remains a traceable content layer beneath a connected report.

### Persistent concrete details in the Rubric layer

The current revision adds `rubric_content.key_details` and `detail_history`.
Worker findings mark concrete parameters, conditions, branches, update rules and
controls with their content dimension and method/evaluation destination. Facts
may faithfully paraphrase their source findings and evidence; verbatim equality
is not an admission criterion. Stable detail IDs and Worker sources are assigned
by the controller. Empty or malformed fields are structural errors; differing
wording is not evidence of an unsupported fact.

Unmentioned facts remain active across Master turns. `detail_updates` may promote,
correct, reclassify, withdraw or restore a detail with explicit reasons and
already available Worker source IDs. Accurate paraphrase is allowed for both
original facts and corrections. Unknown/future sources and invalid identifiers
remain structural errors. Revision history is preserved. Reflection keeps its
existing evidence boundary and does not directly modify the ledger.

Synthesis writes a connected explanation and judgment, freely rephrasing or
translating while preserving concrete values, units, conditions and uncertainty.
It records each active detail's disposition, report location, source IDs and a
report excerpt. Missing dispositions, malformed locations, unknown sources and
explicit omissions remain separate structural/accounting warnings. A source ID
or matching string does not validate scientific truth or semantic completeness.

**Text matching is human-only, advisory and non-blocking.** Nonverbatim Worker
facts are admitted; Master replacements are applied; final wording differences
never remove content, change provenance/retention status, trigger another call,
or request rewriting. These notices are stored as `human_review_warnings` in the
saved content layer/final judgment and removed from model-facing Rubric state.
They are not fed to Master, Reflection or Synthesis. Human review can investigate
both valid paraphrases and real detail loss. A structurally complete result may
still have these notices; complete does not mean semantically verified. Empty
legacy detail layers remain not_checked. All original findings remain available.

### Correction of the previous implementation contract

The first detail-retention plan and the earlier version of this section wrongly
required exact source text for admission and replacement. Worker/Synthesis
prompts demanded copying, and tests expected paraphrases to be rejected. The
2026-09-19 detail-retention experiment exposed the resulting rejected metadata
and brittle final checks. This was an implementation/design error subsequently
reinforced by documentation, not a requirement to restore in future work.
The governing rule is now: **use textual comparison only to help a human inspect
content; never use it as a semantic filter or a model-output constraint.** Keep
structural source checks distinct from this advisory policy. Requirements to
quote actual paper evidence accurately remain separate from paraphrasable Rubric
facts and narrative explanations.

The ledger cannot recover facts never extracted or guarantee semantic fidelity.
The subsequent ReMe run below used the corrected advisory policy. Frozen
experiment results, earlier prompts, warnings and review records remain unchanged.

New manifests identify `report_contract=method-understanding-and-judgment-v1`,
`rubric_version=paper-reading-12-v1`, `rubric_content_version=rubric-content-v3-advisory`, and
`finding_provenance=worker-id-method-and-judgment-v2`. Role prompt versions and
the detail-ledger module are included in reproducibility metadata, together with
`detail_retention_policy=persistent-details-human-advisory-v2`.

Saved experiments, frozen prompts and historical rubric IDs remain unchanged.
No old task label is automatically treated as a new final-evaluation
classification; historical results do not acquire method reports retroactively.
Legacy full-paper/single-pass comparison paths keep
their existing judgment-only shape and do not use Worker-only method citations;
they are not an evaluation of this new reading output contract.

## September 19 ReMe observation

The local `reme-20260919-worker-context-01` run used the current complementary
reading prompts and advisory detail policy. Eleven Workers completed across five
explicit reading rounds, with two Reflections and one combined action. Eight
Workers received one to three earlier findings; eleven findings recorded
dependencies belonging to their actual input. Examples include linking the
earlier retrieval workflow to fixed-pool ablations and separating same-model
latency from a larger-model efficiency claim. These are observed uses, not proof
that the new policy causes better reasoning.

The final report explains experience acquisition, reuse and dynamic refinement,
including concrete parameters and branches. It retains earlier arithmetic and
component-attribution findings and adds a retrieval-key qualification relative
to the September 16 report: usage-scenario retrieval leads most columns, but
keywords lead Qwen3-32B Avg@4. A paper example was not fully extracted upstream;
the final report explicitly labels its example as a constructed illustration.

The run ended after exhausting five rounds, without an explicit DECIDE. Final
Synthesis succeeded after one transport retry: 30 logical calls, 31 attempts,
617,009 returned tokens and 556.21 seconds. This exceeds the September 16 run's
406,170 tokens and 462.16 seconds; changed objectives, prompts and retry behavior
prevent attributing the difference to any one mechanism. The content layer
retained 95 key details, but final provenance and detail accounting remained
incomplete (one Worker-disposition warning and 51 detail-destination warnings).
Counts and structural links do not certify semantic coverage or accuracy.

Raw experiment results and reviews remain local and are not distributed with
this repository. This summary records a single observed run and its limitations;
the public offline tests verify implementation contracts, not model quality.
