# From Full-Paper Judging to an Adaptive Paper Audit

ScholarTrace did not begin with five roles or a carefully separated context
model. The first version was a much simpler question: can a strong model read a
paper once and produce a useful, evidence-grounded judgment?

That version worked often enough to be a real baseline. It also made the
project's later direction less obvious. The adaptive system was initially much
more expensive, sometimes less reliable, and not consistently better in its
final prose. Most of the architecture in the current repository came from
following those failures through the trace rather than assuming that more
agents would help.

This document records that progression. The experiments are single runs unless
stated otherwise, so their token and latency numbers describe observed runs,
not stable benchmark estimates. Human reading notes were used after each run to
compare substantive issue coverage; they were never included in model context.

## Starting with a comparable single-pass baseline

The earliest implementation sent the extracted paper to one model and asked
for a structured judgment containing an assessment, findings, evidence,
caveats, unresolved questions, and checklist coverage. The adaptive variant
used the same final schema, but inserted a loop before the judgment:

1. Master selected a paper-internal question.
2. Locator selected a small page range.
3. Evidence read those pages and returned grounded findings.
4. Master chose whether to investigate again or stop.
5. A final Synthesis call converted the history into the same judgment schema as
   the single-pass baseline.

The shared output format mattered. Without it, a longer adaptive trace could
look better simply because it produced more text.

The first formal comparison used DRB-II, DEER, and H-MEM. Successful
single-pass runs consumed 117,143 tokens in total; the corresponding adaptive
runs consumed 563,611, about 4.81 times as much. The final judgments were often
substantively similar. On DRB-II, the single-pass result was slightly more
critical. On H-MEM, the adaptive Workers collected useful table evidence, but
Synthesis compressed it into a weaker final result. DEER also exposed the
larger failure surface of a multi-call system: one adaptive run and its first
rerun failed because of provider errors before a later create-once rerun
completed.

This ruled out the easy argument that adaptive reading was already better
because it made more calls. Its clearest early advantage was process
observability. The trace showed which question was asked, which pages were
selected, what evidence was returned, and why another round followed. It did
not yet show a better reasoning process: many runs were still a sequence of
table checks without an explicit competing explanation.

The comparison also corrected the evaluation procedure. Some early reference
documents had been model-generated rather than genuinely human ground truth.
Later reviews therefore stopped ranking answers by wording similarity. They
used issue units instead: whether the Agent found the same substantive problem,
whether its evidence supported it, and whether it introduced a new issue that
survived manual checking.

## Improving local evidence work before adding another role

The next experiments concentrated on H-MEM because its tree, graph, temporal
windows, planner, and retrieval budgets create several plausible explanations
for the same aggregate result.

The first change was selected context. Master could attach a small number of
earlier finding IDs to a new task. Locator and Evidence received those findings
as unverified research context, not as facts. This improved continuity and
cross-table reconciliation. A selected-context canary used fewer logical calls
than its no-context counterpart and made later checks more precise.

It did not solve the central problem. Three H-MEM scheduling variants all found
many useful local facts:

- ablations removed unequal bundles of structure, access paths, and budgets;
- a single top-k value controlled more than one candidate source;
- fixed MIXED scope achieved similar accuracy with roughly 1.8 times the
  retrieval-token cost;
- the reported cost figure was cumulative on a fixed workload rather than a
  scaling curve over history length.

Workers were already capable of extracting and checking those details when the
question was specific. What remained missing was the step that joined them into
a competing mechanism. For example, none of the variants naturally proposed a
matched comparison between hierarchical tree summaries and a flat collection
of multiscale summaries. Explicitly labeling Workers as Discovery or
Cross-check strengthened discipline but did not create this missing reasoning
step.

That observation led to Reflection. It was not added as another evidence
source. A Reflection report remained separate from Worker findings and could
only propose an alternative explanation, a missing control, or a bounded
question. Master still decided whether to turn the proposal into a Worker task.

## Reflection changed the route, but not automatically

The first H-MEM Reflection run showed a visible change in later actions. After
the initial evidence batch, Reflection separated tree access-path value from
summary-consolidation value, questioned the coupling between the retriever and
top-k, and identified the absence of chronological update evidence. Master then
opened targeted tasks on streaming updates, planner and decomposition behavior,
graph construction, retrieval evidence, and cross-table configuration.

One of those tasks produced the first careful ledger across Tables 2, 5, 8, 10
and Figure 3. This was a concrete improvement over adding more generic Worker
tasks. The run also showed the limit: the second Reflection mostly improved
calibration and did not justify another reading round.

The next revision moved the first Reflection earlier in the reasoning process.
Master initially focused on representations, runtime access paths, and the
actual evaluation target before treating headline results as an explanation.
Reflection was asked to consider cheap substitute mechanisms, component
compensation, representation consistency, and the smallest discriminating
control.

This produced more specific alternatives. Graph expansion might mainly enlarge
the reranked candidate pool rather than provide uniquely necessary multi-hop
reasoning. Query decomposition might be a portable upstream optimization rather
than evidence for the hybrid memory index. Aggregate QA scores might hide loss
of rare or changing facts because raw events compensate for summary errors.

Master converted several of these ideas into evidence tasks, but not all. It
still changed the strongest functional substitutes into safer questions such
as “does the paper report an independent ablation?” The remaining bottleneck
was therefore no longer hypothesis generation or page-level reading. It was
the conversion of a useful hypothesis into a precise, discriminating task.

The staged Reflection runs were expensive—roughly 600,000 total tokens on
H-MEM—and the prompt structure was still growing. They nevertheless supplied
the first evidence that adaptive reading could change the investigation route,
not merely collect more details. This became the process-level reason to keep
the adaptive architecture even though early final judgments had not beaten the
single-pass baseline consistently.

## Splitting planning quality from Worker cost

A later DRB-II run used Reflection to question whether source-derived rubrics
partly reward reconstruction of a reference report rather than independent
research ability. The run followed this into semantic-equivalence rules,
alternative evidence, leakage, human comparison, and score aggregation. It was
strong on investigation quality but consumed 530,035 tokens and exhausted the
five-round budget.

Running the complete loop with different model families clarified where the
stronger model mattered. An all-Luna run stopped more efficiently and reached a
similar overall assessment, but Master did not dispatch Reflection's most
distinctive lead about reference reconstruction. Individual Luna Workers were
still effective at local evidence analysis. A Kimi run also produced strong
local observations, but repeated provider timeouts made the long loop
operationally unreliable.

The resulting default uses Terra for Master and Reflection, where planning and
cross-finding reasoning have the largest downstream effect. Locator, Evidence,
and Synthesis use Luna. This was a cost allocation decision rather than a claim
that the smaller model defines the system's quality ceiling.

## Moving to self-evolution papers

The first transfer experiment reused the same architecture and a neutral audit
target on FlowEvo and ReMe. No paper-specific answer or human note was supplied
to the Agent.

FlowEvo used all five rounds, one Reflection, 31 model calls, 541,365 prompt
tokens, and 43,554 completion tokens. The trace reconstructed the
workflow-to-skill loop, skill admission and routing, staged ablations, transfer
boundaries, and incomplete lifecycle-cost reporting.

ReMe also exhausted five rounds. It used two Reflections, 35 calls, 621,899
prompt tokens, and 46,052 completion tokens. It reconstructed experience
acquisition, retrieval, reranking, rewrite, validation, utility updates, and
retrieval-conditioned deletion. It also found the ambiguous credit assignment
over jointly retrieved memories, unspecified dynamic-evaluation chronology,
missing long-horizon pool and cost evidence, and the reversed 7.29/8.83
Avg@4/Pass@4 labels.

The transfer was encouraging because the same roles found load-bearing issues
in two different self-evolution mechanisms. It also made the efficiency problem
impossible to ignore. Both runs repeated large histories and continued until
the round limit even after the central judgment had stabilized.

Manual comparison identified another boundary. Some questions—such as who is
responsible for ReMe's rewrite and whether applicability conditions survive
that transformation—were harder for the Agent to prioritize without prior
experience building agent systems. Other questions depended on external
evidence and should not be counted as failures of a paper-only reader.

## Teaching Master when to stop

The first efficiency change was deliberately small. Before opening another
round, Master had to state what conclusion was at risk, which paper evidence was
missing, and how the answer could change or qualify the judgment. It could stop
with an explicit reason when the remaining gaps had already been checked as
unreported or could only be resolved through code or external evidence. A
second pre-decision Reflection ran only when Master named a specific unresolved
conflict.

On ReMe, this reduced the run from five reading rounds to three and from 35 to
18 logical calls. Prompt tokens fell from 621,899 to 292,795, completion tokens
from 46,052 to 26,143, and wall time from about 1,231 seconds to 944 seconds.
The top-level assessment was preserved.

The detailed quality gate did not pass. The shorter run omitted the
stronger-summarizer dependence, validator reliability, matched total-compute
qualification, and the headline label error. Inspection showed that several of
these losses occurred before Synthesis: Master had never assigned the relevant
Worker question. The lesson was not that early stopping should be removed. It
was that stopping and coverage had to be evaluated separately.

This trace also showed duplicated effort. Two later rounds repeatedly searched
for dynamic-evaluation write, order, persistence, and reset rules after a
targeted check had already established that the supplied paper material did not
report them. The stopping contract was tightened so that an unreported detail
would remain an unresolved boundary rather than be reopened under different
wording.

## Compressing repeated state

The next step changed only what later Master calls received. The first call kept
the overview and two page images. Later calls retained completed task summaries,
finding IDs, findings and caveats, the latest Reflection, the current
assessment, unresolved questions, and checklist state. They no longer repeated
the overview, images, or raw evidence quotations. The full trace still retained
those items for audit and final synthesis.

In the resulting ReMe run, Master prompt usage fell from 113,344 to 51,011
tokens. The complete run used 270,006 prompt tokens and 28,831 completion
tokens. It completed two reading rounds and then decided. The final judgment
recovered rewrite-fidelity limits, summarizer dependence, the matched-compute
boundary, and the swapped headline labels.

The direct Master reduction was meaningful; the apparent end-to-end wall-time
gain was not a clean attribution. The previous run's Synthesis had suffered a
transport retry, while this run's Synthesis was much faster. Reporting the full
wall-time difference as a context-engineering gain would have mixed a prompt
change with provider variance.

The larger design lesson was that model context and audit storage need not be
the same object. Raw evidence can remain in the trace for replay and human
checking without being resent to Master every round.

## Removing the final full-paper safety net

At this point Synthesis still received the full paper together with the entire
investigation history. That made the final call expensive and blurred the
architecture: the system performed adaptive reading, then ended with something
close to another single-pass judgment.

The context-ownership experiment moved the paper text upstream. First-round
Master received selected main-body text; later Master calls received only the
incremental state. Reflection saw the structured state and Worker evidence but
not the overview or page index. Synthesis saw the investigation history and
evidence but not the paper or page index.

This run reached a sound paper-level judgment, but it exposed what the
full-paper Synthesis had been silently recovering. The trace missed or failed
to preserve several exact claims, including the reversed headline labels, the
validator threshold, and the precise smaller-model comparison. Removing the
safety net made responsibility clearer: a load-bearing claim seen in the first
call had to become a durable Worker task or it could disappear.

Master and Locator prompts were then adjusted narrowly. Master was asked to
turn one or two load-bearing claims into bounded checks when they depended on
arithmetic, metric labels, model matching, or a named table or figure. Locator
was asked to select the page containing a named source rather than a merely
related discussion.

The next ReMe run used 237,927 prompt tokens, 31,340 completion tokens, 25
logical calls, and 674 seconds of wall time. Relative to the original transfer
run, prompt usage fell 61.74%, total tokens 59.69%, calls 28.57%, and wall time
45.22%. Locator recovered the exact validator rule from Table 13, and a Worker
correctly recalculated the 7.29 and 8.83 point gains from Table 1.

The final judgment still omitted that arithmetic because the selected page did
not contain the complete headline sentence and Synthesis did not join the two
findings. This separated two different problems: Locator had improved, but
claim preservation across the whole trace was not finished.

## Overview versus more first-round text

A final ReMe context comparison returned first-round Master to the two-page
overview instead of injecting selected main-body text. All later role boundaries
remained unchanged.

The overview run used 23 calls, 257,453 prompt tokens, 30,902 completion tokens,
and 570 seconds wall time. Master dispatched eight Worker tasks across four
reading rounds. Its first round directly covered the lifecycle, dynamic-memory
protocol, the 8B-versus-memoryless-14B headline, the 7.29/8.83 values, and
end-to-end cost. Later rounds covered auxiliary-model reliability, rewrite
fidelity, statistical uncertainty, dynamic ordering, and retrieval keys.

It did not use fewer tokens than the immediately preceding claim-locator run,
but it used fewer calls and completed faster. More importantly, selected
main-body injection had not produced a clear quality advantage. The useful
context change was primarily subtraction: raw evidence stopped flowing back to
Master, and full-paper content stopped flowing to Synthesis. Giving Master more
paper text was less important than making each role responsible for preserving
what it had verified.

The overview-first profile therefore became the public default.

## Transferring the reduced-context design to HarnessBank

HarnessBank was used as a transfer paper because its semantic Gene Bank,
recombination, and four-gate screening process differ substantially from ReMe's
procedural memory lifecycle.

The adaptive loop completed three reading rounds, six Worker tasks, and 33
structured findings. It used 123,048 prompt tokens and 19,744 completion tokens
before final synthesis. The trace reconstructed bank entries and semantic
coordinates, parent selection and recombination, the four screening gates,
train/test handling, verifier roles, cross-model results, and the lack of a
matched-resource comparison. It also found that the paper did not report a
bank-disabled or size-matched unstructured-archive control, so the end-to-end
gain did not isolate the Gene Bank itself.

The original Synthesis call failed after its two allowed transport attempts.
The run was not overwritten or restarted. A later Synthesis-only replay reused
the exact recorded prompt and history, called no other role, and completed with
24,059 prompt tokens and 4,978 completion tokens in 108 seconds.

That replay preserved most of the important mechanism and evaluation caveats,
but it revealed a cross-finding reasoning failure. The history contained both
facts: HarnessBank used Claude as evolver in its default setup, while the
reported GEPA and DGM comparison described Qwen as task agent and proposer.
Synthesis still called the comparison protocol-aligned instead of treating the
unresolved proposer/evolver difference as a possible confound. The facts had
been collected; the final model failed to combine them.

The Synthesis instruction was subsequently amended to compare task model,
proposer or evolver, verifier, data split, and resource budget before
attributing a difference to a method. Reflection was also narrowed toward one
consequential mechanism conflict rather than acting as a general second
reviewer. These prompt revisions have offline tests but have not yet been
validated by another paid HarnessBank run.

One piece of instrumentation was removed rather than strengthened. An evidence
integrity warning attempted to match model quotations exactly against selected
PDF text. PDF line breaks, hyphenation, and model-added quotation marks made the
warning fire on most Worker results, while some genuine paraphrases were mixed
into the same category. Because it had lost its ability to distinguish the two
cases, the warning was deleted. Workers still retain evidence text and page
locators for manual inspection; a noisy alarm is not treated as provenance.

## What remains outside the current system

The supported CLI now runs one create-once, paper-only audit. It does not search
the web, inspect an implementation repository, or execute a paper's experiments.
Those boundaries are deliberate.

A manual external-evidence pilot showed where a later audit stage could matter.
For HarnessBank, external checking strengthened the concern about evolver-model
comparability and showed that much of the seven-domain evidence came from an
author-associated benchmark suite. For ReMe, citation scope and evaluation
chronology changed how strongly “lifelong evolution” could be interpreted.
These findings should not be smuggled into a paper-only prompt. A future
external stage would need explicit triggers, source provenance, and a recorded
statement of how each source changed the judgment.

The current system also does not independently prove a paper's mechanism. It
audits whether the paper's own method, appendices, ablations, and experiments
support the stated interpretation. Missing code, unreported chronology, or an
unmatched control remains unresolved rather than being converted into a
negative fact.

## Condensed timeline

| Date | Experiment or change | Main observation |
| --- | --- | --- |
| Aug 20–22 | Single-pass and early Adaptive on DRB-II, DEER, H-MEM | Similar final judgments; Adaptive used 4.81× the tokens and exposed a larger failure surface. |
| Aug 23–24 | H-MEM selected-context and scheduling comparisons | Context improved local verification, but no role owned competing-mechanism construction. |
| Aug 24 | Reflection-D | Reflection changed later tasks; hypothesis-to-task conversion became the next bottleneck. |
| Aug 25–26 | Role-model split and FlowEvo/ReMe transfer | Terra was retained for planning; both transfer papers were understood, but both exhausted five rounds. |
| Aug 27 | ReMe marginal-value stopping | Prompt tokens fell 52.9%, but several claim-level issues disappeared. |
| Aug 28 | Incremental Master context | Master prompt tokens fell 55%; raw evidence remained in the trace instead of every Master call. |
| Aug 28 | History-only Synthesis and claim/locator work | Removing full-paper Synthesis clarified ownership and exposed missing claim preservation. |
| Aug 28 | ReMe overview-first profile | More first-round text showed no clear advantage; subtraction mattered more than injection. |
| Aug 28–29 | HarnessBank transfer and replay | The loop transferred, while Synthesis missed a model-role confound already present in the history. |
| Aug 29 | Public runner and repository cleanup | Historical runners were replaced by one `audit PAPER --output DIR` interface. |

The project ended this phase with a smaller public surface and a more explicit
division of responsibility. Master plans and stops, Locator navigates, Evidence
grounds local claims, Reflection proposes one consequential competing
explanation, and Synthesis writes only from the accumulated audit history. The
main improvement was not adding more agents or more context. It was making
information ownership and failure attribution easier to see.
