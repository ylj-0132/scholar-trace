# Architecture

ScholarTrace performs one create-once, paper-only adaptive audit. Its supported
entry points are `scholar-trace audit PAPER.pdf --output OUTPUT_DIR` and the
post-hoc `scholar-trace external-audit RESULT.json --output OUTPUT_DIR`.

## Execution flow

```text
local PDF
  -> Master
  -> Locator -> Evidence Worker (one per bounded question)
  -> Master (repeat while needed)
  -> Reflection (at most two bounded reviews)
  -> Synthesis -> FinalJudgment
                   -> external-audit (optional, separate create-once output)
                      -> Terra Planner -> Grok native web search -> Terra Auditor
                      -> external judgment delta
```

The Master owns the global investigation state and decides whether to read more
or make a supported decision. A `READ_PAPER` action can dispatch independent
questions in parallel. Locator navigates a compact page index; Evidence reads
only the selected page text and corresponding page images. Evidence reports
both findings and caveats, so missing implementation details remain visible.

Reflection is not a source of paper facts. It reviews accumulated structured
state and can identify one consequential mechanism question for the Master to
check. Synthesis is the sole final-writing step and turns the structured history
into a `FinalJudgment`.

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
- Locator receives the compact index and one bounded evidence question.
- Evidence receives only the Locator-selected pages and their images, plus its
  bounded question and decision relevance.
- Reflection receives the complete structured state, including raw Worker
  evidence, but not the overview or compact index.
- Synthesis receives complete structured history and Worker evidence, but not
  full-paper text, the compact index, or images.

The execution trace retains the full recorded state even when a role receives a
smaller operational payload.

## Safety and outputs

The audit rejects a non-PDF input, unsupported Worker parallelism, or an output
directory that already exists before it constructs model clients. It writes a
manifest, progress record, and result record under the requested output
directory. Serialized public outputs are checked for credentials, base64 image
payloads, and local absolute paths.

The internal audit is paper-only. It can report that information is unreported,
but cannot infer it from sources outside the PDF. External evidence is available
only through the separate post-hoc branch after `FinalJudgment`.
