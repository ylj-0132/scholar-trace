# Optional explicit prompt cache comparison

Experiment paths in this document identify local historical records; the raw
results, wire captures and paper PDFs are not included in the public repository.
Later dated observations supplement the earlier validation stages below.

ScholarTrace keeps `--prompt-layout standard` as the default and offers
`--prompt-layout cache-friendly` as an experiment. The cached group now requests
explicit caching and sets prefix breakpoints. The standard group sends no cache
controls and can still receive provider cache hits; it is not an explicitly
cache-disabled control. Both make fresh model requests. No Function Calling
conversion, result replay or local response cache is added.

The cache-friendly layout moves existing principles, checklists, output contracts,
instructions, investigation target and paper navigation fields before changing
questions and state, then splits that JSON text into reusable and dynamic
content blocks. Between layouts, values, arrays, evidence scope, image order,
role system prompts and v20 control flow are preserved. In current v3, phase
and task-context instructions occupy separate dynamic fields, so a phase switch
or added task context does not alter the common instruction prefix. Different
roles and papers need not share one. Ordering can influence model outputs, so equivalent JSON
content is not a guarantee of equal judgment quality or cache reuse.

## Explicit request contract (2026-09-08)

For the existing GPT-5.6 role models, the cached group sends
`prompt_cache_options={"mode":"explicit"}` via LiteLLM's `extra_body`, which
places the option at the top level of the Chat Completions HTTP body. It adds
`prompt_cache_breakpoint={"mode":"explicit"}` to at most two text blocks:

1. The unchanged role system prompt, when present.
2. The reusable user JSON prefix containing the selected fields above, when present.

The second boundary is immediately after the last reusable field value; the
comma, changing question/state and remaining JSON text follow it. Concatenating
the text blocks reproduces the reordered JSON exactly. Images remain in their
original order after the text and have no breakpoint. If all user fields are
reusable, the complete JSON is the prefix; if none are reusable, the user text
gets no breakpoint. No empty text blocks are sent. No cache key or custom TTL is
set. This reuses existing message roles and does not change role responsibilities.

The current [OpenAI Chat Completions schema](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create)
documents these parameters for GPT-5.6 and later. The
[caching guide](https://developers.openai.com/api/docs/guides/prompt-caching)
explains why a common prefix alone may not match an implicit end-of-message
cache entry. The installed LiteLLM and OpenAI SDK are tested against an offline
HTTP transport. Zhizengzeng passthrough, actual writes/reads and billing are not
established by this test. Rejection follows the normal retry/error path; the
client never removes the parameters and silently reruns as standard.

A subsequent live probe on 2026-09-08 (`data/audits/prompt-cache-probe-20260908-01/review.md`)
sent eight requests: six succeeded, one timed out and one was rejected for
insufficient balance. Both successful explicit requests (Terra) reported zero
cache reads. Captured outgoing HTTP JSON contains the explicit options and
breakpoints; this does not establish gateway enforcement. Luna's explicit group
was not reached. Actual billing and a speed benefit remain unverified.

After the account was recharged, a 12-request follow-up (`data/audits/prompt-cache-probe-20260908-02/review.md`)
completed successfully. One exact Terra request repeat reported 3,843 cached
tokens. However, all four diagnostic explicit-mode requests without breakpoints
still reported cache writes, contrary to the upstream documented behavior for
that condition. Gateway handling and usage semantics remain unresolved.

The subsequent complete Measure audit (`data/audits/measure-20260908-cache-friendly-01/review.md`)
finished with DECIDE in 531.515 seconds versus 457.601 seconds for the ordinary
baseline (+16.15%). Two Locator calls reported 7,764 cached tokens, 3.12% of input.
Total returned tokens were 279,922 versus 315,089 (-11.16%), but the runs took
different agent paths (four rounds/one Reflection versus five/two), and the
candidate recovered from one transport error. This is partial observed reuse,
not an established speed or cost improvement attributable to explicit caching.
Actual billing is unknown, and the earlier paper-evidence omissions persist.

Those probes and the complete Measure audit used `explicit-breakpoints-v2`.
The subsequent v3 change removes one local source of prefix variation; it does
not explain every observed miss. In that v2 audit, the later Master prefixes
were already identical yet reported no hits. Reuse is also limited when a role
runs once, while dynamic findings and selected-page images grow. Gateway cache
handling, generation time and a retry on the slowest parallel branch can dominate
wall time. No causal v3 speed, billed-cost or judgment improvement is established. New
navigation and role-prompt changes apply to both layouts; a future controlled
comparison must run both groups from the same new source, not attribute their
differences from historical runs to caching alone.

The experiment asks whether this explicit-prefix strategy helps relative to
ordinary requests. A strict cache-off comparison would require a separately
validated control; it is not provided by `standard`.

`rubric-union` remains the default Reflection context. `full-history` remains an
independent option and should be held constant in a layout comparison.

## Run and compare

The first two commands below are **real, paid audits** when credentials are
configured. Each output directory must be new. The third command is offline and
only reads saved results and their adjacent manifests.

```bash
scholar-trace audit paper/example.pdf --output data/audits/layout-standard-01 --prompt-layout standard
scholar-trace audit paper/example.pdf --output data/audits/layout-friendly-01 --prompt-layout cache-friendly
python -m deep_research.audit_comparison data/audits/layout-standard-01/result.json data/audits/layout-friendly-01/result.json --rates rates.json
```

Omit `--rates` to compare tokens and timing without a monetary estimate. Python
callers can use `AuditConfig(prompt_layout="cache-friendly")`. The manifest records
`prompt_layout` and `prompt_layout_version` (`explicit-breakpoints-v3` for the
cached group, `field-order-v1` for standard). Each model call records its layout,
version, concatenated sent user text and `prompt_cache_prefix_chars` (the original
user-prefix character count; `null` in standard). The cached strategy also marks
the end of the system text when present. Existing output redaction still applies;
offsets describe the original request and may not index the redacted text.
The external-audit workflow is unchanged and is outside this comparison.

## Observations and costs

New internal call records retain `cached_prompt_tokens` and
`cache_write_prompt_tokens`, read from the Chat Completions response's
`usage.prompt_tokens_details.cached_tokens` and `cache_write_tokens`. Missing or
invalid counters stay `null`, including when a gateway or client omits them.
An explicit zero means the response reported zero; a missing field does not.
The implementation does not infer cache hits from repeated text or latency.

The comparison reports total input/output tokens, token cache-hit fraction, calls
with reported hits, missing-counter coverage, logical calls, extra transport
attempts, errors and per-role statistics. A total with missing usage is unknown;
it is not the sum of just the known calls.

`wall_seconds` measures extraction, rendering, the adaptive loop and Synthesis in
`run_local_paper_agent`. It excludes client/manifest setup and final serialization.
`summed_call_seconds` adds recorded request latencies, including transport retries;
parallel overlap means it is not wall time. Old results without wall timing remain
unknown. This non-streaming client does not measure time to first token.

Rates are supplied explicitly for each **actual recorded model**, per million
tokens in one currency. For example, the following is an upstream-rate illustration,
not a verified Zhizengzeng invoice or a promised account price:

```json
{
  "currency": "USD",
  "models": {
    "openai/gpt-5.6-terra": {"input": 2, "cached_input": 0.2, "cache_write": 2.5, "output": 12},
    "openai/gpt-5.6-luna": {"input": 0.2, "cached_input": 0.02, "cache_write": 0.25, "output": 1.2}
  }
}
```

The illustration combines the input/read/output rates listed in
[Zhizengzeng's model table](https://doc.zhizengzeng.com/doc-3979947) with the 1.25×
write rate for GPT-5.6 and later in the
[OpenAI caching guide](https://developers.openai.com/api/docs/guides/prompt-caching),
checked on 2026-09-07. Confirm the applicable gateway tariff, any surcharges,
long-context tiers and response-field passthrough before interpreting a live run.
Provider routing, cache boundaries and lifetime determine actual reuse; sending
cache parameters does not establish that this channel honors them.

For one returned response, the estimate is:

```text
((input - cached - written) * input_rate
 + cached * cached_input_rate
 + written * cache_write_rate
 + output * output_rate) / 1,000,000
```

`cache_write` is the total rate for written tokens, not a surcharge to add twice.
When writes are unreported and their rate differs from ordinary input, cost stays
unknown. When the two supplied rates are equal, the missing split does not affect
the estimate; the write counter itself still stays unknown. Unknown models,
missing read/input/output usage or inconsistent token counts also yield unknown
cost. Set all four prices according to the actual tariff, not to make an estimate
appear. The report embeds supplied rates and input/manifest hashes.

`estimated_recorded_cost` covers only recorded response usage. It is not a bill:
failed transport attempts may have consumed tokens without returning usage, and
gateway fees or pricing rules may require account-side reconciliation. Retry and
failure counts remain visible even when a returned response can be priced.

## Reading a comparison fairly

The tool reports `(candidate - baseline) / baseline * 100`; negative cost/time
changes mean a smaller observed value. Missing or zero baselines yield `null`.
It flags differing or missing manifest controls, including paper/source hashes,
models, prompt versions, context choices and budgets. Actual role/model/temperature
sets are also compared. A matching manifest is not proof of equivalent runs.
The two current groups intentionally have different layout versions; the tool
reports that difference for review. Source differences against older baselines
also remain visible.

Hold the PDF, models, temperature, Reflection context, parallelism, budget and
source version constant. Alternate execution order across repeated trials and
separate first-use from repeated-use observations using returned cache counters.
Earlier requests can warm shared prefixes for later requests in either layout;
do not label a first request cold without evidence. Avoid simultaneously running
both groups if the purpose is to compare latency under similar load.

Inspect final claims, caveats, evidence, unresolved issues, source links and
dispositions alongside call counts, output length and cache counters. An adaptive
run may take different actions or stop earlier, so a shorter run cannot by itself
be attributed to caching. A fixed-state prompt trial can isolate layout effects
more closely; a full audit measures the combined operational result. Repeated
observations and a quality check are needed before changing the default.

## Historical runs and validation

The original `cache-friendly` implementation, `field-order-v1`, only reordered
fields and sent no explicit cache controls. Existing results, manifests, frozen
sources, prompts and hashes retain that meaning. Explicit controls were introduced
in v2; the current CLI uses v3 and is not a byte-for-byte replay of either. The Measure baseline at
`data/audits/measure-20260907-standard-01/` used ordinary requests: all 22 calls
reported zero cache reads, but it did not disable caching through a parameter.

Implementation validation was offline; the subsequent paid probes and full-paper
cached run are described above. No causal speed, cost or quality improvement is
claimed.
Existing historical experiments and their conclusions are unchanged.

Validation on 2026-09-07: the existing `.venv` ran the main tests and the retained
Reflection comparison tests with `-B -m pytest -q -p no:cacheprovider`: **337 passed**.
Tests ran from an empty temporary directory with absolute repository/config paths,
so the project's `.env` was not loaded. An additional offline check reordered all
18 recorded v20 HarnessBank prompts across the five roles and confirmed equal
decoded JSON values. This checks payload preservation, not model behavior.

Validation on 2026-09-08: **354 passed in 5.36s**, using the same empty-directory
method and existing `.venv` (main tests plus
`variants/reflection-ablation-20260907/test_reflection_ablation.py`). New checks
cover both layouts with and without images, exact payload preservation, dynamic
fields outside the breakpoint, empty-prefix/suffix cases, versioned manifests,
cache-parameter rejection without fallback, and final HTTP bodies through the
installed LiteLLM/OpenAI SDK using `httpx.MockTransport`. The HTTP responses and
usage in that test are synthetic, not evidence of a provider cache hit.

Subsequent v3 validation on 2026-09-08: **368 passed in 6.09s**, with the same
offline command and environment. Added checks cover actual Master prefixes
across phases, Locator/Worker prefixes with and without task context, successful
Reflection snapshot versus input exposure, and bounded page-internal navigation.
The v2 artifacts, source snapshots, failed requests, retries and recorded hashes
remain unchanged; current source does not byte-reproduce their old prompts.

From an empty working directory, the PowerShell test command is:

```powershell
$auditRepo = 'C:/path/to/scholar-trace'
& "$auditRepo/.venv/Scripts/python.exe" -B -m pytest -q -p no:cacheprovider -c "$auditRepo/pyproject.toml" "$auditRepo/tests" "$auditRepo/variants/reflection-ablation-20260907/test_reflection_ablation.py" --tb=short
```

## AiHubMix transport probe (2026-09-08)

After a first attempt was rejected for model permissions (preserved in
`data/audits/aihubmix-cache-probe-20260908-01/`), the authorized retry completed
20/20 requests without retries. See
aihubmix-cache-probe-20260908-02/review.md (`data/audits/aihubmix-cache-probe-20260908-02/review.md`)
and its `summary.json`, wire requests, raw response bodies and frozen sources.

Both Luna and Terra reported hits on all 12 warmed requests, including all eight
changed-suffix requests. Explicit breakpoints worked both with and without
`prompt_cache_key`; four explicit-mode requests without breakpoints reported zero
reads and zero writes. The probe adds the optional key only in its runner; it
does not add a new production configuration option or change the default layout.

The historical Zhizengzeng probe and the new probe's first Luna no-key HTTP bodies
match after normalizing their synthetic random identifiers. This supports a
channel/backend or reporting difference; it does not establish that Zhizengzeng
dropped a particular parameter. These were not simultaneous controlled trials.

Warm requests averaged 2.265 seconds versus 2.294 seconds for the four
no-breakpoint controls: no reliable speed improvement is established. Assuming
the documented write/read multipliers, each four-request breakpoint group would
reduce input charges by about 60.72% relative to billing all its input uncached,
including the initial write premium. This is a conditional input-cost estimate,
not a measured account debit or a full-audit savings claim. Actual charges remain
unknown. This synthetic pure-text probe did not exercise paper reasoning,
images, long changing contexts or parallel requests; no full audit was launched.

### Subsequent full H-Mem audit

The separately authorized full-paper run is recorded in
hmem-20260908-aihubmix-v20-01/review.md (`data/audits/hmem-20260908-aihubmix-v20-01/review.md`).
It completed DECIDE in 365.26 seconds with 26 calls/26 attempts, 393,056 total
tokens, and two Reflections (first full-history, second Master-requested union
of 23 findings from a 35-finding pre-batch snapshot). Cache reads totaled 61,068
tokens, 17.38% of input, concentrated in Master and Locator. All ten image-bearing
requests reported zero reads/writes; this is an observation requiring a separate
modality check, not a demonstrated root cause. No additional paid probe was run.

Relative to the August 24 staged-Reflection H-Mem run, observed total tokens fell
38.75% and elapsed time fell about 71.57%. Model allocation, provider, prompts,
context, call counts, stopping policy and a historical Synthesis retry differ;
the elapsed-time change cannot be attributed solely to caching. Main causal
qualifications were retained and a concrete category-gain arithmetic check was
added, while some method details and stale missing-information caveats still need
correction. Actual dollar charges remain unknown. The review treats the strong
historical result as a reference for retained quality, useful additions and lower
resource use, rather than requiring improvement on every dimension.
