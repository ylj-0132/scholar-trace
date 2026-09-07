# Reflection input comparison

The current script offers only `full-history` and v20 `rubric-union`, using the
active `src/deep_research` runtime. It no longer constructs the retired
anchor-only focused or routing inputs. This is a bounded research comparison
over the saved HarnessBank state, not the public paper-audit entry point.

From the repository root, prepare two prompts offline in a new output directory:

```powershell
.\.venv\Scripts\python.exe -B variants/reflection-ablation-20260907/run_reflection_ablation.py prepare --output data/audits/harnessbank-reflection-new
```

The source is the saved routing audit result, used only as historical input
evidence. `prepare` does not load credentials or call models. Its state ends
before the last Master decision; neither that decision nor the Synthesis answer
is supplied. It preserves the original question, rubrics and anchors.

Raw audit data is local-only and excluded from GitHub. `prepare` therefore needs
that saved source result in the development workspace; a fresh public checkout
can run the synthetic offline tests without it.

`run --output <that-new-directory>` makes two paid Reflection calls and no other
role calls. It checks source/runtime/script and prompt hashes first and refuses
already-started output or historical manifests. No real-model run was performed
during formalization. Offline tests use synthetic source state and fake clients:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -q variants/reflection-ablation-20260907/test_reflection_ablation.py
```

## Historical four-group experiment

The immutable results remain at
[`data/audits/harnessbank-reflection-ablation-20260907-01`](../../data/audits/harnessbank-reflection-ablation-20260907-01/review.md).
The v1 manifest records the script and prompts actually used then. That script
has now been adjusted; its old hash has not been rewritten, and it is not
retained as a hidden runnable backup. The current script cannot byte-reproduce
the historical four-group experiment.

New `reflection-context-comparison-v2` manifests describe current native
payloads, including the modes' own guidance and metadata. Unlike the historical
four-group control, full history retains task routing labels and the native
union prompt retains its slice-specific instructions. New results are a
comparison of the supported policies, not the same controlled four-group
experiment. Their context metadata records the actual supplied findings.

The old four runs used the same modern v20 duties, each succeeded once, and did
not continue the audit. v20 saved 46.48% total tokens versus full history in that
one case; full history also performed well. See the
[comparison and limits](../../docs/HARNESSBANK_VERSION_COMPARISON.md).
