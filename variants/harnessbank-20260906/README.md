# HarnessBank historical comparison materials

As of 2026-09-07, the public mechanism is v20 in the main `src/` tree, with
`rubric-union` as the default and `full-history` as an explicit comparison.
Use `scholar-trace audit`; this directory is no longer a multi-version launcher.

The frozen and reconstruction directories described below remain local and are
excluded from the public repository, together with raw audit data and planning
notes. This README documents their disposition; it does not ship those copies.

The two retired reconstruction directories retain only their historical
`RECONSTRUCTION.md` and `snapshot.json` (and any empty historical working
directory). Their Python packages, generic runnable prompt assets, tests and
package configurations have been deleted. `run_variant.py`, its tests and
bytecode cache have also been deleted. Snapshot entries for removed files are
historical hashes, not a declaration that their source is still available.

`current-v20/` remains the unchanged frozen source/test reference from September
6. Its own instructions and `run_template` describe the historical runner,
which has been removed. Its old `rubric-focused` name denotes the v20 union,
not either retired anchor-only selector. Do not treat this frozen package as
the current public entry point or update its historical hashes.

All sent prompts, outputs, manifests, events, run records, failures, retries and
reviews remain in `data/audits/`. No retired executable backup has been created.
The reconstructions were untracked at handoff, so Git recovery is not confirmed;
hashes alone do not recover deleted source. The separate v20 frozen source is
retained as a v20 reference.

See the [full comparison](../../docs/HARNESSBANK_VERSION_COMPARISON.md),
[deletion inventory](../../docs/V20_FORMALIZATION.md), and
[current two-policy Reflection comparison](../reflection-ablation-20260907/README.md).
