# v20 formalization and cleanup ? 2026-09-07

## Pre-change evidence

Current src Python files matched the frozen current-v20 copy byte-for-byte. Each of the three snapshots verified all 25 recorded files. Baseline offline suite: 290 passed. Existing working-tree modifications are the starting point, not a reason to restore Git HEAD.

## Exact deletion targets

All paths below were resolved and confirmed inside this repository before deletion. The two src trees include their generic historical_prompt_static.json runnable prompt assets; actual sent prompts remain in historical results.

- `variants/harnessbank-20260906/rubric-focused-reconstructed/src`
- `variants/harnessbank-20260906/rubric-focused-reconstructed/tests`
- `variants/harnessbank-20260906/rubric-focused-reconstructed/pyproject.toml`
- `variants/harnessbank-20260906/rubric-focused-reconstructed/.pytest_cache`
- `variants/harnessbank-20260906/rubric-routing-reconstructed/src`
- `variants/harnessbank-20260906/rubric-routing-reconstructed/tests`
- `variants/harnessbank-20260906/rubric-routing-reconstructed/pyproject.toml`
- `variants/harnessbank-20260906/rubric-routing-reconstructed/.pytest_cache`
- `variants/harnessbank-20260906/run_variant.py`
- `variants/harnessbank-20260906/test_runner.py`
- `variants/harnessbank-20260906/__pycache__`

## File inventory before deletion

| File | SHA256 |
| --- | --- |
| `variants/harnessbank-20260906/__pycache__/run_variant.cpython-311.pyc` | `93a4ceff1586b986490a41e2373b982ed673deef4d1540238ff1d5093fa76085` |
| `variants/harnessbank-20260906/__pycache__/test_runner.cpython-311-pytest-7.4.0.pyc` | `53299381547785276fef17d8e4adf4eb5f317ed5619cc75ccc1f8ede25e123ba` |
| `variants/harnessbank-20260906/__pycache__/test_runner.cpython-311.pyc` | `b673c46b51454a414604e3247754bbfbdf9b85ba0c00b229414aac1f248d445a` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/.pytest_cache/.gitignore` | `e7c6bb30148cf667606dcd63e7ca77acaa3cfb0c8303bf09e6419e1e1669dc6d` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/.pytest_cache/CACHEDIR.TAG` | `37dc88ef9a0abeddbe81053a6dd8fdfb13afb613045ea1eb4a5c815a74a3bde4` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/.pytest_cache/README.md` | `420e808d79a6c25d3cda0af33bc4782314a14949866682c68ce8149e89b66b70` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/.pytest_cache/v/cache/lastfailed` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/.pytest_cache/v/cache/nodeids` | `a3e2ecb9f06b3538a3fc61c41ab0385a0fe87a055ff7b2c469e82863ddf71434` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/.pytest_cache/v/cache/stepwise` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/pyproject.toml` | `6082ace8e1095fa23e4b842fcd81799df22adcc5296a81f6eb96fd3db5b7e996` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__init__.py` | `434732c71c599af0e485d876c7e50583007b3f93b6b98794e2bb22a2eb3f5303` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__main__.py` | `3dee42fba689842ce1919a9573c0eada34c967e33a744757a6461be22ed4ff95` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/__init__.cpython-311.pyc` | `190a2f8f56e40926610d9d11a595b7ca3fdaa83faff0ffbe7eddf984d1cfcea5` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/cli.cpython-311.pyc` | `afd80c70d3e72fb219a5ddc9e32e90ef9dcc188bc43e09fb0793fff33c7a84ba` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/config.cpython-311.pyc` | `160980353cf092f7e71b80eac9ec965ca1a9f0da045fa4d4c04b31dde6d8cb1c` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/experiment.cpython-311.pyc` | `7b332779b2f7bdb72a62c9eab069a89b048054d1589bfc70b86b16306dabc8c2` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/external_audit.cpython-311.pyc` | `ada0692a73aba2d6f0389b952762304fa5e37c0c2781e8af409999b6b97b277f` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/llm.cpython-311.pyc` | `ef8ed32079caac24064ac14af44aff7e4d3387643b6320bb86ceaf0b153da4b8` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/paper_agent.cpython-311.pyc` | `acedf994ee93b9b28c1ddbbc9cc8d6555fbc15eb8d0fb9c897ae1fd2bfd56d30` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/paper_agent_runtime.cpython-311.pyc` | `a7b6f752104dcd384117d019d8fdc70a99a5b23d8d722851534aa047203aa4cf` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/__pycache__/paper_reading.cpython-311.pyc` | `a1ce60b6f8d3d24f8ec1c5dd9b3de3fdc2906f1f25f38f9dac5a7b9d7b23bbdf` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/cli.py` | `8051aefc30565d94eb342a69d0d0bc69b7b50b4a38c7cd0fb9ff4dad3d107646` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/config.py` | `e372fa4d15fc6e5c59f2012d3396b68076e7d90e1d95b3250190c68bd6a41131` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/experiment.py` | `0aa3d60d265197f8913216a211e4039f5ee9acaa691fb64ab5f49f0f06354cdb` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/external_audit.py` | `1a1eff0c7103ba4a867211bce971fd19ab8d40d00c03aebe895324449010598a` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/historical_prompt_static.json` | `5b7e3fc8c819e789b692d3da9e66369f588d54b3518b0a43fe01ce1063541201` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/llm.py` | `6b26b0c29edf5179ccfa7a109ca9b687df92b9556e9a3901d880ec072d21325b` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/paper_agent.py` | `6159f0368baeb567527925ed2d3ed37ee0fe3edc0d95f076d8c6fe9e2a6a32d6` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/paper_agent_runtime.py` | `0a59b4349e511e24830247d978b329d90733caa9495269a9daf864f36816c69f` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/src/deep_research/paper_reading.py` | `a17ef9e63c459717548407e675e8d2c0d0d6e3d89f326f07df736a9131ee9be6` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_cli.py` | `3932acbfed9dae81027d724dc94dbe6a3744a26dbbb97e4c9edf4e1c8deeced4` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_config.py` | `a838c85ed369effae9c538a979021ba5263c898d235136c2b1316d8228452848` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_experiment.py` | `371a6cb846caa124dc217e93bb052aec1ff22a97b79911ca4bc0b5a2ab303e14` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_external_audit.py` | `691f8e30f9c51e0e093369a7a796da81b87a81f8e96a748736b16fa5c75b8369` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_llm.py` | `6830d2d97365487b507ee18f12f2580ce99fcd5c757cfd99e9b4fd5f191363f5` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_paper_agent.py` | `590b5e7b83bf209c8caf99af0dff912345e2bc3487b8841c15ddecc77eecd7c9` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_paper_agent_reflection.py` | `24dc782f6a9807ad06766eb15b7a09f124ba93a25ca78e8ff37d3dd5b54e4f02` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_paper_agent_reflector_runtime.py` | `16672157a53b3448f869583c3d5ab108ffb3acb38aa7cb6f613a319006dfc051` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_paper_agent_runtime.py` | `92badbbf876c4c9224484fd28d9ecad4ed5147bc032faad9f31ee9d3d7edb0d2` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_paper_reading.py` | `095d6b603b4106bab1ee31e7c47b1892dcd0d2e3a2fe8d48527c1299a44a1b62` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_reconstruction.py` | `6c848a2494e2265b2e6691f07d3d3f1038f7c62b5b4a89e59cbbb8a903219f14` |
| `variants/harnessbank-20260906/rubric-focused-reconstructed/tests/test_reconstruction_parity.py` | `c81bffc419eda0471cee4d36ce92a664a466475bfe21c07de7fe4cec6a2f0e47` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/.pytest_cache/.gitignore` | `e7c6bb30148cf667606dcd63e7ca77acaa3cfb0c8303bf09e6419e1e1669dc6d` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/.pytest_cache/CACHEDIR.TAG` | `37dc88ef9a0abeddbe81053a6dd8fdfb13afb613045ea1eb4a5c815a74a3bde4` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/.pytest_cache/README.md` | `420e808d79a6c25d3cda0af33bc4782314a14949866682c68ce8149e89b66b70` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/.pytest_cache/v/cache/lastfailed` | `44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/.pytest_cache/v/cache/nodeids` | `c77cec076de6fae10a7cc826200750e92400a879c2735e9b58346fba74b611aa` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/.pytest_cache/v/cache/stepwise` | `4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/pyproject.toml` | `6082ace8e1095fa23e4b842fcd81799df22adcc5296a81f6eb96fd3db5b7e996` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__init__.py` | `434732c71c599af0e485d876c7e50583007b3f93b6b98794e2bb22a2eb3f5303` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__main__.py` | `3dee42fba689842ce1919a9573c0eada34c967e33a744757a6461be22ed4ff95` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/__init__.cpython-311.pyc` | `190a2f8f56e40926610d9d11a595b7ca3fdaa83faff0ffbe7eddf984d1cfcea5` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/cli.cpython-311.pyc` | `afd80c70d3e72fb219a5ddc9e32e90ef9dcc188bc43e09fb0793fff33c7a84ba` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/config.cpython-311.pyc` | `160980353cf092f7e71b80eac9ec965ca1a9f0da045fa4d4c04b31dde6d8cb1c` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/experiment.cpython-311.pyc` | `7b332779b2f7bdb72a62c9eab069a89b048054d1589bfc70b86b16306dabc8c2` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/external_audit.cpython-311.pyc` | `ada0692a73aba2d6f0389b952762304fa5e37c0c2781e8af409999b6b97b277f` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/llm.cpython-311.pyc` | `ef8ed32079caac24064ac14af44aff7e4d3387643b6320bb86ceaf0b153da4b8` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/paper_agent.cpython-311.pyc` | `acedf994ee93b9b28c1ddbbc9cc8d6555fbc15eb8d0fb9c897ae1fd2bfd56d30` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/paper_agent_runtime.cpython-311.pyc` | `a7b6f752104dcd384117d019d8fdc70a99a5b23d8d722851534aa047203aa4cf` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/__pycache__/paper_reading.cpython-311.pyc` | `a1ce60b6f8d3d24f8ec1c5dd9b3de3fdc2906f1f25f38f9dac5a7b9d7b23bbdf` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/cli.py` | `8051aefc30565d94eb342a69d0d0bc69b7b50b4a38c7cd0fb9ff4dad3d107646` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/config.py` | `e372fa4d15fc6e5c59f2012d3396b68076e7d90e1d95b3250190c68bd6a41131` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/experiment.py` | `2aca9eb372a0b1fd450358845e56dcfb28c3630f1fbd0d49e084e0e7436ecc67` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/external_audit.py` | `1a1eff0c7103ba4a867211bce971fd19ab8d40d00c03aebe895324449010598a` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/historical_prompt_static.json` | `8e5256702fa5531dd581510bf6d3ec8a8907c95ca541deb78c75deca1e0a1b55` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/llm.py` | `6b26b0c29edf5179ccfa7a109ca9b687df92b9556e9a3901d880ec072d21325b` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/paper_agent.py` | `bc4d26f00aa26b4482dcfdf4a2eca2b43f75edfcbd59722eb0b16dbefd6ef3bc` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/paper_agent_runtime.py` | `2aa1f349b32f9ef47e08cdab6dc37aa7143b8ab0eb25d515a0bc1c5229ad9761` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/src/deep_research/paper_reading.py` | `a17ef9e63c459717548407e675e8d2c0d0d6e3d89f326f07df736a9131ee9be6` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_cli.py` | `3932acbfed9dae81027d724dc94dbe6a3744a26dbbb97e4c9edf4e1c8deeced4` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_config.py` | `a838c85ed369effae9c538a979021ba5263c898d235136c2b1316d8228452848` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_experiment.py` | `2fe7697159e78492db252cf1090ab364a997c07670cb584f72e8f4b59f3d4d5c` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_external_audit.py` | `691f8e30f9c51e0e093369a7a796da81b87a81f8e96a748736b16fa5c75b8369` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_llm.py` | `6830d2d97365487b507ee18f12f2580ce99fcd5c757cfd99e9b4fd5f191363f5` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_paper_agent.py` | `590b5e7b83bf209c8caf99af0dff912345e2bc3487b8841c15ddecc77eecd7c9` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_paper_agent_reflection.py` | `24dc782f6a9807ad06766eb15b7a09f124ba93a25ca78e8ff37d3dd5b54e4f02` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_paper_agent_reflector_runtime.py` | `16672157a53b3448f869583c3d5ab108ffb3acb38aa7cb6f613a319006dfc051` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_paper_agent_runtime.py` | `c13a14a5f0c0fffa1750e8d9bbb067c461de5fa993d176a3904796d11c525af9` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_paper_reading.py` | `095d6b603b4106bab1ee31e7c47b1892dcd0d2e3a2fe8d48527c1299a44a1b62` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_reconstruction.py` | `47bc9b9aa58a450358db1e24d1b8b783884c829add311bfb09d07fe61e226799` |
| `variants/harnessbank-20260906/rubric-routing-reconstructed/tests/test_reconstruction_parity.py` | `296f1b818efa5605158f3ab57f179cc64ced0182f5cdd0be696b04574b434cda` |
| `variants/harnessbank-20260906/run_variant.py` | `4853625f392a005887058b3a5de18c5ef218402ac37cfe1d09ed90d3fbde7e4b` |
| `variants/harnessbank-20260906/test_runner.py` | `f89c28a7ae51b22ce32cd3623727aa05cce1b7d2db33b29b80525f970622b045` |

## Preserved material

All data/audits records and the current-v20 frozen directory are protected by a pre-change hash inventory (103 files, held in temporary verification storage). The two reconstruction notes and snapshot manifests remain in place. No legacy executable backup is created. local_archive/, data/experiments/, paper/ and unrelated documents are outside this change.

## Recovery boundary

The removed reconstructions were untracked at handoff. Their manifests/hashes cannot restore their source bytes, and the current Git history is not a confirmed recovery source. Actual sent prompts, outputs, events, failure/retry records and reconstruction notes remain available for historical audit. Frozen current-v20 remains a source reference for v20, not a backup of the retired mechanisms. The old runner and ablation-script hashes remain historical identifiers; the modified script cannot byte-reproduce the old four-group experiment.


## Completed changes

- Removed all 11 enumerated targets (83 files including bytecode/test caches). Each retired reconstruction now contains only RECONSTRUCTION.md and snapshot.json; the focused directory also retains its pre-existing empty offline-test-cwd. No removed executable was renamed or backed up.
- Kept the frozen current-v20 directory byte-for-byte. Its historical run_template refers to the removed launcher; the parent README explains this. It is not the current public entry point.
- Current production changes relative to that frozen source are confined to cli.py, experiment.py and paper_agent_runtime.py: mode naming, diagnostic naming, and explicit union defaults for the lower-level Reflection adapters. The controller, evidence extraction, LLM transport, provenance logic, convergence duties and external audit implementation were not rewritten.
- The comparison script now imports active src, constructs only the two native supported inputs, requires --output, writes reflection-context-comparison-v2 manifests, rejects historical manifests before client construction, and records the actual context metadata. It no longer strips labels or constructs either anchor-only selector.

## Configuration migration

The public default remains v20 behavior, now named `rubric-union`. Use:

```text
scholar-trace audit paper/example.pdf --output data/audits/example
scholar-trace audit paper/example.pdf --output data/audits/example-full --reflection-context full-history
```

Python callers use `AuditConfig()` or `AuditConfig(reflection_context_mode="full-history")`.
Replace former v20 `rubric-focused` configuration with `rubric-union`; the old
name is rejected rather than interpreted ambiguously. Direct PaperReflector and
run_local_paper_agent callers that depended on their previous full-history
default must now request it explicitly. New fallback diagnostics use
`rubric_union_context_fallback:*` in place of `focused_context_fallback:*`.
Saved historical configurations, prompts, traces and hashes are not migrated.

## Verification actually performed

Baseline: **290 passed in 0.89s**. Updated CLI/config/default selector tests first
failed on the former names/defaults. The two-policy runner tests also failed
against the original four-policy script. Final combined suite:
**300 passed in 1.46s** (295 main-suite tests plus 5 comparison-script tests).

The existing virtual environment was used from an empty temporary working
directory so configuration helpers did not read the project's real .env.
Dotenv unit tests read only their own synthetic fixtures. No dependencies were
installed. The final test invocation is shown below with machine-specific paths replaced by variables:

```powershell
# Set $repo to the repository path and run from an empty temporary directory.
& "$repo/.venv/Scripts/python.exe" -B -m pytest -q -p no:cacheprovider -c "$repo/pyproject.toml" "$repo/tests" "$repo/variants/reflection-ablation-20260907/test_reflection_ablation.py"
```

Coverage includes automatic Reflection, standalone and overlapping combined
actions using prior state, repeated review without new findings, budgets and
stops, union plus unassigned/cross-rubric anchors, uncapped matching findings,
full history, fallbacks, all latest Worker suggestions, provenance warnings and
successful serialization of benign paths/long strings. The external-audit tests
use fake clients/transports; Phase 6 was not run against any real source or model.

Additional checks:

- All **103 protected files** matched pre-change SHA256 values after cleanup,
  including every data/audits file, frozen v20 file, and retained reconstruction
  note/manifest in the inventory.
- Native prompt construction against the saved pre-final state gave **33 full-history
  findings and 18 union findings**. The union state matched the recorded historical
  v20 state exactly, included Worker finding r1-t3-f4 (K=1 selection versus K=3 final
  evaluation), and contained no final DECIDE entry. This invoked no model.
- CLI help displayed only `{full-history,rubric-union}`, with rubric-union as
  default. The comparison script required an explicit output directory. The
  package is not installed into this existing environment, so module help from
  the temporary directory was checked with PYTHONPATH pointing to the repo src;
  no installation was needed.
- Source scan found no retired selector or runnable historical-prompt asset in
  current src or comparison scripts. Old names remain only in rejection tests,
  historical documentation/manifests, and v20's preserved prompt-version labels
  or frozen source. Both retired package trees and the shared runner are absent.
- Updated relative documentation links resolved. `git diff --check` passed;
  Git printed existing LF/CRLF conversion notices, not whitespace errors. No
  Git write operation was performed during the cleanup phase.

## Documentation and remaining limits

Updated README.md, docs/ARCHITECTURE.md, docs/CASE_STUDY.md,
docs/HARNESSBANK_VERSION_COMPARISON.md and both variant-directory READMEs.
The comparison document links the four original prompts/results and clearly
separates historical checkpoints from current availability.

The three complete HarnessBank audits each used only one automatic Reflection;
they do not validate the second-input policy. The later four-input experiment
used common modern v20 duties, each ran once, and stopped after Reflection.
v20 used 46.48% fewer total tokens than full history in that one selected case,
while full history also performed well. The K=1/K=3 fact already existed in a
Worker finding. No stable capability improvement, overall superiority, or
end-to-end performance improvement follows from these observations.

Locator/Worker completeness, missing-detail scope and interpretation accuracy
remain future work. Task labels cannot guarantee relevance, correct evidence
classification or scoring; omitted context can still matter. Structural source
accounting cannot prove semantic correctness. Known historical unlinked
source/disposition warnings, failures and retry costs remain visible. No new
paper experiment, mandatory closing Reflection, web search or external audit
was added or executed.

## Public packaging

A subsequent user request authorized GitHub publication. Formal sources, tests,
public documentation and the current two-policy comparison script are included.
Raw audit records, paper files, frozen copies, reconstruction metadata and local
planning notes remain on disk and are excluded from the public repository.
Evidence links into data/audits are local-workspace references. Machine-specific
paths in this delivery record have been replaced by variables for publication.
Pre-publication offline verification: **300 passed in 1.15s**.
