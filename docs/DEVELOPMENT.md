# Development and maintenance

This document contains repository-maintenance, test, CI, source-pin, and packaging information. It is intentionally separate from the user-facing README.

## Scope

The package is an H3-specific ComfyUI custom-node implementation with no declared runtime Python dependencies of its own. ComfyUI supplies PyTorch and the surrounding execution environment.

The code assumes native MiniMax H3 joint audio/video behavior rather than a generic image/video diffusion contract. Runtime compatibility checks fail closed when critical H3 assumptions are absent or ambiguous.

The current package metadata is defined in `pyproject.toml`:

```text
project = comfyui-minimax-h3-flow-aligned-regenerate
python >= 3.10
build backend = hatchling
```

## Local development

From the repository root:

```bash
python -m pip install -e '.[test]' build
ruff check .
ruff format --check .
pytest
python -m compileall -q .
python -m build
```

The test extra currently installs NumPy, pytest, Ruff, and PyTorch. Runtime dependencies remain empty in `pyproject.toml` because this custom node is executed inside ComfyUI.

## CI

`.github/workflows/ci.yml` runs on pushes and pull requests.

### Python matrix

The main test job runs on Python:

```text
3.10
3.11
3.12
3.13
```

Each lane performs:

1. editable install with test/build dependencies;
2. `ruff check .`;
3. `ruff format --check .`;
4. `pytest`;
5. `python -m compileall -q .`;
6. `python -m build`;
7. isolated wheel-install/import validation.

The isolated wheel check verifies that the built package imports outside the repository checkout and exposes the expected trajectory API.

### Source-contract job

A separate CI job fetches pinned upstream/sibling repositories and runs `tools/check_source_contracts.py` against the exact revisions listed below.

This job exists because several important integration assumptions cannot be validated from this repository alone. It checks current source structure/contracts rather than relying only on comments or README claims.

## Pinned executable/source contracts

The current CI pins are:

| Source | Revision |
|---|---|
| ComfyUI | `1af040bf022569d7a890241c8dd79b296cda483f` |
| ComfyUI-Spectrum-MiniMax-H3 | `beb32dd210ef9e95520453107f158241d4f2ecf3` |
| ComfyUI-H3-Continuum | `bf25353d8bec44afea22c89717c4301ce13c2036` |
| ComfyUI-DiffAid-Patches | `ba9d9efbcf7e64c755e068cb76547d8cc85481eb` |
| ComfyUI-MiniMax-H3-RefDelta-Solver | `034e4c4c14c56bf76813cee4765e7164b0c7e0db` |
| ComfyUI-Untwisting-RoPE | `299d4c56a3f057a97b3140d2136189bcd1e7d6bb` |
| Comfyui_Minimax_h3_latent_Upscaler | `bdc670e5926bcefbe4022e17fe8b171fbfcf15de` |

MiniMax-H3 main was additionally inspected at:

```text
d21241f0a4b3acbb34c97dae47fa417b7065e438
```

The companion upscaler revision above includes the learned-handoff provider API used by `handoff_transfer=learned_3d`.

## Updating a source pin

Do not update a pin solely because a newer upstream commit exists.

For each pin update:

1. inspect the relevant upstream changes between the old and proposed revisions;
2. identify whether H3 packing, model-wrapper, sampler, mask, conditioning, geometry, RoPE, Spectrum provenance, Continuum state, or learned-provider contracts changed;
3. update the CI checkout revision;
4. update `tools/check_source_contracts.py` when the legitimate contract changed;
5. run the complete Python matrix and source-contract job;
6. run targeted runtime/decoded-media validation if the change can affect generation behavior;
7. update `docs/USAGE.md`, `docs/ARCHITECTURE.md`, `docs/BENCHMARKS.md`, or `docs/PERFORMANCE.md` when user-visible behavior/evidence changes.

A green source-contract test proves compatibility with the asserted structural contract. It does **not** prove decoded-media quality.

## What the tests establish

The repository test suite is primarily for structural and behavioral invariants such as:

- package/custom-node registration;
- H3 trajectory transaction and provenance behavior;
- flow-coordinate matching;
- sampler predictor/corrector handling;
- progressive source/target geometry transitions;
- exact handoff probes and history boundaries;
- deterministic target/private noise contracts;
- audio preservation across spatial transitions;
- mask and conditioning reconstruction;
- learned-provider API validation and target-shape enforcement;
- Spectrum actual/forecast provenance;
- reference-budget and attention-experiment guards;
- metrics counters/events and autosave behavior;
- resolution-aware sigma identity/remap behavior;
- failure-closed behavior for unsupported or ambiguous execution paths.

Passing unit tests, source-contract tests, lint, build, and packaging checks establishes implementation consistency. It does **not** establish that an experimental guidance mode improves generated media.

## Media validation policy

Decoded video and audio are the quality gate for claims about perceptual behavior.

When comparing two generation modes, keep unrelated variables fixed where possible:

- seed;
- prompt;
- reference media/conditioning;
- LoRAs;
- sampler/scheduler;
- Spectrum policy;
- DiffAid / Untwisting RoPE state;
- VAE;
- Continuum chunk/session settings;
- masks;
- audio behavior;
- crop/decode path.

Use `docs/BENCHMARKS.md` for the evidence ledger and benchmark protocol. Use `docs/PERFORMANCE.md` for timing claims.

Telemetry can prove that a correction, handoff, reset, forecast, or sigma remap executed. It cannot by itself prove a perceptual improvement.

## Documentation ownership

Keep information in the document that matches its purpose:

| Document | Purpose |
|---|---|
| `README.md` | What the project/nodes do, installation, core workflows, concise current posture |
| `docs/USAGE.md` | Detailed user wiring, settings, mode behavior, compatibility, practical starting points |
| `docs/ARCHITECTURE.md` | Internal H3 contracts and implementation design |
| `docs/RESEARCH.md` | Research-transfer rationale and conclusions |
| `docs/BENCHMARKS.md` | Media-validation ledger, test matrix, exact experiment topology |
| `docs/PERFORMANCE.md` | Performance measurements and claim boundaries |
| `CREDITS.md` | Research/implementation attribution and provenance |
| `RELEASE_NOTES.md` | Release history and user-visible release changes |
| `docs/DEVELOPMENT.md` | Development setup, CI, tests, source pins, maintenance rules |

Do not copy long benchmark logs, CI details, source SHA lists, or development commands back into the main README.

## Research-derived changes

When a new paper or repository materially influences a node, algorithm, heuristic, or default:

1. add it to `CREDITS.md`;
2. state what idea transferred and what did not;
3. update `docs/RESEARCH.md` when the transfer changes the research rationale;
4. preserve upstream copyright/license notices if source code is incorporated rather than independently reimplemented;
5. record the relevant upstream source revision when code was materially audited;
6. validate the implementation structurally and, for quality claims, with decoded media.

## Release checks

Before a release that changes runtime behavior:

1. ensure the final branch is based on current `main`;
2. run the complete CI matrix and source-contract checks;
3. verify package versioning and build output;
4. update `RELEASE_NOTES.md`;
5. update README/usage documentation only for actual user-visible behavior;
6. update benchmark/performance documents only when new evidence supports the change;
7. avoid promoting experimental behavior solely because structural tests are green.
