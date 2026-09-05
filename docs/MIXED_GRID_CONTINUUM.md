# Mixed-grid Continuum continuation (experimental)

Use **MiniMax H3 Progressive Mixed-Grid Continuum [Experimental]** with
`handoff_transfer=learned_3d` and the companion `H3_LATENT_UPSCALER` provider.
Keep Continuum configured for the final target geometry. Chunk 1 uses the
existing low-grid/probe/learned-transfer/high-grid path. Exact-prefix chunks
use the mixed-grid path described below. This remains a draft implementation
pending matched GPU and decoded-media acceptance.

## Call contract

The low sampler owns `[1,24,T,source_h,source_w]`. Its whole-frame prefix mask
is zero. An immutable per-chunk snapshot independently owns the original
model-domain target-grid prefix and the corresponding original target-grid
sampler noise. The low carrier copy is never used as H3 transformer
conditioning.

1. A diffusion wrapper supplies a native-compatible carrier layout, preserving
   target-grid keyframes and the target-output audio width bounds. References
   keep their native geometry. The target-video segment uses a low grid.
2. Immediately before block 0, reconstruct the protected target-grid input with
   the same Native Masked H3 visual-conditioning augmentation used by ComfyUI:
   `VISUAL_COND_TIMESTEP * P_hi + (1 - VISUAL_COND_TIMESTEP) * N_hi`, currently
   `0.999 * P_hi + 0.001 * N_hi`. Patchify/project that target-grid prefix
   independently and replace the carrier prefix. The authoritative clean prefix
   itself is never spatially resized for H3 conditioning. Keep genuine
   source-grid suffix patch embeddings. All transformer blocks receive the mixed
   layout, mixed RoPE, and expanded native per-row modulation indices.
3. Prefix positions are the native target grid. Suffix positions are the native
   source grid, sliced from the full global temporal timeline. This preserves
   the native five-slot temporal-span period across any prefix length. Protected
   prefix rows retain native pinned timestep/AdaLN labels while suffix rows use
   the active sampler timestep.
4. After the final block, discard its target-prefix outputs and insert zero
   low-carrier hidden rows. Keep suffix and non-video hidden rows exactly.
   Spectrum observes this regular carrier layout. Native final projection and
   unpatchification therefore receive a valid low-grid sequence. The diffusion
   wrapper discards carrier-prefix velocity, including on forecast calls.
5. The exact handoff probe uses this same mixed model in a separate sampler
   lifetime. The learned upscaler receives clean suffix latents and a temporary
   low-grid copy of all original prefix frames. Its 3D attention and GroupNorm
   provide no demonstrated finite-context truncation rule.
6. Discard upscaled prefix output. Restore the original target-grid prefix in
   the transferred state, reconstruct high-stage noise with the existing H3
   law, preserve the original inpaint noise/mask, and start a fresh high-stage
   sampler. Verify its first call is actual and its returned prefix is exact.

There is no interpolation of generated hidden rows between transformer blocks.
The ordinary conservative Target Input node and the previous target-sparse
experimental node remain available as control arms.

## VDN contract

VDN-H3 must advertise external-sequence capability **2**. API 1 remains supported
for the existing target-sparse path. Mixed calls use the existing key
`vdn_h3_external_sequence_v1` with `api=2`, `mode=dense_gate_no_linear`, and
`topology=mixed_grid_low_suffix`.

| Field | Meaning |
|---|---|
| `native_sequence_rows` | Published regular low-carrier sequence length |
| `sequence_rows` | Actual mixed sequence length |
| `video_start` | Unchanged non-video row count |
| `temporal`, `prefix_t` | Full and protected temporal lengths |
| `source_rows_per_frame` | Native source patch rows per frame |
| `prefix_rows_per_frame` | Native target patch rows per prefix frame |

VDN validates both layout identities and the complete row-count equation,
requires matching explicit RoPE, then uses learned gated dense attention with
geometry-dependent windows/linear complement disabled. The high stage has no
external contract and resumes ordinary VDN. Missing or stale contracts fail
closed. Flow checks VDN capability when applying the node and again before
sampling the assembled workflow.

## Source audit

Audited native ComfyUI `250b2e9551a7bc7a8ebb5beb07e0fecd2983e04a` against the
existing pinned `1af040bf022569d7a890241c8dd79b296cda483f`. The relevant change
adds the allocation-compiler wrapper and prefetch scope arguments; packing,
modulation, and final projection contracts remain equivalent. The existing
pin remains the regression oracle; GPU compiler behavior still needs testing.
The source-contract lane also pins MiniMaxH3 `scale_latent_inpaint` semantics so
changes to the protected-prefix `VISUAL_COND_TIMESTEP` noise augmentation fail
CI instead of silently changing mixed-grid conditioning.

Spectrum `beb32dd210ef9e95520453107f158241d4f2ecf3` observes the last block
outside pre-existing replacements, which permits Flow to restore carrier
geometry before observation. Its state-input capture also uses carrier
geometry. DiffAid `ba9d9efbcf7e64c755e068cb76547d8cc85481eb` derives language
ranges from modulation segments; those segments are expanded coherently.
Untwist `299d4c56a3f057a97b3140d2136189bcd1e7d6bb` targets reference ranges,
which remain before the changed video segment. Continuum
`bf25353d8bec44afea22c89717c4301ce13c2036` supplies whole-frame zero prefixes.
The upscaler's provider/network files are unchanged between existing pin
`bdc670e5926bcefbe4022e17fe8b171fbfcf15de` and current
`0744761a2021ec459206ad5f5e1d0e1ff310342a`.

## Validation and remaining acceptance

CPU tests exercise native packing/positions, the real native `_forward` with
small test components, independence from poisoned low-prefix carrier values,
dependence on both the authoritative target prefix and its native target-grid
sampler-noise augmentation, exact suffix projection rows, modulation expansion,
carrier observation, low/probe/high contract lifetimes, exception cleanup, and
replacement of poisoned probe-prefix context before learned transfer. These
establish structural behavior only.

Run matched conservative/mixed two-chunk A/B renders, then at least four
chunks, holding prompt, seed, references, geometries, checkpoints, VAE,
sampler/scheduler, VDN, Spectrum, DiffAid, and Untwist settings fixed. Require
exact prefixes, no seam/flash/motion/identity/audio regression, independent
mixed stages for every continuation, and no stale histories or layouts.
Inspect `mixed_grid_plan`, `mixed_grid_transformer`, `mixed_grid_transfer`, and
`mixed_grid_complete` alongside existing low/probe/upscale/transfer/high wall
events. At T=62, prefix=12, target=48x64, source=34x44, require 27,916 rows.

The reported low-stage preview lattice remains unclassified. Save and explicitly
VAE-decode a low-stage latent, compare with the live preview, then use matched
VDN on/off runs if the decoded latent is patterned. Compare chunk 1 with the
mixed continuation suffix. Any contamination of final media blocks acceptance.
The live preview alone cannot establish decoded latent quality.

No GPU timing, quality, peak-memory, four-chunk, or allocation-compiler result
is claimed for this implementation. Keep both PRs draft; do not merge/release.
