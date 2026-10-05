# 01144: returned displacement and residual-coordinate candidate

## Evidence and scope

Run 01144 uses source 44x44, target 62x62, 47 video tokens and an exact 12-token
prefix. The handoff sigma is 0.8780487775802612. The runtime reports zero user
model/CLIP hooks and executes `h3_dense_patch_center_lattice_v2` clean transfer,
the boundary-dense attention policy and five future decode-context tokens.
The prior visually improved run does not qualify continuity across seeds/scenes.

All eight exported stage operands match their manifest hashes, byte counts,
shapes and finite-value contract. The two exported carried-prefix tokens are
byte-identical to the authoritative prefix in pre-high and final clean tensors.
The initial high video mask protects precisely these two tokens.
[Verified receipts](CONTINUATION_01144_VERIFIED.json) preserve the manifest and
selected runtime events; [native replay](CONTINUATION_01144_NATIVE_TILE.json) and
[residual comparison](CONTINUATION_01144_RESIDUAL.json) preserve numerical results.
Tensor bytes and model weights are external inputs, not repository assets.

The sampler-input prefix differs from the authoritative clean prefix by
0.00140919 RMS, maximum 0.0060673. It follows native H3 conditioning
`0.999 * exact + 0.001 * noise`, with recovered noise RMS 1.002485. The first
predicted prefix equals that conditioned input. This is intended conditioning,
not evidence of lost exact-mask ownership, and the candidate does not change it.

The provider-to-pre-high change consists of the authoritative prefix replacement
and one channel-mean-only correction on the first generated token. The latter
has RMS 0.17525899 and centered RMS approximately 3e-8. Later exported generated
tokens are unchanged by that construction. The high-guidance DC contract executes
on all six evaluations. A decoded micro-flash correction executes once, but
hardware trajectory receipts are recorded before it.

## Localized native decode experiment

Decode the six clean stages with the unchanged native decoder and one of its
actual spatial tiles: row 1, column 2 in the full 62x62 lattice. Its latent origin
is (11,22), pixel origin (176,352), and size 256x256 pixels. Analyze only the
interior y=[80,176), x=[80,192) in that tile, corresponding to global
y=[256,352), x=[432,544). Neighboring spatial tiles do not contribute to this
interior in the native blend. This is a central upper-face region, not a
whole-frame estimate.

| Stage | First-transition dx (px) | First-transition dy (px) |
| --- | ---: | ---: |
| Provider native | +0.037245 | +0.015509 |
| Provider with authoritative prefix | -0.183124 | +0.047202 |
| Pre-high after DC | -0.089682 | +0.027372 |
| First high before Flow | -0.599750 | +3.246390 |
| First high after Flow | -0.467476 | +3.736658 |
| Final | -1.590492 | +2.905442 |

These are the first pairwise phase-correlation estimates on the complete cropped
interior, without comparison-image rescaling. They establish a large local change
by the first actual high prediction before Flow. They do not establish a uniform
global translation, nor exclude smaller transfer defects elsewhere. In this
region, exact-prefix replacement alone does not explain the measured shift.

The decoder uses official unquantized FP16 checkpoint weights with FP32 CPU
arithmetic and native PyTorch SDPA. Production used INT8 ConvRot. The checkpoint
SHA256 is `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`,
from Comfy-Org/MiniMax-H3 at revision
`bf92c4091e333e69b8ca1998e0a669f15cb0832b`. Core VAE source and Continuum diagnostic
hashes are recorded in the replay JSON. The preceding temporal decoder window is
absent: the first comparison uses a raw unblended prefix frame. Full-frame
spatial blending, preceding temporal blending and production sampling are not
reproduced. This experiment is not rendered acceptance.

The original replay selected the tile through an offline `_adaptive_decode`
override that calls native `_decode_pixels` on exactly that tile. The reusable
tool now exposes the equivalent direct native tile call:

```text
python tools/decode_native_boundary_evidence.py \
  --bundle /path/to/verified-bundle \
  --core-checkout /path/to/ComfyUI \
  --continuum-checkout /path/to/ContinuumPlus \
  --vae /path/to/minimax_h3_video_vae_fp16.safetensors \
  --output /path/to/replay --dtype float32 --threads 8 --native-tile 1 2
```

The report explicitly marks the limited spatial scope and stores its interior
measurements separately from measurements over the whole tile. No production
decoder is patched by this tool.

## Residual operator inconsistency

The previous v1 residual operator is an orthogonal Gaussian refinement. For a
target group of n cells in one latent phase it emits
`source / sqrt(n) + innovation - mean(innovation_group)`. Its normalized coarse
projection preserves the source value and its marginal is white Gaussian when
the input is white Gaussian. Replacing that entire noise construction with image
interpolation would change noise variance and covariance.

However, the source effective flow residual
`r = (source_state - (1-sigma) * source_clean) / sigma`
also contains structured model drift. Passing that drift through the phase
operator can produce two peaks separated by a zero, and attenuates a constant
field according to group size. Those are unsuitable spatial transforms for
structured values. Clean transfer already uses a continuous physical dense map;
the old effective-residual path therefore couples different spatial operators.

The candidate uses the original full source-grid sampler noise N, not a newly
generated realization, and the actual model noise scale S:

```text
drift = r - S*N
target_r = S*gaussian_refinement(N) + dense_physical_map(drift)
target_state = (1-sigma)*transferred_clean + sigma*target_r
```

Gaussian innovation retains the same seed and coarse projection. The structured
component uses `h3_dense_patch_center_lattice_v2`. Same-grid transport clones r
exactly. The total residual intentionally has a different coarse projection;
receipts scope projection and Gaussian variance claims to the initial Gaussian
component. Drift depends on the model/noise trajectory, so the complete residual
is not asserted to be Gaussian. This is a coordinate-coupling correction, not
proof that it is the sole cause of the high prediction in 01144.

For the five exported generated tokens, recover the historical source residual
by the v1 normalized coarse projection of the saved target residual. Using the
full 47-token innovation RNG shape reproduces the saved target residual with
RMS error 1.954014e-7, maximum 1.907349e-6. The inferred source drift RMS is
0.023799917, and residual/initial-noise cosine is 0.99972147. Changing only the
structured operator changes target residual RMS by 0.021114627, corresponding
to 0.018539673 in handoff-state units at this sigma.

Source residual bytes were not independently exported: this recovery is an
inference conditional on the historical v1 contract. The consistency check tests
its seeded fine component, not independent knowledge of its coarse source
values. Source noise is reconstructed with the documented seed, full temporal
shape and FP32 dtype. Unexported source tokens are filled only to preserve RNG
shape; the comparison uses exported generated tokens exclusively. No H3 or
decoder counterfactual is executed by the residual comparison:

```text
python tools/analyze_handoff_residual_evidence.py \
  --bundle /path/to/verified-bundle --source-hw 44 44 \
  --historical-policy source_residual_patch_refinement_v1 \
  --output /path/to/residual-comparison.json
```

## Routing, validation and remaining acceptance

The active candidate applies only with frame-gauge repair enabled, the main
AV handoff source and a low-stage sampler that is a deterministic flow of its
initial noise (`res_multistep`, zero-churn Euler/Heun/DPM-2 and the other listed
deterministic solvers). Stochastic or unrecognized samplers add fresh white noise
that `r - S*N` would misclassify as drift; dense interpolation removes about 54%
of the variance of white input at these geometries. Those runs keep v1 for the
whole residual, and the provenance receipt names the sampler and contract. The
OFF/shadow paths retain independent noise; the v1 helper remains available for
historical replay. No provider/model/VAE evaluation,
sampler lifetime or history boundary is added. Audio, native high conditioning,
exact-prefix restoration, the one-token DC bridge and learned guidance retain
their ownership. No final latent/image warp is introduced.

Regression tests check Gaussian-only byte equality, empirical scaled covariance,
analytical impulse correspondence without parity holes on both axes/phases of
44x44->62x62, 46x40->66x58 and 36x54->50x76, constant-field DC, exact same-grid
identity, input immutability, dtype/audio/provider ownership and invalid-input
rejection before provider invocation. Tests execute the actual scheduler
selection/provenance/handoff-call AST to check original-noise and scale routing;
this is not a complete H3 sampler rerun. Replay tests distinguish inferred coarse
values from seeded fine-component consistency.

Flow #93 carries the candidate. Companion heads stay Upscaler Plus #16
`a843aff79ff410f75d133fb6cb22394d3aad27bf`, VDN Plus #37
`a1ee145d041c5a88bde5506f79876528c032f11d` and upstream Core #16783
`417b41c0350cd071d359a02637ee0b4a48b3438a`. CI checks coordinate ownership against
the pinned upscaler/Core sources. The candidate remains unqualified for rendered
continuity and GPU cost.

A matched 01144 replay must execute `source_residual_dense_drift_v2`, report
`coarse_projection_scope=initial_gaussian_component` and
`drift_lattice=h3_dense_patch_center_lattice_v2`, and retain exact-prefix/audio
ownership and one provider call. Compare the first high prediction, raw decoded
join and final boundary for displacement, doubling and tone separately. Preserve
seed, prompt, references, geometry and decoder precision. Repeat unchanged to
test reproducibility; record wall time and peak VRAM. Neither the passing
operator tests nor this limited replay closes rendered acceptance.
