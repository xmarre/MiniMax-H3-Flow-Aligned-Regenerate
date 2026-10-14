# Legacy continuation controls

These settings preserve older workflows. Use Native Source Continuation for new workflows.

## Other continuation modes (retained)

The following `spatial_stage_control` values remain available for existing
workflows and controlled comparisons. They are not the recommended path and may
be retired in a later release after a separate review.

- `same_grid_target_control` (the default of v0.3.9 and v0.3.10): low/probe and
  high operate at the target video resolution with identity clean/residual
  handoff. It costs more low/probe time than a reduced-grid stage.
- `progressive_low_to_high`: low/probe runs a reduced-grid suffix next to the
  exact target-grid prefix, with learned transfer of the suffix.
- `progressive_target_band`: described below, with its comparison controls.

### Target-band continuation

`spatial_stage_control=progressive_target_band` is an experimental alternative
for exact-prefix continuations. The protected prefix and the first
`target_band_tokens` generated H3 temporal latent tokens after it stay on the
target grid through low/probe. Every later generated token runs on the
configured reduced grid. At the handoff the band keeps its own target-grid clean
prediction, and the remaining tokens use the learned 3D transfer of
`progressive_low_to_high`. Every generated token, band included, is then
re-noised with the same independent Gaussian noise regardless of
`frame_gauge_repair`. The band's raw low-stage sampler state is not resumed, and
source residual transport is not used in this mode.

The low/probe sampler state stays on the uniform target grid. Each reduced-grid
token stores its values in a source-sized window of its frame, and the rest of
that frame is masked padding excluded from the transformer blocks. Stochastic
samplers may add noise to raw padding; it is masked before model calls and
discarded when forming the reduced-grid view. The transformer blocks receive
`[target-grid prefix | target-grid band | reduced-grid tail]`. Their video row
count falls between the reduced-grid and target-grid controls. The video input
projection and final layer still process the target-sized carrier. Wall time
and peak memory therefore require measurement.
Spectrum forecasting, the native final layer and Flow guidance keep their usual
contracts. Guidance binds to the actual high-stage clean operand (the band's own
prediction plus the learned tail).

High refinement runs on one uniform target grid. Only the original carried
prefix remains protected; the band stays fully generated. Every local query
group, including protected-prefix and band frames, uses the selected backend's
native selection, as in a first chunk.

With `suffix_dc_bridge=true`, the one-token channel-mean bridge applies to the
first reduced-grid token after the band and is measured against the target-grid
head. The band must leave at least one generated token on the reduced grid.
The mode requires `handoff_transfer_control=learned_3d`,
`prefix_transformer_context=exact_target_partitioned`, the main handoff and
guidance sources.
It requires VDN-H3-Plus transport that accepts a target-grid native partition
carrier. When Sol attention is selected, its history contract must also accept
that carrier. Unsupported combinations fail before sampling.

With `capture_boundary_witness=true`, target-band continuation saves CPU copies
of the low/probe carrier, uniform reduced-grid provider input, provider output,
pre-high state, first actual high prediction before and after Flow, sampler
input, mask and final state. The bundle is written under
`output/h3_flow_regenerate/residual_geometry` for Local Boundary Audit.
This uses the existing stage-evidence path with a geometry-derived CPU byte
budget (see below) and requires native decoder timing. It adds copies and disk I/O without model,
provider or VAE calls. It does not enable the VDN feature sink or change
`frame_gauge_residual_mode`; capture remains off by default.

The default `vdn_temporal_carrier_policy=native_grid_then_map_v1` applies each
neighbor's spatial short-convolution on that neighbor's grid before resampling
it for a temporal tap. The opt-in `destination_grid_stencil_v1` resamples the
raw projected neighbor onto the receiving frame's grid before applying the
checkpoint spatial convolution. This keeps a temporal tap's spatial stencil
on the receiving grid while retaining cross-grid temporal coupling. Same-grid
taps are unchanged. Use `vdn_linear_diagnostic=normal` with this policy; it
cannot be combined with cross-grid tap suppression. Paired VDN checkpoint
capability and completed carrier work are verified. High refinement restores
the native policy. The destination policy uses the batched linear path rather
than the fused short-convolution path, so wall time requires measurement.
Rendered quality of this target-band combination remains unqualified.

`vdn_linear_diagnostic=suppress_cross_grid_temporal_taps` remains available as
an ablation: it removes only temporal taps crossing the grid boundary, retaining
same-grid taps and the learned linear complement. Improvement under suppression
does not establish that the destination policy produces the same improvement.

For stage localization, set `frame_gauge_residual_mode=measure`. Target-band
runs export the native decoder window and full video snapshots of the low/probe
carrier, the uniform source-grid provider input, the learned provider output
before the band splice, the high-stage entry clean state, the first actual high
prediction before/after Flow, and the final clean video. The manifest identifies
the protected prefix, band/tail edge, tensor grids and stage domains. The native
low/probe carrier contains reduced-grid storage windows and padding and must
not be decoded as ordinary target-grid video. The CPU tensor budget is derived
from the target grid and timeline: every named full-video stage, its decoder
window, the prefix and the bounded high-call windows, with a 256 MiB floor.
Host memory for captured tensors therefore grows with resolution and length
(about 486 MiB for a 96×64-latent, 62-token timeline), plus a transient copy
per tensor during export. A snapshot outside the budgeted stage set, or one
that exceeds the budget, reports an error. This opt-in mode adds CPU
copies and output-file I/O, with no additional model/provider/VAE evaluations.
The paired-prefix frame gauge remains inactive in band mode.

The boundary quality of this mode has not been qualified. Compare its join, the
band's far edge, tone and audio against `same_grid_target_control` before using
it for production output. The low/probe saving depends on the reduced-grid size:
a larger `source_scale` (or larger explicit source size) leaves less to save, and
the video-row reduction is not a wall-time measurement.

#### Target-band comparison controls

Two selectors apply only to `progressive_target_band`. Both default to the
behaviour described above and publish no option leaf at their defaults. Either
can be enabled independently. They are comparison controls, not qualified
quality fixes. Requesting either one disables the target-grid fallback: an
unsupported combination fails before sampling instead of running a different
path.

`target_band_handoff_state=carry_raw_band` resumes the band tokens from their
actual low/probe sampler state at the handoff sigma. This is the same entry
that `same_grid_target_control` uses for every generated token. The protected
prefix, the re-noised tail (including the DC-bridged first tail token), audio
and the high-stage masks are unchanged. The band stays fully generated in high.
This control cannot change anything that low/probe has already produced.

`target_band_context=domain_uniform_v1` changes only the low/probe transformer
evaluation. Each model call evaluates two uniform-grid hidden streams through
every block:

- Target stream: text, references, the audio covering the head's duration, the
  exact target-grid prefix and the target-grid band. Its layout and RoPE rows
  are those of a native clip of `prefix + band` tokens on the target grid.
- Source stream: text, references, all audio, the prefix and band projected to
  the reduced grid, and the reduced-grid tail. Its layout and RoPE rows are
  those of the native chunk on the reduced grid.

Each stream owns its own conditioning rows. Within a model call, no attention
key, VDN local window, global or anchor query, linear-complement state, or
modulation/MLP row is shared between the streams. VDN and Sol see an ordinary
equal-grid partition contract per stream, so key measures are unity. Row and
column anchors are those of each stream's clip; the target stream's last band
frame is an anchor of that stream. Every local query group in both streams uses
the selected backend's native selection, as in a non-partitioned clip.

The streams communicate only through the sampler state between model calls.
On the first low call the source stream reads the band's current state
projected with the H3 physical lattice; a fixed complementary noise field
restores the nominal per-cell variance of `sigma * noise_scale` after the convex
resample. Every later low/probe call holds the band as known frames in the
source stream: the target stream's latest clean band estimate
(`x - sigma * v`), projected and conditioned exactly like the protected prefix.
This gives the tail the previous call's clean band estimate as context; it
does not give the target stream the tail's hidden states. Both streams read
the shared audio state. The band
velocity comes from the target stream; tail and audio velocity come from the
source stream. The protected prefix and padding remain excluded. The band
cannot see tail hidden states within a call, so band content is produced as the
end of a shorter clip. The `partitioned_target_band_domain_plan` receipt reports
whether its length matches H3's `5k + 2` temporal pattern as
`target_stream_native_clip_length`. Other supported band lengths remain valid;
matching this pattern does not establish rendered continuity.

The mode requires `vdn_linear_diagnostic=normal`, `softmax_diagnostic=normal`
and `vdn_temporal_carrier_policy=native_grid_then_map_v1`: no cross-grid taps
or non-unit key measures exist for those selectors to act on. Keyframe-anchored
layouts and audio whose duration differs from the native video/audio relation
are rejected. The audio-position selector does not apply because each stream
uses its own native positions; selecting `source_carrier` leaves this mode's
output unchanged and does not request mixed-grid execution verification.
Audio-only references also use each receiving stream's native spatial
endpoints while preserving reference times. Duration and keyframe restrictions
are checked before opening the low-stage sampler lifetime.
VDN-H3-Plus must accept domain-stream API 1. If Sol attention is selected,
its history contract must also recognize that API. Completed work is verified: every
low/probe stage must run both streams through every transformer block.

The learned band/tail handoff and the high stage are unchanged. This control
therefore does not make the band and the upscaled tail one coherent trajectory.
The source stream adds its projected head rows and both streams carry their own
conditioning rows. Attention cost depends on the sum of the two sequences'
squared lengths instead of the square of one combined length. Paired VDN weights
that are streamed per block may be fetched twice. Wall time and peak memory
require measurement.

`target_band_context=domain_uniform_all_stages_v1` preserves that isolated head
operator during high refinement. Low/probe and the learned handoff are identical
to `domain_uniform_v1`. High also evaluates two separate hidden streams through
every block: the same short prefix/band clip with head-duration audio, and the
full clip with all audio. Both streams use the target grid. The full stream reads
the band's current native sampler state directly; high adds no projection or
noise complement. Band velocity comes from the short stream; tail and audio
velocity come from the full stream. The band remains generated, with the native
final layer, original masks and selected handoff-state policy. Each completed
high stage also emits a verified `partitioned_target_band_domain_stage` event.

This experimental mode removes the within-call change from isolated head
conditioning to full-clip conditioning at the start of high. It has not been
validated for rendered tone continuity. Shared audio state and Flow guidance can
still transmit information between calls or frames, and the learned band/tail
splice remains. High now processes the extra short head and duplicated
conditioning rows, and streamed VDN weights may be fetched twice per block.
Wall time and peak memory need measurement. Existing modes and defaults retain
their previous execution paths.

### Experimental uniform-source detail transport

`uniform_source_detail_transport=false` is the default for new nodes, saved
workflows that omit the widget, and the canonical overlay. Keep it off when
checking `progressive_uniform_source_exact_context` alone.

Explicitly enabling it adds a weighted copy of the last exact-prefix frame's
exact-minus-learned latent residual to generated frames before high refinement.
Weights depend on local learned change and residual persistence fitted on
prefix-frame pairs. A separate set of prefix pairs must improve before the
transport applies. This checks prefix residual prediction; it does not establish
suffix motion safety or rendered quality. Fine detail can move while its reduced
projection stays unchanged, causing the transport to carry stale detail or an
overlay. The weights can be fractional. GPU quality remains unvalidated.

It applies to both uniform-source modes and adds no model, upscaler or VAE call,
though calibration has compute and memory costs. When applied, it supersedes the
one-token suffix DC bridge. The `partitioned_uniform_source_detail_transport`
receipt records the decision, prefix error ratio, weights and suffix weight mean.
Use this only as a separate experimental comparison on identical inputs and
seed; inspect moving detail and the actual assembled join.

### Suffix DC bridge

`suffix_dc_bridge` controls the one-token channel-mean bridge on learned-transfer
continuations. It is off in the recommended profile. When enabled, the first
transferred generated token receives the per-channel spatial-mean offset that the
learned transfer produced on its carried context. When disabled, that token stays
exactly as transferred, and transfer receipts report the bridge as disabled.
Same-grid continuation measures a zero offset either way. Workflows saved with
an older release store `true`, the previous default.

