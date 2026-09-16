# Mixed-Grid Continuum — deprecated compatibility path

**MiniMax H3 Progressive Mixed-Grid Continuum [Deprecated]** is retained for one compatibility release so existing serialized workflows continue to load with their original semantics.

It is no longer the recommended target-input/Continuum production path. New workflows should use **MiniMax H3 Progressive Handoff (Target Input)**.

The old node ID `H3ProgressiveMixedGridHandoff` is intentionally not remapped to Target Input. Remapping would silently change saved workflow behavior. During the compatibility window the original Mixed-Grid implementation remains executable, but it is not a production release gate and no compatibility render is required.

## Historical contract

Mixed-Grid was designed for exact-prefix continuation where the protected prefix remained on the target spatial grid while the generated suffix ran on a genuine smaller sampler grid. Its execution path:

1. retained the original target-grid protected prefix as transformer conditioning;
2. generated the new suffix on the low grid;
3. executed a mixed target-prefix/source-suffix transformer sequence;
4. performed an exact handoff probe;
5. supplied low-grid clean video to the learned 3D latent upscaler;
6. restored the authoritative target-grid prefix;
7. applied target-side seam reconciliation/DC correction where enabled;
8. started a fresh full target-grid sampler lifetime.

The historical path required `learned_3d` transfer and, when VDN was active, its external mixed-sequence contract.

## Historical defaults

The serialized compatibility node retains these values unchanged:

```text
source_mode             = scale
source_scale            = 0.70
source_width            = 864
source_height           = 640
handoff_coordinate      = 0.35
handoff_selection       = fixed
guidance_mode           = direction+temporal
direction_weight        = 0.25
acceleration_weight     = 0.25
consistency_weight      = 0.25
low_frequency_cutoff    = 0.25
temporal_weight         = 0.20
handoff_transfer        = learned_3d
suffix_dc_bridge        = true
suffix_geometric_bridge = true
```

These values are compatibility state, not current production recommendations.

## Why production moved away from Mixed-Grid

Later exact-prefix debugging established a simpler contract with stronger ownership: the ordinary Target Input node conservatively stays on the final target grid whenever exact protected video is present. That path avoids a private low-grid continuation lifetime, mixed-resolution transformer ownership, learned transfer at the exact-prefix boundary, and the corresponding cross-repository weighted-measure/external-sequence contracts.

The Target Input fallback also gained the independently validated four-audio-tick guided overlap while keeping the original exact video/audio output authoritative. This became the standard continuation behavior.

Retiring Mixed-Grid from production therefore removes an unnecessary acceptance surface; it does not retroactively invalidate the historical Mixed-Grid experiments or their measured results.

## Historical seam work

Three seam mechanisms were investigated separately:

- **one-token suffix DC bridge** — corrected a measured boundary DC mismatch on the Mixed-Grid transfer path;
- **target exact-overlap representation reconciliation** — corrected the learned-prefix replacement mismatch but did not alone explain every framing artifact;
- **attention-measure normalization** — compensated for unequal physical K/V carrier density between target-grid prefix frames and source-grid suffix frames.

The source-space affine/trajectory warp family was rejected: finite correction horizons moved the discontinuity to the corrected-to-untouched transition rather than eliminating it.

See [mixed-grid-seam-repair.md](mixed-grid-seam-repair.md) and historical `RELEASE_NOTES.md` for the evidence chain.

## Compatibility boundaries

During the deprecation release:

- the node class and node ID remain registered;
- its historical inputs/defaults remain unchanged;
- existing saved graphs are not reinterpreted as Target Input;
- Mixed-Grid-specific runtime machinery remains only to make compatibility real;
- new documentation/workflow overlays do not use Mixed-Grid;
- Mixed-Grid companion PRs are not part of the current production Patcher topology;
- structural compatibility tests may run, but decoded-media compatibility is not a release requirement.

After the compatibility window, the Mixed-Grid runtime/measure machinery, bridges, external-sequence integration, and dedicated tests/docs can be removed in a separate cleanup change while preserving this historical record in release history.
