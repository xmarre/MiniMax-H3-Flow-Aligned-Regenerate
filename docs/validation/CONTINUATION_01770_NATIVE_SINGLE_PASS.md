# Run 01770: rejected resizing fix and native single-pass continuation

The owner reports continued frame shift and multi-frame darkening after 6dd02eb.
All three audit reports identify manifest
`97673df5b3d604c5563d867bc24a024f1d3268487e4f260ce77ac9503e27dd20`.
The half-pixel prefix/provider policy did execute. This rejects its sufficiency,
not its operator-routing tests. Different preceding-prefix hashes between runs
01768–01770 prevent treating those runs as matched causal experiments.

| Within-run measurement | Pre-high | First actual high before Flow | Final |
| --- | ---: | ---: | ---: |
| Upper45 175→176 vertical estimate, decoder pixels | -0.633 | -2.848 | -2.796 |
| Adjacent 175→176 RGB RMS | 0.03768 | 0.06786 | 0.06958 |
| Frame 181 mean luma | 0.32486 | 0.32159 | 0.32012 |

Immediate Flow guidance raises frame181 luma to0.32303 and slightly reduces
boundary motion. It is not the sole cause. The first actual high prediction
already changes both, before forecasts. PT227 independently reports zero sampled
duplicate-overlap pixel difference and unit gains for the 17-frame interior and
five-frame right-context tail; PT212 records the assembled motion change.
Learned-prefix same-frame fits remain below0.08 target latent cells. A pure
uncompensated translation or VAE overlap exposure mismatch is insufficient.
Native parity tests also do not establish trained visual acceptance.

## A different execution path

`native_target_single_pass` bypasses the cross-resolution continuation altogether.
It preserves the caller's target-grid noise, latent image, masks, conditioning,
references, complete sigma schedule and solver. One sampler lifetime runs from1
to0. There is no prefix projection, clean probe, learned transfer, independent
re-noising, intermediate solver restart or additional prefix-reference timeline.
The existing exact-mask adapter restores only caller-owned protected values at
return. Generated values remain owned by the native solver.

This differs from `same_grid_target_control`, which still splits into low,
probe and high sampler lifetimes. Removing those boundaries lets Spectrum use
one ordinary history lifetime. No forecast is forced, no acceptance threshold is
relaxed and no step budget is shortened. Full-resolution early model calls cost
more than reduced-grid calls; fewer probes and history restarts may recover some
cost, but the GPU speed tradeoff is unmeasured. The first all-generated chunk
continues to use progressive generation.

Refresh PR99 through ComfyUI Patcher and restart ComfyUI. Select
`spatial_stage_control=native_target_single_pass` in the advanced handoff node,
with `frame_gauge_repair=false` and `uniform_source_detail_transport=false`.
Use the existing normal VDN/attention and exact audio mask defaults. Existing
selectors and serialized widget positions remain available. The new selector
has no handoff operands, so the handoff-specific Local Boundary Audit cannot
supply its transfer/high breakdown; compare the actual video and native
continuation metrics instead. No extra VAE decode or model unloading is added.

This removes a demonstrated failing transition from the production execution
path. It does not prove that every native target-grid continuation is visually
clean or that progressive continuation is mathematically impossible. GPU
rendered acceptance and timing remain pending. The old fast path remains
explicitly available for further independent investigation.

## Receipt correction

Uniform-source modes did invoke the learned checkpoint in01770, although
`partitioned_transfer` incorrectly said otherwise. The selector set used by
receipt formatting omitted both uniform modes. It is now complete; this changes
reporting only, not inference.

## Validation

Pending exact-head tests and native Core sampler parity. There is no trained
model/GPU here; structural and solver tests cannot certify rendered output.
