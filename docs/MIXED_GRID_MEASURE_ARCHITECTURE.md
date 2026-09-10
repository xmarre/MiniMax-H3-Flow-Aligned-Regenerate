# Mixed-Grid measure and temporal decode architecture

Status: research checkpoint; not yet an implementation specification. Design-only work; no production changes.

## Verified starting points

- Flow PRs #24 and #26, Sol-H3 #2 and Spectrum #106 are merged. Preserve their history; do not repurpose them.
- Open Flow #25 owns joint-AV decode adapter work; #28 owns learned-upscaler/default updates. Coordinate without modifying those PRs.
- Flow currently couples measure publication and representation reconciliation to `suffix_geometric_bridge`. Its source-warp implementation is retired.
- The mathematical candidate is source-carrier-normalized spatial key measure: prefix mass per key = source rows / prefix rows; other keys remain unit mass. Its equivalence to the trained model and media acceptance remain unproven.
- Current comfy-kitchen exposes `key_bias` on ordinary Sol attention, but only exact/sink-covered blocks consume it. An implementation must not apply bias only to the final exact logits while leaving biased keys in an unweighted pooled tail.
- Native H3 VAE uses overlapping decoder windows. Physical decoding must preserve the global window lattice and retained-frame ownership independently of sampling.

## Outstanding audit before final design

Trace chunked bias support, Sana routing/tail math, Spectrum receipt lifecycle, native VAE window constants and Continuum physical-group assembly. Compare serious operator alternatives and specify contracts, migration, validation gates and companion order.

## Source snapshots

- xmarre/MiniMax-H3-Flow-Aligned-Regenerate: `0c0a87107f3927cc9e3f87a9ada62932c2860438`
- xmarre/ComfyUI-Sol-H3: `f82ff2693be37dbad3438a30eb389d77136c0276`
- xmarre/ComfyUI-Spectrum-MiniMax-H3: `455bd357cb45637c8e852f7f448dc57b52de94f8`
- xmarre/ComfyUI-VDN-H3-Plus: `76b31323f9e09019b435237dcd8bad1e05476ce1`
- xmarre/ComfyUI-H3-Continuum-Plus: `a5b8943844594545301b20d01af5d9e3fa38ae29`
- ukr8b3g-cmyk/ComfyUI-H3-Continuum: `f41e6937476df85d5cbc4198f8a5322081b09506`
- Comfy-Org/ComfyUI: `1f641fd9337f0ec4d635a28415a8d25a8d15f753`
- Comfy-Org/comfy-kitchen: `21003fa97bf3b180393446d729ae630ceb6c2a52`
