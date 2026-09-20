# Exact-prefix continuation evidence audit checkpoint

Status: investigation checkpoint, not an implementation specification. No production source changes.

The 2026-09-20 live API audit confirms Flow #49 -> #54 -> #56 -> #57 -> #58, VDN #19 -> #22 -> #23 -> #26 -> #27 -> #28, Sol #15, Continuum #24 -> #28 -> #29 at the supplied exact heads. Flow #58 remains one commit on #57. This design branch starts at Flow #58 8118a293d9e98fc43fcb2f46e3609c8fbddbde87.

The retrieved 00545 and 00546 logs match PT209 source digest, both compiled physical text hashes, physical ranges [0,175) and [136,345), and PT210 seven-reference hashes/order. The embedded MP4 execution graphs differ in Flow diagnostic selectors and wildcard processor seed; wildcard source and populated strings match. This does not prove conditioning tensors or the complete runtime are byte-identical.

Visual inspection reveals first-chunk fish motion differs before the continuation intervention. The source outer wrapper delegates no-prefix chunks to the existing progressive runtime, so a continuation-only mechanism cannot directly explain that first-chunk difference. This is an unresolved matched-control limitation.

The 00546 PT212 upper45 first-three median is -0.1194, but its first pair dy is +2.49175 and dx is +16.56507. The video transitions from lizard to iguana near the physical continuation boundary. A phase-correlation median across different scenes cannot prove preservation of same-shot motion. Exact frame indexing and output duration mapping are still under investigation. The user-reported acceptable audio remains an acceptance constraint.

Current heterogeneous transformer source injects target-prefix embeddings only at layer zero, then evolves prefix hidden rows through every block; it discards prefix outputs only after the last block. Exact sampler ownership does not imply frozen transformer context. Any suffix-only-Q design must specify the layerwise context recurrence and learned-linear frame statistics; rectangular attention support alone is insufficient.

The installed Core abbreviated SHA 673afefe cannot be resolved through Comfy-Org/ComfyUI or xmarre/ComfyUI. Public source is a comparison reference, not proof of installed patcher/stack bytes. The final specification must retain this provenance gap.
