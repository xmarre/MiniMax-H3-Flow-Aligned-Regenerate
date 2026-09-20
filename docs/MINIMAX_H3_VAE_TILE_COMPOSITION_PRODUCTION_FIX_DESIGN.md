# MiniMax-H3 VAE tile composition production design

Status: investigation checkpoint; not an implementation specification yet.

Scope: spatial video-VAE composition only. No production code changes. Preserve Flow #49/#52 and Continuum #24/#28; continuation frame shift remains separate.

Source baseline: Flow main b659b311fa548c7e3275db0e6f0a05c037dac730; diagnostic #52 c7322b9f1c983ff12566337842b97d8390993344 on #49 d1b32f530b4cc2298eba68f08c2da44a094c7cfe. Core master c8ed2c8ce957475459731135c4ca31c6856a4542; independent ai-toolkit port 8fa15e356939a922b8fd3307610bf03100c803f6.

Observed source issue requiring proof: tiled_decode saves raw neighbor tails, vertically blends the current tile, then horizontally blends against the raw left neighbor. This drops the diagonal tile contribution and can reset an existing vertical blend at the next horizontal write boundary. The independent ai-toolkit _stitch_tiles has the same ordering; agreement with that port is not proof of correct two-dimensional composition.

Next gates: reproduce the composition counterexample using extracted live methods, verify 00536 output provenance and lattice movement, audit Core history and official source, specify the smallest correction and runtime acceptance. No safe halo width or complete visual fix is established.
