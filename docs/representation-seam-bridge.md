# Experimental exact-overlap representation seam bridge

## Why the affine path was retired

The previous PR #24 experiment treated the remaining Continuum jump as a camera-like affine residual. Real run `metrics_00276` showed that the implementation did what it was designed to do: only the independently authorized vertical-scale axis was corrected, its final signed residual fell from about `-0.010249` to `+0.002361`, and the protected prefix remained exact. The decoded shrink/top-edge reveal nevertheless remained. That is direct evidence that a good affine fit was a proxy for the discontinuity, not its sufficient cause.

The stronger signal is the exact-prefix replacement itself. In the same run, the learned 3D upscaler's native learned-prefix→learned-suffix seam measured about `0.545` raw RMS, `0.250` low-pass RMS and `0.184` spatial-mean RMS. Hard-restoring the authoritative prefix raised those to about `0.671`, `0.341` and `0.259` respectively. The exact-prefix splice therefore amplified the seam by roughly `1.23x`, `1.37x` and `1.40x`. The existing DC bridge reduced the mean component, but the corrected splice was still about `1.10x` native raw RMS and `1.15x` native low-pass RMS.

This identifies a representation-continuity problem at the point where a jointly upscaled sequence is split into an exact native prefix plus learned suffix. It does not require assuming that the learned upscaler has a globally different latent basis, nor does it justify modifying the protected prefix.

## Exact overlap residual transplantation

The discarded learned prefix is a private calibration overlap. For its final token:

```text
D = exact_prefix_last - learned_prefix_last
```

`D` is measured between two target-grid latents representing the same protected frame. It is therefore more direct evidence than registering two different temporal frames and fitting scale/translation.

The experimental bridge decomposes it into:

```text
D_dc         = spatial_mean(D)
D_structural = D - D_dc
```

The legacy `suffix_geometric_bridge=true` switch now applies only `D_structural` to the first learned suffix token. `suffix_dc_bridge=true` retains its existing responsibility for `D_dc`. With both enabled:

```text
corrected_suffix_0 = learned_suffix_0 + D

corrected_suffix_0 - exact_prefix_last
    = learned_suffix_0 - learned_prefix_last
```

So the hard exact-prefix replacement preserves the learned upscaler's native boundary transition exactly (modulo output dtype rounding) before the high-resolution sampler resumes. There is no fitted camera model and no post-high affine warp.

## Why only suffix token 0

The exact overlap proves the residual at the boundary frame. It does not prove that the same full tensor residual should persist into future generated frames. Propagating it over several tokens would require an extrapolation model and risks creating the delayed wobble that the earlier handcrafted three-token fade demonstrated. The current bridge therefore corrects exactly one generated latent token. Fresh target-grid H3 refinement then operates on the coherent boundary state.

## Safety and no-op contract

The bridge:

- never changes protected-prefix values;
- never changes suffix token 1 or later;
- never changes audio, caller noise, masks or conditioning;
- adds no transformer call/NFE and no VAE call;
- does not alter VDN API 2 or Spectrum history/forecast accounting;
- runs in float32 for the residual calculation and returns the original video dtype;
- returns the original tensor object when disabled or when the measured zero-mean overlap residual is already matched;
- verifies that transplantation strictly reduces the measured centered boundary error, otherwise it returns the original tensor object.

The existing one-token DC bridge remains a separate feature and remains enabled by default for Mixed-Grid because it already has decoded-media validation. The representation bridge is Mixed-Grid-only and remains off by default.

## Validation gate

Structural tests can prove the algebraic boundary invariant, exact-prefix preservation, later-suffix preservation, dtype behavior, and unchanged NFE/accounting. They cannot prove perceptual success. The next matched real-media run must keep every workflow input unchanged except enabling both:

```text
suffix_dc_bridge = true
suffix_geometric_bridge = true
```

Before high refinement, `corrected_over_native_seam_rms_ratio`, `corrected_over_native_seam_lowpass_ratio`, and `corrected_over_native_seam_spatial_mean_ratio` should all be approximately `1.0`. After refinement, inspect the final seam metrics and decoded video for the original shrink/top-edge reveal and for any new one-token pulse. Do not enable this bridge by default or mark PR #24 ready until decoded media passes.
