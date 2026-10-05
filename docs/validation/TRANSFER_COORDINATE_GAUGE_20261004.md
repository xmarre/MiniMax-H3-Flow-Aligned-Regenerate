# Dense transfer and continuation evidence

The later [01144 investigation](CONTINUATION_01144.md) reproduces displacement
with dense v2 clean transfer active, localizes a large local change to the first
high prediction, and adds a separate Gaussian/drift residual transport candidate.
The operator findings below remain valid; they do not establish a universal
continuity fix.

## Runtime evidence

`metrics_01138_.json`, `metrics_01139_.json` and their shared runtime log report
zero user model/CLIP hooks and execute the physical-lattice v1 learned transfer.
The reported first run is visually better; the second run has a doubled
continuation. Seed, prompt, first reference and orientation change between the
runs. Their order alone does not establish a stale cache or adapter defect.

| Run | Source -> target latent H/W | Learned upper45 / full | Restored upper45 / full | Final upper45 / full |
| --- | --- | --- | --- | --- |
| 01138 | 54x36 -> 76x50 | 0.637382 / 0.452741 | 0.877536 / 2.032663 | 0.848097 / 0.492536 |
| 01139 | 36x54 -> 50x76 | 0.888911 / 0.905882 | 1.135982 / 1.025767 | 1.928787 / 1.406685 |

These are boundary-versus-pre-motion norms in target latent cells. They are
translation diagnostics, not a measure of double-image strength. Source-low
full norms are 0.290996 and 0.814620 in their respective source cells. VDN #37's
three-frame boundary-dense policy executes in both runs. A global same-frame
prefix fit returns approximately zero affine displacement in both; that does
not rule out parity-dependent local distortion. The exported tensor hashes
are present, but the tensor bytes and decoded videos are not supplied here.

## Reproduced operator defect

The v1 transfer splits dense Conv3d feature maps into four even/odd spatial
lanes, resamples each lane on the token lattice, then interleaves them. This
is not a continuous affine transform of dense cells. On 36x54->50x76, source
coordinate increments alternate between 1.0 and 0.4305 cells. At 36->50, a
single nonzero source row produces equal target peaks at rows 24 and 26,
separated by a zero. Both axes and both phases reproduce the defect.

The original coordinate experiment proved that two sampling conventions differ.
Its successful round trip did not qualify the individual operators or the
trained decoder. The v1 implementation still passes four round-trip tests but
fails all 14 added impulse/spacing regressions. This establishes a concrete
doubling mechanism; it does not prove that it explains every 01139 artifact.

## Corrected construction

`h3_dense_patch_center_lattice_v2` extends each native patch center into two
uniformly spaced dense-cell centers. For axis length n and spatial area A:

`coordinate(i) = 16*(1-n/sqrt(A)) + (i-0.5)*32/sqrt(A)`.

The mean coordinate of each adjacent pair equals H3's native endpoint-excluded
patch position. Flow projects its source prefix carrier and the upscaler
transports encoder features with the identical continuous map. The transform
remains before learned decoder convolutions. No fitted correction or final
latent/image warp is added. Authoritative target-prefix restoration remains
exact; attention token/RoPE construction, audio, sampler schedule, handoff
residual policy and the VDN boundary-dense scope are unchanged.

The explicit provider capability is now `h3_patch_lattice_api=2`. Flow rejects
v1 before sampling to prevent mismatched companion maps. Ordinary upscaling
retains the trained half-pixel path; same-grid identity and the named bicubic
diagnostic retain their paths. Interpolation scratch is bounded per chunk.

Tests cover unimodal edge support, dense coordinate spacing, native patch
centers, both orientations, transport in both directions, batch/time/dtype
ownership and repeated shape changes through one cached network. Cross-repo
tests compare the actual Flow and upscaler transforms. These establish
structural correctness, not trained-checkpoint visual acceptance.

## Integration and runtime acceptance

Core integration now uses upstream [ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783)
at `2c1e08938a7aee39cc681dea0b050e039e363a5d`. Its fused-adapter repair remains
independent. The standalone branch does not contain #16720's workspace
optimization, so CI runs the adapter/MiniMax and sibling lifecycle contracts
against the replacement head rather than requiring the absent workspace test.

Update Flow #93 and Upscaler Plus #16 together through their PR overlays.
Replay the failing landscape workflow, then repeat without changing seed,
prompt, references or dimensions. Require:

- `spatial_lattice=h3_dense_patch_center_lattice_v2` at `encoder_to_decoder`;
- matching v2 prefix-projection and transfer receipts, one provider call and
  unchanged exact-prefix/audio ownership;
- measured provider/restored/first-high/final trajectories and decoded evidence
  that tests doubling, boundary motion and tone separately;
- wall time and peak VRAM without additional sampler/provider/VAE evaluations.

The trained upscaler was trained on half-pixel interpolation. Its response to
the corrected physical map remains an empirical risk. Neither passing tests
nor zero affine prefix fits can replace the rendered replay.
