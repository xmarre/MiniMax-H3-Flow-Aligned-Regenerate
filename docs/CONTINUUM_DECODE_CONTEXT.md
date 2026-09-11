# Continuum video decode context

Use **MiniMax H3 Continuum Decode Context** immediately before the video VAE
decoder, after the final sampling, refinement, or latent-upscale operation:

1. Final `video_latents` list, or a final native joint H3 AV LATENT list, → Decode Context `video_latents`.
2. Corresponding Continuum `assembly_plan` → Decode Context `assembly_plan`.
3. Decode Context `video_latents` → existing **Video VAE Decode**.
4. Decoded images → existing **Continuum Assemble**.

The input can therefore be either of the representations used by the coordinated
Continuum/refinement stack:

- Continuum's split video LATENTs with plain `samples: [1,24,T,H,W]`; or
- native joint H3 sampler output with `samples = NestedTensor([video, audio])`,
  including the output of **MiniMax H3 Latent Upscaler + Refine (3D)**.

For native joint AV input, Decode Context validates both members and exposes only
the existing video tensor through a decode-only LATENT view. It does not copy or
modify the accepted audio member, sampling state, or source LATENT. The normal
Continuum audio path stays unchanged.

Keep the original assembly plan connected to Assemble. The context node's output
is **decode-only**: do not feed it back into sampling, latent refinement/upscaling,
Run Storage, continuation state, or the audio path.

This works independently of the progressive node: conservative Target Input,
target-sparse, mixed-grid, and an external learned-upscale/refine second pass can
all feed the same decode boundary. No H3 transformer evaluation is added. The
selected VAE remains external.

## Ownership and migration

This Flow node is the current compatibility adapter for the released external
Video VAE Decode + Continuum Assemble topology. The canonical long-term owner of
physical-group decode views is Continuum itself, as specified by Flow design PR
#29. Once supported Continuum releases expose the shared decode-view contract,
this node should delegate to that implementation while keeping the existing node
ID loadable for saved workflows. Do not duplicate or double-extend an already
prepared Continuum decode view.

The present adapter deliberately does **not** claim the complete future Continuum
cache/revision contract from that design. It handles the released schema-1
assembly-plan boundary, exact adjacent physical overlaps, and native joint-AV to
video-only normalization described below. Review/reroll neighbor invalidation,
short-group forward gathering, decoder-capability discovery, and integrated
decode receipts belong to the Continuum-owned implementation rather than being
silently approximated here.

## Boundary defect

ComfyUI's native `MiniMaxH3VideoVAE.decode_temporal` decodes seven latent tokens
per window with a five-token stride. It retains 17 frames from each window's
first part and blends five frames with the preceding window's terminal part.
When a sequence ends, its last five frames are emitted without a following
window. Independently decoding the next Continuum chunk and trimming its
protected prefix does not retroactively supply that missing window.

Consequently, bit-identical protected latents do not make independent chunk
decoding equivalent to continuous decoding. The difference occupies the five
frames immediately before the assembly join. This mechanism is shared by
progressive paths and exists without any mixed-grid attention or learned-prefix
replacement. It is separate from a discontinuity already present in generated
latents, which this node does not repair.

## Correction

For each adjacent physical decode group, verify that the next prefix matches
the previous latent tail exactly and that their geometry, dtype, and device
agree. The overlap must have the native 5/22/39/... frame phase. Append the next
five **generated** latent tokens to a newly allocated temporary copy of the
preceding chunk. The VAE can now evaluate the formerly missing window. Assemble
already trims each decoded result to its original `total_frames`, so the 17
extra output frames are discarded and the timeline length stays unchanged.

For existing split-video input, a terminal or otherwise unextended chunk keeps
its original LATENT dictionary identity. For native joint AV input, even an
unextended chunk is returned as a minimal zero-copy video-only decode view so a
video VAE never receives the AV `NestedTensor`. Accepted source latents, masks,
the source AV wrapper, audio tensors, and the plan remain unchanged. Existing
physical `decode_groups` take precedence over logical chunk entries, including
terminal-merged plans.

Non-exact overlaps (for example, independently upscaled/refined duplicate
prefixes or Guide-mode regeneration) remain unextended and are identified in
the report. Native joint AV input is still normalized to its video member so it
remains a valid video-decode input. No approximate equality, implicit resizing,
color matching, or replacement of generated suffix latents is used. Such
boundaries require a different correction; a `0/N` report means the temporal
right-context fix did not activate.

## Evidence and limits

The regression oracle executes the pinned native VAE's actual temporal window,
padding, blending, and write methods with a context-sensitive stand-in for its
learned pixel decoder. Four-chunk tests with 5-, 22-, and 39-frame overlaps:

- reproduce the previous five-frame boundary discrepancy;
- produce bit-identical results to a single continuous latent decode with
  right context;
- preserve accepted latent values, plan contents, and output duration.

Additional contract tests feed the native joint `NestedTensor([video, audio])`
shape produced by the integrated H3 refiner and verify that Decode Context:

- returns plain 24-channel video LATENTs suitable for Video VAE Decode;
- keeps terminal video extraction zero-copy;
- leaves the source AV wrapper, audio values, masks, and assembly plan untouched;
- fails closed on malformed AV member counts or audio shape.

Source: ComfyUI `1af040bf022569d7a890241c8dd79b296cda483f`,
`comfy/ldm/minimax/vae.py`; Continuum
`bf25353d8bec44afea22c89717c4301ce13c2036`, `v3/assembly.py`, `hardening.py`,
and `temporal.py`. The native source oracle runs in CI.

The structural decode-context correction is proven by those oracles. Actual
checkpoint media and speed are not established by a synthetic pixel decoder.
There is one extra seven-token VAE window and 17 discarded decoded frames per
corrected boundary, plus temporary extended latent storage. Joint-AV
normalization itself adds no H3 sampling or VAE work and does not copy the
terminal video tensor. The implementation does not combine the entire sequence
into a single large decode call. TRT or other replacement VAEs must preserve
native H3 temporal window semantics for the equivalence argument to apply.
Inspect decoded media with the actual VAE; remaining generation-domain flashing
and learned-prefix splice behavior remain separate questions.
