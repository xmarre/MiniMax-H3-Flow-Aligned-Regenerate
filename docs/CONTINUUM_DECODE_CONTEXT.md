# Continuum video decode context

Use **MiniMax H3 Continuum Decode Context** immediately before the video VAE
decoder, after the final sampling, refinement, or latent-upscale operation:

1. Final `video_latents` list → Decode Context `video_latents`.
2. Corresponding Continuum `assembly_plan` → Decode Context `assembly_plan`.
3. Decode Context `video_latents` → existing **Video VAE Decode**.
4. Decoded images → existing **Continuum Assemble**.

Keep the original assembly plan connected to Assemble. Keep the audio path
unchanged. The context node's output is **decode-only**: do not feed it back into
sampling, latent refinement/upscaling, Run Storage, or continuation state.

This works independently of the progressive node: conservative Target Input,
target-sparse, and mixed-grid can all produce exact-prefix continuation chunks.
No H3 transformer evaluation is added. The selected VAE remains external.

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

The final chunk is unchanged. Accepted latents, masks, original dictionaries,
the plan, and audio remain unchanged. Existing physical `decode_groups` take
precedence over logical chunk entries, including terminal-merged plans.

Non-exact overlaps (for example, independently upscaled/refined duplicate
prefixes or Guide-mode regeneration) remain unchanged and are identified in
the report. No approximate equality, implicit resizing, color matching, or
replacement of generated suffix latents is used. Such boundaries require a
different correction; a `0/N` report means this fix did not activate.

## Evidence and limits

The regression oracle executes the pinned native VAE's actual temporal window,
padding, blending, and write methods with a context-sensitive stand-in for its
learned pixel decoder. Four-chunk tests with 5-, 22-, and 39-frame overlaps:

- reproduce the previous five-frame boundary discrepancy;
- produce bit-identical results to a single continuous latent decode with
  right context;
- preserve accepted latent values, plan contents, and output duration.

Source: ComfyUI `1af040bf022569d7a890241c8dd79b296cda483f`,
`comfy/ldm/minimax/vae.py`; Continuum
`bf25353d8bec44afea22c89717c4301ce13c2036`, `v3/assembly.py`, `hardening.py`,
and `temporal.py`. The native source oracle runs in CI.

The structural decode-context correction is proven by that oracle. Actual
checkpoint media and speed are not established by a synthetic pixel decoder.
There is one extra seven-token VAE window and 17 discarded decoded frames per
corrected boundary, plus temporary extended latent storage. The implementation
does not combine the entire sequence into a single large decode call. TRT or
other replacement VAEs must preserve native H3 temporal window semantics for
the equivalence argument to apply. Inspect decoded media with the actual VAE;
remaining generation-domain flashing and the learned-prefix splice remain
separate research questions. Both integration PRs remain draft.
