# Native target-high temporal conditioning — experimental candidate

This option is a **model-native masked diffusion transition**, not a final-frame
warp/duplicate, VAE replay, or a repair qualified by GPU media. It is **off by default**.

On the recommended **H3 Native Source Continuation Handoff**, select
`high_native_temporal_continuity = native_two_token_v1` to test the first two
generated temporal tokens with 0.5 and 0.75 denoising masks. These weights are
exact multiples of 1/256, matching Core H3 mask quantization. The first token
uses the pre-high low-motion prediction with less sigma than the rest of the
high refinement; subsequent video tokens retain the ordinary full sigma.

To make Core's inpainting model input and timestep labels coherent, the high
sampler's private `latent_image` is populated from the exact-restored
pre-high clean trajectory **only at those two tokens**. The sampler noise
argument is recomputed so the initial conditional high state remains unchanged.
The caller-owned clean latent, original masks, exact prefix and audio are not
modified. The later generated suffix and total sigma schedule are unaffected.

**Limitation:** At the final step Core's native inpaint returns a partial mix of
the high prediction and the pre-high clean for these two tokens. This can
preserve the earlier camera trajectory but could retain softer details or
create ghosting. A rendered fix has NOT been demonstrated. The old
prefix-release overlap experiment was hardware-falsified and is not revived.

No new VAE decode node, tensor-bundle audit or additional H3 model invocation
is introduced. Ordinary video generation, unchanged seed/refs/prompts, is the
only necessary eventual quality check. Keep this candidate disabled by default
until actual media validates a disappearing join-frame hold without moving it
to the next pair or harming sharpness.

The invariant tests cover mask quantization, exact protected prefix/audio,
unchanged suffix outside the two transition tokens, clean-source ownership,
and reconstructing the same initial high noisy state under the corrected
high-private clean operand.
