# PR #32 ComfyUI-Patcher activation contract

PR #32 is a temporary diagnostic overlay and is intended to be materialized through ComfyUI-Patcher. Applying the PR is the activation boundary for the bounded audio guided-overlap experiment; no extra workflow node and no shell-level feature enablement are required.

The custom-node package entry point defaults `H3_FLOW_PR32_AUDIO_GUIDED_OVERLAP_TICKS` to `4` only when the variable is unset. An explicit environment value still wins, so `H3_FLOW_PR32_AUDIO_GUIDED_OVERLAP_TICKS=0` remains an emergency disable/bisect hook.

ComfyUI-Patcher materializes tracked overlays as synthetic merge commits. A local repository HEAD created by Patcher is therefore not expected to equal the pull-request head SHA. Validation must use the materialized overlay/runtime telemetry rather than raw local SHA equality.

When the continuation reaches the conservative exact-prefix fallback, the diagnostic must emit `pr32_audio_guided_overlap` metrics and the `PR #32 audio guided overlap active ticks=4 ... final_exact_restore=true` runtime warning. The video mask remains unchanged and the caller-visible originally protected AV prefix is restored exactly after sampling.

This remains media-unvalidated until a matched CUDA run demonstrates improved generated-audio quality while the click remains absent and the protected-prefix contract remains exact.
