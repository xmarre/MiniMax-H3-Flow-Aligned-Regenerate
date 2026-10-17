## Coordinated companion versions — 17 October 2026

| Component | Release | Included PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.14](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.14) | [#102](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/102), [#103](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/103), [#105](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/105), [#107](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/107), [#108](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/108) |
| Sol-H3 | [v0.1.11](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.11) | [#41](https://github.com/xmarre/ComfyUI-Sol-H3/pull/41) |
| VDN-H3-Plus | [v1.5.10](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.10) | [#41–#44](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/44) |
| H3 Continuum-Plus | [v3.4.9](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.9) | [#44](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/44), [#46](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/46), [#47](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/pull/47) |

Update companions together for the full Sol/VDN/Flow/Continuum stack. Sol is optional and only selected when configured as the attention backend. The completed CPU checks and owner observations do not establish identical work between Sol full VDN (550/550) and Core MSA/BSA reduced VDN (166/550), or full-tensor SM120 bitwise parity for the new VDN temporal kernel. Keyless PRs and other upstream patches are **not** part of this release.

## Coordinated companion versions

Flow v0.3.13 includes PR #101. The companion packages below are separate releases;
install compatible versions together for the progressive H3 workflow.

| Component | Release | Implementation PRs |
| --- | --- | --- |
| Flow-Aligned Regenerate | [v0.3.13](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/releases/tag/v0.3.13) | [#101](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/pull/101) |
| Sol-H3 | [v0.1.10](https://github.com/xmarre/ComfyUI-Sol-H3/releases/tag/v0.1.10) | [#40](https://github.com/xmarre/ComfyUI-Sol-H3/pull/40) |
| VDN-H3-Plus | [v1.5.9](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/releases/tag/v1.5.9) | [#39](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/39), [#40](https://github.com/xmarre/ComfyUI-VDN-H3-Plus/pull/40) |
| H3 Continuum-Plus | [v3.4.6](https://github.com/xmarre/ComfyUI-H3-Continuum-Plus/releases/tag/v3.4.6) | unchanged |
| Latent Upscaler-Plus | [v0.2.2](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus/releases/tag/v0.2.2) | unchanged |
| Spectrum MiniMax H3 | [v0.2.29](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/releases/tag/v0.2.29) | [#121](https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3/pull/121) |

Native Source Continuation requires VDN-H3-Plus and the learned 3D upscaler for
single chunks and sequences. Continuum supplies the protected prefix for a
sequence. Sol-H3 and Spectrum are optional.

## Upstream patches

For INT8 fused MLP runtime adapters, retain the
[ComfyUI #16783](https://github.com/Comfy-Org/ComfyUI/pull/16783) Patcher overlay
until that repair is available in the installed Core release.

The comfy-kitchen overflow correction
[#242](https://github.com/Comfy-Org/comfy-kitchen/pull/242) and
packaging/interoperability work
[#246](https://github.com/Comfy-Org/comfy-kitchen/pull/246) are separate upstream
changes. They are not bundled in Flow v0.3.13 or declared included in the
companion versions above. Installations using a provisional Kitchen build must
retain that build until the required upstream correction is released.
Separate Keyless and sampler experiments are outside this release.
