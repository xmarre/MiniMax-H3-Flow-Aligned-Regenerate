# Sol attention determinism investigation — retired live protocol

The live low-sampler replay and internal kernel observers were removed in the
v0.3.12 cleanup. Their serialized controls remain inert compatibility fields;
they cannot add H3 calls or install observers. Saved frozen A/B bundles remain
readable through the legacy audit node.

The original protocol and implementation are preserved in the
[pre-cleanup checkpoint](https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/cd4802200118eb29ff833d07d1fe47c9369d4d6b/docs/sol-attn-determinism.md).
The tested numerical overflow correction now belongs to
[comfy-kitchen #242](https://github.com/Comfy-Org/comfy-kitchen/pull/242), with
packaging/interoperability handled by
[#246](https://github.com/Comfy-Org/comfy-kitchen/pull/246).

The qualified frozen A/B/A comparison favored native source carry over the
half-pixel latent candidate, whose source decode lost substantial spatial detail.
It did not qualify a final target-high B rendering or a darkening correction.
Use [Native Source Continuation](NATIVE_SOURCE_PREFIX_CARRY.md) for current setup
and [full-stage replay](FULL_STAGE_CONTINUITY.md) for saved boundary analysis.
