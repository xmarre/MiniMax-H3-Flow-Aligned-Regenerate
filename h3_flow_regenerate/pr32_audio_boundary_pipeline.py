"""PR #32 one-node H3 Continuum audio boundary diagnostic pipeline.

This node intentionally collapses the decoded-audio oracle and latent-phase
alignment intervention into one executable graph node.  The user-facing path is
therefore strictly serial:

    Continuum audio_latents + Audio VAE + assembly_plan
        -> Audio Boundary Pipeline
        -> phase_aligned_audio
        -> Continuum Assemble.audio

The original LATENT groups and assembly plan are still used internally to prove
exact protected carry and derive the global 40-Hz audio-latent origin.  They are
not parallel audio branches and are never emitted as competing AUDIO outputs.

No additional H3 evaluation or VAE decode is introduced: the oracle decodes each
physical group once, then phase alignment operates only on those decoded AUDIO
containers.  The existing standalone diagnostic nodes remain available for
isolated investigation, but they are not required for the matched PR #32 test.
"""

from __future__ import annotations

import logging

from .pr32_audio_decode_oracle import decode_audio_boundary_oracle
from .pr32_audio_phase_align import phase_align_decoded_audio

_LOGGER = logging.getLogger(__name__)


def decode_and_phase_align_audio_boundary(
    audio_latents: list[dict],
    vae,
    assembly_plan: dict,
) -> tuple[list[dict], str]:
    """Decode once, apply the shared-gain oracle, then phase-align that AUDIO.

    The ordering exactly matches the intended former two-node chain:

        decode oracle.shared_gain_audio -> phase alignment -> Assemble

    Returning only the final AUDIO prevents accidental direct Oracle -> Assemble
    wiring from silently pruning the phase intervention.
    """

    _core_audio, shared_audio, oracle_report = decode_audio_boundary_oracle(
        audio_latents,
        vae,
        assembly_plan,
    )
    phase_aligned_audio, phase_report = phase_align_decoded_audio(
        shared_audio,
        audio_latents,
        assembly_plan,
    )
    report = "\n".join(
        (
            "PR #32 integrated decoded-audio boundary pipeline",
            oracle_report,
            phase_report,
            "FINAL AUDIO OUTPUT = shared-gain decoded waveform after global latent-phase alignment. "
            "Connect phase_aligned_audio directly to Continuum Assemble.audio; keep Audio Seam = Off.",
        )
    )
    return phase_aligned_audio, report


class H3ContinuumAudioBoundaryPipelineDiagnostic:
    CATEGORY = "MiniMax H3/flow regenerate"
    DESCRIPTION = (
        "PR #32 integrated audio boundary diagnostic. Connect the ORIGINAL Continuum "
        "audio_latents, the Audio VAE, and the unchanged assembly_plan here, then connect "
        "only phase_aligned_audio to Continuum Assemble.audio. Internally this performs "
        "the self-contained decode oracle followed by global 40-Hz latent-phase alignment. "
        "It adds no H3 call, no extra VAE decode, no resampling, and no crossfade."
    )
    INPUT_IS_LIST = True
    RETURN_TYPES = ("AUDIO", "STRING")
    RETURN_NAMES = ("phase_aligned_audio", "report")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "decode_and_align"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_latents": ("LATENT",),
                "vae": ("VAE",),
                "assembly_plan": ("H3_CONTINUUM_ASSEMBLY_PLAN",),
            }
        }

    def decode_and_align(self, audio_latents, vae, assembly_plan):
        if len(vae) != 1:
            raise ValueError("integrated audio boundary diagnostic requires exactly one VAE")
        if len(assembly_plan) != 1:
            raise ValueError("integrated audio boundary diagnostic requires exactly one assembly plan")
        result, report = decode_and_phase_align_audio_boundary(
            audio_latents,
            vae[0],
            assembly_plan[0],
        )
        _LOGGER.warning(report)
        return result, report


NODE_CLASS_MAPPINGS = {
    "H3ContinuumAudioBoundaryPipelineDiagnostic": H3ContinuumAudioBoundaryPipelineDiagnostic,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ContinuumAudioBoundaryPipelineDiagnostic": ("MiniMax H3 Continuum Audio Boundary Pipeline (PR32 Diagnostic)"),
}
