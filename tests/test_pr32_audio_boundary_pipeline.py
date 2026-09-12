from __future__ import annotations

import torch

from h3_flow_regenerate.pr32_audio_boundary_pipeline import (
    H3ContinuumAudioBoundaryPipelineDiagnostic,
    decode_and_phase_align_audio_boundary,
)


class _FakeAudioVAE:
    audio_sample_rate_output = 32000

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        stereo = latent.mean(dim=1)
        waveform = stereo.repeat_interleave(800, dim=-1)
        return waveform.movedim(1, -1)


def _case_00410_geometry():
    generator = torch.Generator().manual_seed(411)
    left_t = 292
    prefix_t = 65
    right_t = 348
    global_latents = torch.randn(
        1,
        32,
        2,
        left_t + right_t - prefix_t,
        generator=generator,
    )
    left = global_latents[..., :left_t].clone()
    right = global_latents[..., left_t - prefix_t :].clone()
    latents = [{"samples": left}, {"samples": right}]
    plan = {
        "magic": "H3_CONTINUUM_ASSEMBLY_PLAN",
        "schema_version": 1,
        "fps": 24,
        "decode_groups": [
            {
                "total_frames": 175,
                "trim_frames": 0,
                "net_frames": 175,
                "expected_audio_latent_t": left_t,
            },
            {
                "total_frames": 209,
                "trim_frames": 39,
                "net_frames": 170,
                "expected_audio_latent_t": right_t,
            },
        ],
    }
    return latents, plan


def test_integrated_pipeline_executes_oracle_then_phase_alignment_serially():
    latents, plan = _case_00410_geometry()
    before = [item["samples"].clone() for item in latents]

    aligned, report = decode_and_phase_align_audio_boundary(latents, _FakeAudioVAE(), plan)

    assert len(aligned) == 2
    assert "PR #32 self-contained decoded-audio oracle" in report
    assert "PR #32 H3 Continuum audio latent-phase alignment" in report
    assert "group 2: origin_latent=227" in report
    assert "native_trim=52000s phase_trim=51733" in report
    assert "phase_delta=+267s" in report
    assert "FINAL AUDIO OUTPUT" in report

    boundary_sample = round(175 / 24 * 32000)
    native_group2_trim = round(39 / 24 * 32000)
    wanted = round((175 + 170) / 24 * 32000) - boundary_sample
    assert native_group2_trim + wanted == aligned[1]["waveform"].shape[-1]
    torch.testing.assert_close(
        aligned[0]["waveform"][..., boundary_sample],
        aligned[1]["waveform"][..., native_group2_trim],
        rtol=0.0,
        atol=0.0,
    )
    assert all(torch.equal(item["samples"], old) for item, old in zip(latents, before, strict=True))


def test_integrated_node_exposes_only_final_audio_and_report():
    latents, plan = _case_00410_geometry()
    node = H3ContinuumAudioBoundaryPipelineDiagnostic()

    aligned, report = node.decode_and_align(latents, [_FakeAudioVAE()], [plan])

    assert len(aligned) == 2
    assert "phase_delta=+267s" in report
    assert node.RETURN_NAMES == ("phase_aligned_audio", "report")
    assert node.OUTPUT_IS_LIST == (True, False)
