from __future__ import annotations

import re

import torch

from h3_flow_regenerate.pr32_audio_phase_align import (
    H3ContinuumAudioPhaseAlignDiagnostic,
    phase_align_decoded_audio,
)


def _case_00410_geometry():
    generator = torch.Generator().manual_seed(410)
    left_t = 292
    prefix_t = 65
    right_t = 348
    future_t = right_t - prefix_t
    left_latent = torch.randn(1, 32, 2, left_t, generator=generator)
    future = torch.randn(1, 32, 2, future_t, generator=generator)
    right_latent = torch.cat((left_latent[..., -prefix_t:].clone(), future), dim=-1)

    sample_rate = 32000
    samples_per_latent = 800
    # Group 2 starts at global latent 227. Construct two decoded views from one
    # continuous synthetic waveform so the correct 267-sample phase adjustment
    # can be proven without a VAE.
    right_origin = (left_t - prefix_t) * samples_per_latent
    global_samples = right_origin + right_t * samples_per_latent
    t = torch.arange(global_samples, dtype=torch.float32)
    global_wave = (
        0.35 * torch.sin(t * 0.0137)
        + 0.17 * torch.sin(t * 0.0311 + 0.4)
        + 0.05 * torch.cos(t * 0.0713)
    ).reshape(1, 1, -1).repeat(1, 2, 1)
    left_wave = global_wave[..., : left_t * samples_per_latent].clone()
    right_wave = global_wave[..., right_origin : right_origin + right_t * samples_per_latent].clone()

    audio = [
        {"waveform": left_wave, "sample_rate": sample_rate},
        {"waveform": right_wave, "sample_rate": sample_rate},
    ]
    latents = [{"samples": left_latent}, {"samples": right_latent}]
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
    return audio, latents, plan


def _metric(report: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}=([+-]?[0-9.]+)", report)
    assert match is not None, report
    return float(match.group(1))


def test_00410_geometry_derives_exact_267_sample_phase_compensation():
    audio, latents, plan = _case_00410_geometry()

    aligned, report = phase_align_decoded_audio(audio, latents, plan)

    assert "group 2: origin_latent=227" in report
    assert "native_trim=52000s phase_trim=51733" in report
    assert "phase_delta=+267s" in report
    assert _metric(report, "phase_future_corr") > 0.9999
    assert _metric(report, "phase_actual_jump") < _metric(report, "native_actual_jump")
    assert aligned[1]["waveform"].shape[-1] == audio[1]["waveform"].shape[-1] + 267

    # Continuum's unchanged 52,000-sample trim must now land on the original
    # right waveform's 51,733rd sample and consume the complete remaining decode
    # without the historical 267-sample tail shortfall.
    native_trim = 52000
    wanted = round((175 + 170) / 24 * 32000) - round(175 / 24 * 32000)
    torch.testing.assert_close(
        aligned[1]["waveform"][..., native_trim : native_trim + wanted],
        audio[1]["waveform"][..., 51733 : 51733 + wanted],
        rtol=0.0,
        atol=0.0,
    )
    assert native_trim + wanted == aligned[1]["waveform"].shape[-1]


def test_nonexact_audio_carry_fails_closed_to_native_trim():
    audio, latents, plan = _case_00410_geometry()
    latents[1] = {"samples": latents[1]["samples"].clone()}
    latents[1]["samples"][..., 0] += 1.0

    aligned, report = phase_align_decoded_audio(audio, latents, plan)

    assert "phase alignment disabled" in report
    assert "group 2: origin_latent=None" in report
    assert "phase_delta=+0s" in report
    torch.testing.assert_close(aligned[1]["waveform"], audio[1]["waveform"], rtol=0.0, atol=0.0)


def test_node_contract_is_list_mapped_and_does_not_mutate_inputs():
    audio, latents, plan = _case_00410_geometry()
    before_audio = [item["waveform"].clone() for item in audio]
    before_latents = [item["samples"].clone() for item in latents]

    aligned, report = H3ContinuumAudioPhaseAlignDiagnostic().align(audio, latents, [plan])

    assert len(aligned) == 2
    assert "latent-phase alignment" in report
    assert all(torch.equal(item["waveform"], old) for item, old in zip(audio, before_audio, strict=True))
    assert all(torch.equal(item["samples"], old) for item, old in zip(latents, before_latents, strict=True))
