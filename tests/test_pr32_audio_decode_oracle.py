from __future__ import annotations

import re

import torch

from h3_flow_regenerate.pr32_audio_decode_context import prepare_audio_decode_context
from h3_flow_regenerate.pr32_audio_decode_oracle import (
    H3ContinuumAudioDecodeOracleDiagnostic,
    decode_audio_boundary_oracle,
)


class _FakeAudioVAE:
    audio_sample_rate_output = 32000

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        # Native shape [B,32,2,T] -> decoder-style [B,S,2].
        stereo = latent.mean(dim=1)
        waveform = stereo.repeat_interleave(800, dim=-1)
        return waveform.movedim(1, -1)


def _sequence():
    generator = torch.Generator().manual_seed(408)
    first_t = 220
    prefix = 65
    new_t = 200
    base = torch.randn(1, 1, 2, first_t + new_t, generator=generator)
    base[..., :first_t] *= 0.5
    base[..., first_t:] *= 3.0
    timeline = base.repeat(1, 32, 1, 1)
    left = timeline[..., :first_t].clone()
    right = timeline[..., first_t - prefix :].clone()
    latents = [{"samples": left}, {"samples": right}]
    plan = {
        "magic": "H3_CONTINUUM_ASSEMBLY_PLAN",
        "schema_version": 1,
        "fps": 24,
        "decode_groups": [
            {
                "total_frames": 132,
                "trim_frames": 0,
                "net_frames": 132,
                "expected_audio_latent_t": first_t,
            },
            {
                "total_frames": 159,
                "trim_frames": 39,
                "net_frames": 120,
                "expected_audio_latent_t": prefix + new_t,
            },
        ],
    }
    return latents, plan


def _metric(report: str, name: str) -> float:
    match = re.search(rf"{re.escape(name)}=([+-]?[0-9.]+)", report)
    assert match is not None, report
    return float(match.group(1))


def test_oracle_is_self_contained_and_exposes_independent_normalization_mismatch():
    latents, plan = _sequence()
    before = [item["samples"].clone() for item in latents]

    core_audio, shared_audio, report = decode_audio_boundary_oracle(latents, _FakeAudioVAE(), plan)

    assert "1/1 exact boundaries" in report
    assert "after exact 65-latent overlap" in report
    assert _metric(report, "raw_future_corr") > 0.999
    assert "raw_future_gain=1.000000" in report
    assert _metric(report, "shared_future_corr") > 0.999
    assert "shared_future_gain=1.000000" in report
    assert _metric(report, "core_future_corr") > 0.999
    assert abs(_metric(report, "core_future_gain") - 1.0) > 0.05
    divisors = [float(value) for value in re.findall(r"core_divisor=([0-9.]+)", report)]
    assert len(divisors) == 2
    assert abs(divisors[1] - _metric(report, "shared_stream_divisor")) > 0.05

    assert len(core_audio) == len(shared_audio) == 2
    assert all(item["sample_rate"] == 32000 for item in core_audio + shared_audio)
    assert all(torch.equal(item["samples"], old) for item, old in zip(latents, before, strict=True))


def test_core_output_matches_core_vaedecodeaudio_normalization_exactly():
    latents, plan = _sequence()
    vae = _FakeAudioVAE()
    extended, _ = prepare_audio_decode_context(latents, plan)
    core_audio, _, _ = decode_audio_boundary_oracle(latents, vae, plan)

    for index, latent in enumerate(extended):
        raw = vae.decode(latent["samples"]).movedim(-1, 1)
        divisor = torch.std(raw, dim=[1, 2], keepdim=True) * 5.0
        divisor[divisor < 1.0] = 1.0
        torch.testing.assert_close(
            core_audio[index]["waveform"],
            raw / divisor,
            rtol=0.0,
            atol=0.0,
        )


def test_shared_gain_output_preserves_same_timeline_future_amplitude():
    latents, plan = _sequence()
    _, shared_audio, _ = decode_audio_boundary_oracle(latents, _FakeAudioVAE(), plan)

    left = shared_audio[0]["waveform"]
    right = shared_audio[1]["waveform"]
    left_cut = round(132 / 24 * 32000)
    right_cut = round(39 / 24 * 32000)
    count = 8000
    torch.testing.assert_close(
        left[..., left_cut : left_cut + count],
        right[..., right_cut : right_cut + count],
        rtol=0.0,
        atol=0.0,
    )


def test_node_contract_returns_both_causal_outputs_and_report():
    latents, plan = _sequence()
    core_audio, shared_audio, report = H3ContinuumAudioDecodeOracleDiagnostic().decode(
        latents,
        [_FakeAudioVAE()],
        [plan],
    )
    assert len(core_audio) == len(shared_audio) == 2
    assert "self-contained decoded-audio oracle" in report
