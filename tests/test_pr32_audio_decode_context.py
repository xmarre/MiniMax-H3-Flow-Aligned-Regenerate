from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate.pr32_audio_decode_context import (
    H3ContinuumAudioDecodeContextDiagnostic,
    prepare_audio_decode_context,
)


def _audio_sequence(*, count: int = 3, prefix: int = 65, first_t: int = 220, new_t: int = 200):
    generator = torch.Generator().manual_seed(404)
    total_t = first_t + new_t * (count - 1)
    timeline = torch.randn(1, 32, 2, total_t, generator=generator)
    audios = [timeline[..., :first_t].clone()]
    groups = [
        {
            "total_frames": 120,
            "trim_frames": 0,
            "net_frames": 120,
            "expected_audio_latent_t": first_t,
        }
    ]
    stop = first_t
    for _ in range(1, count):
        start = stop - prefix
        stop += new_t
        audio = timeline[..., start:stop].clone()
        audios.append(audio)
        groups.append(
            {
                "total_frames": 39 + 120,
                "trim_frames": 39,
                "net_frames": 120,
                "expected_audio_latent_t": prefix + new_t,
            }
        )
    return (
        timeline,
        [{"samples": audio} for audio in audios],
        {
            "magic": "H3_CONTINUUM_ASSEMBLY_PLAN",
            "schema_version": 1,
            "fps": 24,
            "chunks": groups,
        },
    )


def test_exact_audio_overlap_gets_full_adjacent_generated_suffix_without_mutation():
    _, latents, plan = _audio_sequence()
    before = [item["samples"].clone() for item in latents]
    old_plan = copy.deepcopy(plan)

    result, report = H3ContinuumAudioDecodeContextDiagnostic().prepare(latents, [plan])

    assert "2/2 exact boundaries" in report
    assert "Audio Seam = Off" in report
    for index in range(2):
        expected_future = before[index + 1][..., 65:]
        assert torch.equal(result[index]["samples"][..., : before[index].shape[-1]], before[index])
        assert torch.equal(result[index]["samples"][..., before[index].shape[-1] :], expected_future)
        assert result[index]["samples"].data_ptr() != latents[index]["samples"].data_ptr()
    assert result[-1] is latents[-1]
    assert plan == old_plan
    assert all(torch.equal(item["samples"], original) for item, original in zip(latents, before, strict=True))


def test_nonexact_overlap_is_not_extended():
    _, latents, plan = _audio_sequence(count=2)
    latents[1]["samples"][..., 0] += 0.125
    result, report = prepare_audio_decode_context(latents, plan)
    assert result[0] is latents[0]
    assert "protected audio overlap is not exact" in report
    assert "0/1 exact boundaries" in report


def test_nonintegral_guide_frame_boundary_is_not_rounded():
    _, latents, plan = _audio_sequence(count=2)
    plan["chunks"][1]["trim_frames"] = 22
    result, report = prepare_audio_decode_context(latents, plan)
    assert result[0] is latents[0]
    assert "no exact 24-fps/40-Hz audio boundary" in report


@pytest.mark.parametrize("corruption", ["count", "length", "schema", "fps"])
def test_stale_audio_plan_fails_closed(corruption):
    _, latents, plan = _audio_sequence()
    if corruption == "count":
        latents.pop()
    elif corruption == "length":
        plan["chunks"][1]["expected_audio_latent_t"] += 1
    elif corruption == "schema":
        plan["schema_version"] += 1
    else:
        plan["fps"] = 25
    with pytest.raises(ValueError):
        prepare_audio_decode_context(latents, plan)


def _noncausal_audio_oracle(z: torch.Tensor, *, radius: int = 5) -> torch.Tensor:
    """Small symmetric decoder oracle exposing endpoint-padding dependence."""

    mono = z.mean(dim=(1, 2), keepdim=False).unsqueeze(1)
    padded = F.pad(mono, (radius, radius), mode="replicate")
    kernel = torch.ones(1, 1, 2 * radius + 1, dtype=mono.dtype) / float(2 * radius + 1)
    return F.conv1d(padded, kernel)


def test_future_context_removes_independent_noncausal_decoder_endpoint_error():
    timeline, latents, plan = _audio_sequence(count=2)
    extended, _ = prepare_audio_decode_context(latents, plan)
    prefix = 65
    first_t = int(latents[0]["samples"].shape[-1])

    expected = _noncausal_audio_oracle(timeline)
    old_left = _noncausal_audio_oracle(latents[0]["samples"])
    corrected_left = _noncausal_audio_oracle(extended[0]["samples"])[..., :first_t]
    right = _noncausal_audio_oracle(latents[1]["samples"])[..., prefix:]

    old = torch.cat((old_left, right), dim=-1)
    corrected = torch.cat((corrected_left, right), dim=-1)
    expected_segment = expected[..., : corrected.shape[-1]]

    # The same continuous latent context can take slightly different CPU conv
    # reduction paths when decoded as one tensor versus a concatenated view, so
    # compare at numerical tolerance rather than requiring bit identity.
    torch.testing.assert_close(corrected, expected_segment, rtol=1e-6, atol=1e-7)
    assert torch.max(torch.abs(old - expected_segment)).item() > 1e-5
    # The old mismatch is localized to the left decoder's terminal receptive-field band.
    torch.testing.assert_close(old[..., : first_t - 5], expected[..., : first_t - 5], rtol=1e-6, atol=1e-7)
    assert torch.max(torch.abs(old[..., first_t - 5 : first_t] - expected[..., first_t - 5 : first_t])).item() > 1e-5


def test_decode_groups_take_precedence_over_logical_chunks():
    _, latents, plan = _audio_sequence(count=2)
    plan["decode_groups"] = plan["chunks"]
    plan["chunks"] = [{}] * 3
    result, report = prepare_audio_decode_context(latents, plan)
    assert len(result) == 2
    assert "1/1 exact boundaries" in report
