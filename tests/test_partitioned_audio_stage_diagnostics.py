from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import compare_audio_latent_stages, measure_audio_latent_boundary
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.partitioned_scheduler import (
    _audit_source_carrier_audio_prefix,
    _resolve_audio_diagnostic_masks,
)


def _audio_mask(channels: int = 4, temporal: int = 10, prefix: int = 6):
    mask = torch.ones(1, channels, 2, temporal)
    mask[..., :prefix] = 0
    return mask


def test_audio_stage_delta_reports_exact_prefix_and_generated_channel_changes():
    reference = torch.ones(1, 4, 2, 10)
    candidate = reference.clone()
    candidate[..., 6:] *= 2.0
    report = compare_audio_latent_stages(
        reference,
        candidate,
        _audio_mask(),
        windows=(4,),
    )
    assert report["audio_prefix_ticks"] == 6
    assert report["exact_prefix_max_abs_delta"] == 0.0
    assert report["exact_prefix_rms_delta"] == 0.0
    window = report["windows"]["4"]
    assert window["candidate_over_reference_rms_db"] == pytest.approx(6.0205999)
    assert window["delta_over_reference_rms"] == pytest.approx(1.0)
    assert window["cosine_similarity"] == pytest.approx(1.0)
    assert window["per_channel_candidate_over_reference_db"] == pytest.approx([6.0205999] * 4)


@pytest.mark.parametrize(
    ("prefix", "reason"),
    [
        (0, "no_exact_audio_prefix"),
        (10, "no_generated_audio_suffix"),
    ],
)
def test_audio_stage_delta_skips_valid_audio_without_continuation_boundary(prefix, reason):
    reference = torch.ones(1, 4, 2, 10)
    candidate = reference.clone()
    candidate[..., 2:] *= 2
    before = candidate.clone()
    report = compare_audio_latent_stages(reference, candidate, _audio_mask(prefix=prefix))
    assert report == {
        "available": False,
        "reason": reason,
        "audio_prefix_ticks": prefix,
        "audio_total_ticks": 10,
        "windows": {},
    }
    assert torch.equal(candidate, before)


def test_audio_stage_delta_skips_noncanonical_mask_without_invalidating_sample():
    reference = torch.zeros(1, 4, 2, 10)
    candidate = reference.clone()
    mask = _audio_mask()
    mask[..., 8] = 0.5
    report = compare_audio_latent_stages(reference, candidate, mask)
    assert report["available"] is False
    assert report["reason"] == "unsupported_exact_audio_mask"
    assert report["audio_prefix_ticks"] == 6
    assert report["windows"] == {}


def test_sampler_overlap_mask_does_not_replace_exact_boundary_diagnostic_mask():
    video = torch.zeros(1, 24, 4, 2, 2)
    audio = torch.zeros(1, 32, 2, 10)
    packed, shapes = pack_streams((video, audio))

    exact_video_mask = torch.ones_like(video)
    exact_video_mask[:, :, :2] = 0
    exact_audio_mask = _audio_mask(channels=32, temporal=10, prefix=6)
    exact_mask = pack_streams((exact_video_mask, exact_audio_mask))[0]

    runtime_audio_mask = exact_audio_mask.clone()
    runtime_audio_mask[..., 2:6] = torch.tensor([0.2, 0.4, 0.6, 0.8]).view(1, 1, 1, 4)
    runtime_mask = pack_streams((exact_video_mask, runtime_audio_mask))[0]

    diagnostic_target, diagnostic_low = _resolve_audio_diagnostic_masks(
        runtime_mask,
        exact_mask,
        list(shapes),
        list(shapes),
    )

    assert torch.equal(diagnostic_target, exact_mask)
    assert torch.equal(diagnostic_low, exact_mask)
    _runtime_video, runtime_audio = unpack_streams(runtime_mask, list(shapes))
    assert bool(((runtime_audio > 0) & (runtime_audio < 1)).any().item())

    report = measure_audio_latent_boundary(
        packed,
        list(shapes),
        diagnostic_target,
        windows=(4,),
    )
    assert report["audio_prefix_ticks"] == 6


def test_audio_diagnostic_mask_rejects_geometry_drift():
    runtime = torch.zeros(1, 10)
    exact = torch.zeros(1, 11)
    with pytest.raises(RuntimeError, match="diagnostic mask geometry drifted"):
        _resolve_audio_diagnostic_masks(runtime, exact, [(1, 10)], [(1, 10)])


def test_audio_boundary_measurement_skips_unsupported_mask():
    video = torch.zeros((1, 24, 4, 2, 2))
    audio = torch.randn((1, 32, 2, 10))
    packed, shapes = pack_streams((video, audio))
    audio_mask = _audio_mask(channels=32)
    audio_mask[..., 8] = 0.5
    mask = pack_streams((torch.ones_like(video), audio_mask))[0]
    previous = packed.clone()
    report = measure_audio_latent_boundary(packed, list(shapes), mask)
    assert report["available"] is False
    assert report["reason"] == "unsupported_exact_audio_mask"
    assert report["audio_prefix_ticks"] == 6
    assert torch.equal(packed, previous)


def test_audio_position_all_generated_does_not_abort_completed_sample():
    original = torch.zeros((1, 32, 2, 10))
    generated = torch.randn_like(original)
    previous = generated.clone()
    report = _audit_source_carrier_audio_prefix(generated, original, torch.ones_like(original))
    assert report["protected_audio_prefix_available"] is False
    assert report["final_exact_audio_prefix"] is None
    assert report["audio_integrity_reason"] == "no_protected_carried_audio"
    assert torch.equal(generated, previous)


def test_audio_position_checks_protected_bytes_when_present():
    original = torch.randn((1, 32, 2, 10))
    final = original.clone()
    final[..., 6:] += 2.5
    report = _audit_source_carrier_audio_prefix(final, original, _audio_mask(channels=32))
    assert report["final_exact_audio_prefix"] is True
    final[..., 2] += 0.01
    with pytest.raises(RuntimeError, match="violated exact carried-audio restoration"):
        _audit_source_carrier_audio_prefix(final, original, _audio_mask(channels=32))


def test_audio_position_rejects_real_geometry_mismatch():
    original = torch.zeros((1, 32, 2, 10))
    with pytest.raises(RuntimeError, match="audio/mask geometry"):
        _audit_source_carrier_audio_prefix(original, original, torch.ones((1, 32, 2, 11)))
