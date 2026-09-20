from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import compare_audio_latent_stages


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
    assert window["per_channel_candidate_over_reference_db"] == pytest.approx(
        [6.0205999] * 4
    )


def test_audio_stage_delta_rejects_noncanonical_mask():
    reference = torch.zeros(1, 4, 2, 10)
    candidate = reference.clone()
    mask = _audio_mask()
    mask[..., 8] = 0.5
    with pytest.raises(ValueError, match="contiguous exact prefix"):
        compare_audio_latent_stages(reference, candidate, mask)
