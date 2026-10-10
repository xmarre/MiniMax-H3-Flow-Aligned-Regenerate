"""Controlled low-stage provenance is necessary but insufficient for causal A/B."""

import copy

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.low_input_pairing import (
    POLICY,
    compare_low_input_pairing_receipts,
    make_low_input_pairing_receipt,
)


def _fixture():
    generator = torch.Generator().manual_seed(20261010)
    video = torch.randn((1, 24, 17, 4, 4), generator=generator)
    audio = torch.randn((1, 32, 2, 4, 4), generator=generator)
    noise_video = torch.randn(video.shape, generator=generator)
    noise_audio = torch.randn(audio.shape, generator=generator)
    mask_video = torch.ones_like(video)
    mask_video[:, :, :7] = 0
    mask_audio = torch.ones_like(audio)
    latent, shapes = pack_streams((video, audio))
    noise, _ = pack_streams((noise_video, noise_audio))
    mask, _ = pack_streams((mask_video, mask_audio))
    target = torch.randn((1, 24, 7, 8, 8), generator=generator)
    return {
        "authoritative_target_prefix": target,
        "source_prefix": video[:, :, :7].clone(),
        "low_noise": noise,
        "low_latent_image": latent,
        "low_mask": mask,
        "low_shapes": shapes,
        "low_sigmas": torch.tensor([1.0, 0.8, 0.6]),
        "prefix_t": 7,
        "source_policy": "native_source_carry_v1",
        "sampler": "sample_euler",
        "seed": 17654,
        "conditioning_signature": "same-conditions",
    }


def test_low_pair_receipt_checks_actual_input_and_preserves_source_tensors():
    params = _fixture()
    copies = {name: value.clone() for name, value in params.items() if isinstance(value, torch.Tensor)}
    report = make_low_input_pairing_receipt(**params)
    assert report["policy"] == POLICY
    assert report["source_temporal"] == 17
    assert report["prefix_t"] == 7
    assert report["checkpoint_identity_verified"] is False
    assert report["rng_state_verified"] is False
    assert report["causal_pair_qualified"] is False
    assert all(torch.equal(params[key], value) for key, value in copies.items())


def test_only_source_prefix_variation_is_pair_eligible_not_causally_qualified():
    left_args = _fixture()
    right_args = copy.deepcopy(left_args)
    different = right_args["source_prefix"] + 0.125
    right_args["source_prefix"] = different.contiguous()
    original_video, original_audio = unpack_streams(left_args["low_latent_image"], left_args["low_shapes"])
    candidate_video = original_video.clone()
    candidate_video[:, :, :7] = different
    right_args["low_latent_image"], _ = pack_streams((candidate_video, original_audio.clone()))
    right_args["source_policy"] = "half_pixel_latent_v1"
    a = make_low_input_pairing_receipt(**left_args)
    b = make_low_input_pairing_receipt(**right_args)
    comparison = compare_low_input_pairing_receipts(a, b)
    assert comparison["matched_known_low_inputs"] is True
    assert comparison["source_prefix_intervention_observed"] is True
    assert comparison["input_pair_eligible"] is True
    assert comparison["causal_pair_qualified"] is False


@pytest.mark.parametrize("field", ["seed", "conditioning_signature", "audio_mask_sha256", "low_sigmas_sha256"])
def test_changed_control_rejects_pair(field):
    params = _fixture()
    original = make_low_input_pairing_receipt(**params)
    modified = dict(original)
    modified[field] = "different"
    modified["source_prefix_sha256"] = "alternate-prefix"
    modified["source_policy"] = "half_pixel_latent_v1"
    result = compare_low_input_pairing_receipts(original, modified)
    assert field in result["mismatched_fields"]
    assert result["input_pair_eligible"] is False


def test_same_source_prefix_is_not_an_intervention():
    report = make_low_input_pairing_receipt(**_fixture())
    assert not compare_low_input_pairing_receipts(report, report)["input_pair_eligible"]


def test_wrong_source_prefix_fails_before_reporting():
    params = _fixture()
    params["source_prefix"] = params["source_prefix"] + 0.5
    with pytest.raises(ValueError, match="differs from actual"):
        make_low_input_pairing_receipt(**params)


def test_missing_provenance_fails_closed():
    report = make_low_input_pairing_receipt(**_fixture())
    altered = dict(report)
    altered.pop("source_suffix_initial_noise_sha256")
    with pytest.raises(ValueError, match="missing required"):
        compare_low_input_pairing_receipts(report, altered)
