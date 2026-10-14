"""Pairing, native phase, transaction publication and bounded carry ownership."""

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.source_prefix_carry import SourcePrefixCarry
from h3_flow_regenerate.source_prefix_projection import configure_source_prefix_projection


def pair():
    gen = torch.Generator().manual_seed(981)
    return torch.randn(1, 24, 17, 4, 6, generator=gen), torch.randn(1, 24, 17, 8, 10, generator=gen)


def prime(owner, source, target):
    with owner.transaction(object()):
        owner.begin_initial()
        owner.stage_source(source)
        return owner.prepare_success(target)


def test_carry_matches_actual_native_suffix_and_owns_cpu_bytes():
    source, target = pair()
    original = source.clone()
    owner = SourcePrefixCarry()
    receipt = prime(owner, source, target)
    source.zero_()
    with owner.transaction(object()):
        output, projection = owner.project(target[:, :, -7:], 4, 6)
        assert torch.equal(output.view(torch.int32), original[:, :, -7:].contiguous().view(torch.int32))
        output.zero_()
        again, _ = owner.project(target[:, :, -7:], 4, 6)
        assert torch.equal(again, original[:, :, -7:])
    assert receipt["retained_cpu_bytes"] == original.numel() * 4
    assert receipt["retained_cuda_bytes"] == 0
    assert projection["previous_source_token_span"] == [10, 17]
    assert projection["extra_vae_decode_calls"] == projection["extra_vae_encode_calls"] == 0


def test_missing_stale_grid_changed_prefix_and_phase_stop_before_use():
    source, target = pair()
    owner = SourcePrefixCarry()
    with owner.transaction(object()), pytest.raises(RuntimeError, match="preceding successful"):
        owner.project(target[:, :, -7:], 4, 6)
    prime(owner, source, target)
    with owner.transaction(object()):
        for prefix, hw in [(target[:, :, -7:] + 1e-4, (4, 6)), (target[:, :, -7:], (6, 6))]:
            with pytest.raises(RuntimeError, match="does not match"):
                owner.project(prefix, *hw)
        with pytest.raises(ValueError, match="5k\\+2"):
            owner.project(target[:, :, -8:], 4, 6)


def test_failure_after_preparing_does_not_publish_unreturned_target():
    source, target = pair()
    owner = SourcePrefixCarry()
    prime(owner, source, target)
    guider = object()
    with pytest.raises(RuntimeError, match="outer check failed"), owner.transaction(guider):
        with owner.transaction(guider):
            owner.stage_source(source + 3)
            owner.prepare_success(target + 2)
        raise RuntimeError("outer check failed")
    with owner.transaction(object()):
        output, _ = owner.project(target[:, :, -7:], 4, 6)
        assert torch.equal(output, source[:, :, -7:])
        with pytest.raises(RuntimeError, match="does not match"):
            owner.project(target[:, :, -7:] + 2, 4, 6)


def test_successful_next_chunk_replaces_previous_and_new_initial_clears_it():
    source, target = pair()
    owner = SourcePrefixCarry()
    prime(owner, source, target)
    with owner.transaction(object()):
        owner.stage_source(source + 3)
        receipt = owner.prepare_success(target + 2)
    assert receipt["generation"] == 2
    with owner.transaction(object()):
        output, _ = owner.project(target[:, :, -7:] + 2, 4, 6)
        assert torch.equal(output, source[:, :, -7:] + 3)
        owner.begin_initial()
        with pytest.raises(RuntimeError, match="preceding successful"):
            owner.project(target[:, :, -7:] + 2, 4, 6)


def test_other_guider_cannot_enter_and_staged_data_is_released_on_failure():
    source, _ = pair()
    owner = SourcePrefixCarry()
    with pytest.raises(RuntimeError, match="failed"), owner.transaction(object()):
        with pytest.raises(RuntimeError, match="already sampling"), owner.transaction(object()):
            pytest.fail("second guider entered")
        owner.stage_source(source)
        raise RuntimeError("failed")
    assert owner._source is owner._pending is None
    with pytest.raises(RuntimeError, match="active sampling transaction"):
        owner.stage_source(source)


def test_carry_configuration_needs_no_vae_and_default_clears_both_modes():
    from h3_flow_regenerate.source_prefix_carry import SOURCE_PREFIX_CARRY_KEY
    from h3_flow_regenerate.source_prefix_projection import SOURCE_PREFIX_PROJECTION_KEY

    model = SimpleNamespace(model_options={SOURCE_PREFIX_PROJECTION_KEY: object()})
    configure_source_prefix_projection(model, "native_source_carry", None, "progressive_uniform_source")
    assert isinstance(model.model_options[SOURCE_PREFIX_CARRY_KEY], SourcePrefixCarry)
    assert SOURCE_PREFIX_PROJECTION_KEY not in model.model_options
    configure_source_prefix_projection(model, "latent_bicubic", None, "progressive_uniform_source")
    assert SOURCE_PREFIX_CARRY_KEY not in model.model_options
    assert SOURCE_PREFIX_PROJECTION_KEY not in model.model_options


def test_sequence_lineage_rejects_cached_initial_or_foreign_session():
    source, target = pair()
    owner = SourcePrefixCarry()
    request = {"active": True, "session_id": "clip-a", "chunk_index": 1}
    guider = SimpleNamespace(model_options={"transformer_options": {"h3_continuum": request}})
    with owner.transaction(guider):
        owner.begin_initial()
        owner.stage_source(source)
        owner.prepare_success(target)
    with owner.transaction(guider), pytest.raises(RuntimeError, match="immediately preceding"):
        owner.project(target[:, :, -7:], 4, 6)
    request["chunk_index"] = 2
    with owner.transaction(guider):
        output, receipt = owner.project(target[:, :, -7:], 4, 6)
    assert torch.equal(output, source[:, :, -7:])
    assert receipt["previous_sequence"] == ("clip-a", 1)
    assert receipt["explicit_session_id_verified"] is True
    request["session_id"] = "clip-b"
    with owner.transaction(guider), pytest.raises(RuntimeError, match="immediately preceding"):
        owner.project(target[:, :, -7:], 4, 6)


def test_continuum_unlabeled_initial_binds_only_first_labeled_continuation():
    source, target = pair()
    owner = SourcePrefixCarry()
    prime(owner, source, target)
    request = {"active": True, "api": 1, "context_frames": 39, "chunk_index": 3}
    guider = SimpleNamespace(model_options={"transformer_options": {"h3_continuum": request}})
    with owner.transaction(guider), pytest.raises(RuntimeError, match="immediately preceding"):
        owner.project(target[:, :, -7:], 4, 6)
    request["chunk_index"] = 2
    with owner.transaction(guider):
        output, receipt = owner.project(target[:, :, -7:], 4, 6)
        assert torch.equal(output, source[:, :, -7:])
        assert receipt["initial_unlabeled_to_chunk_2"] is True
        assert receipt["current_sequence"] == (None, 2)
        assert receipt["explicit_session_id_verified"] is False
        owner.stage_source(source + 1)
        owner.prepare_success(target + 1)
    with owner.transaction(guider), pytest.raises(RuntimeError, match="immediately preceding"):
        owner.project(target[:, :, -7:] + 1, 4, 6)
    request["chunk_index"] = 3
    with owner.transaction(guider):
        _, receipt = owner.project(target[:, :, -7:] + 1, 4, 6)
        assert receipt["initial_unlabeled_to_chunk_2"] is False
        assert receipt["explicit_session_id_verified"] is False
    owner = SourcePrefixCarry()
    with owner.transaction(object()):
        owner.stage_source(source)
        owner.prepare_success(target)
    request["chunk_index"] = 2
    with owner.transaction(guider), pytest.raises(RuntimeError, match="immediately preceding"):
        owner.project(target[:, :, -7:], 4, 6)


def _blob(h, w, cy, cx, radius):
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    return torch.exp(-((yy - cy).square() + (xx - cx).square()) / (2 * radius**2))


def _checker(h, w):
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    return ((yy + xx) % 2).float() * 2 - 1


def test_target_band_takes_scene_layout_and_keeps_carry_detail_and_tone():
    from h3_flow_regenerate.source_prefix_carry import SOURCE_PREFIX_TARGET_BAND_POLICY, reconcile_target_band

    gen = torch.Generator().manual_seed(4)
    h, w = 24, 32
    scene = torch.randn(1, 24, 7, h, w, generator=gen) * 0.05
    plant = _blob(h, w, 8, 22, 2.5)
    detail = _checker(h, w) * 0.3
    target = scene + plant
    carry = scene + detail + 0.2
    carry[:, :, 3] += 0.1
    out, receipt = reconcile_target_band(carry, target)

    assert out.shape == carry.shape and out.dtype == carry.dtype
    assert torch.allclose(out.mean((-1, -2)), carry.mean((-1, -2)), atol=1e-5)
    added = out - carry
    region = plant > 0.5
    plant_dc = plant - plant.mean()
    assert float(added[..., region].mean()) > 0.6 * float(plant_dc[region].mean())
    assert float((out - target - (carry - target).mean((-1, -2), keepdim=True))[..., region].abs().mean()) < float(
        (carry - target - (carry - target).mean((-1, -2), keepdim=True))[..., region].abs().mean()
    )
    detail_kept = ((out - target) * detail).mean() / detail.square().mean()
    assert float(detail_kept) > 0.95
    assert receipt["policy"] == SOURCE_PREFIX_TARGET_BAND_POLICY
    assert receipt["extra_h3_nfe"] == receipt["extra_vae_calls"] == 0
    assert receipt["temporal_mixing"] is False


def test_target_band_agreement_is_identity_and_frames_stay_independent():
    from h3_flow_regenerate.source_prefix_carry import reconcile_target_band

    gen = torch.Generator().manual_seed(5)
    carry = torch.randn(1, 24, 7, 16, 20, generator=gen)
    same, receipt = reconcile_target_band(carry, carry.clone())
    assert torch.equal(same, carry)
    assert receipt["carry_to_target_rms"] == 0.0

    offset_only, _ = reconcile_target_band(carry, carry + torch.randn(1, 24, 7, 1, 1, generator=gen))
    assert torch.allclose(offset_only, carry, atol=1e-6)

    target = carry.clone()
    target[:, :, 6] += _blob(16, 20, 8, 10, 3.0)
    moved, _ = reconcile_target_band(carry, target)
    assert torch.equal(moved[:, :, :6], carry[:, :, :6])
    assert not torch.equal(moved[:, :, 6], carry[:, :, 6])


def test_target_band_rejects_unpaired_or_non_finite_prefixes():
    from h3_flow_regenerate.source_prefix_carry import reconcile_target_band

    carry = torch.zeros(1, 24, 7, 8, 8)
    with pytest.raises(ValueError, match="paired"):
        reconcile_target_band(carry, torch.zeros(1, 24, 7, 8, 10))
    bad = carry.clone()
    bad[0, 0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        reconcile_target_band(carry, bad)
    with pytest.raises(ValueError, match="positive sigma"):
        reconcile_target_band(carry, carry, sigma=0.0)


@pytest.mark.parametrize("mode", ["native_source_carry", "native_source_carry_target_band"])
def test_projection_mode_selects_target_band_owner(mode):
    model = SimpleNamespace(model_options={})
    configure_source_prefix_projection(model, mode, None, "progressive_uniform_source")
    owner = model.model_options["h3_flow_source_prefix_carry_v1"]
    assert isinstance(owner, SourcePrefixCarry)
    assert owner.target_band is (mode == "native_source_carry_target_band")
