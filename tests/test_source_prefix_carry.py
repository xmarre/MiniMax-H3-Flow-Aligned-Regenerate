"""Pairing, native phase, transaction publication and bounded carry ownership."""

import math
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


def _textured_bar(h, w, x0, width=5):
    """A mid-size object whose fine texture moves with it: it spans both bands."""
    xx = torch.arange(w).float().view(1, w).expand(h, w)
    yy = torch.arange(h).float().view(h, 1).expand(h, w)
    inside = ((xx >= x0) & (xx < x0 + width)).float()
    return inside * (1.0 + 0.4 * ((yy + xx - x0) % 2 * 2 - 1))


def _scene(seed, h, w, t=3):
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(1, 24, t, h, w, generator=gen) * 0.05


def _centered(value):
    return value - value.mean((-1, -2), keepdim=True)


def _reconcile(carry, target):
    from h3_flow_regenerate.source_prefix_carry import reconcile_target_band

    out, receipt = reconcile_target_band(carry, target)
    assert out.shape == carry.shape and out.dtype == carry.dtype
    assert receipt["extra_h3_nfe"] == receipt["extra_vae_calls"] == 0
    assert receipt["temporal_mixing"] is False
    return out, receipt


def _detail_kept(out, target, detail, region):
    residual = (out - target)[..., region]
    return float((residual * detail[region]).mean() / detail[region].square().mean())


def test_target_band_keeps_carry_detail_where_layout_agrees():
    h, w = 24, 32
    scene = _scene(4, h, w)
    detail = _checker(h, w) * 0.3
    carry = scene + detail + 0.2
    out, receipt = _reconcile(carry, scene)
    assert receipt["policy"] == "native_source_carry_target_band_v1"
    assert receipt["detail_gate_active_fraction"] < 0.1
    assert _detail_kept(out, scene, detail, torch.ones(h, w, dtype=torch.bool)) > 0.95


def test_new_target_object_replaces_carry_layout_and_detail_locally():
    h, w = 24, 32
    scene = _scene(5, h, w)
    detail = _checker(h, w) * 0.3
    plant = _blob(h, w, 8, 22, 2.5)
    carry, target = scene + detail, scene + plant
    out, receipt = _reconcile(carry, target)
    core = plant > 0.5
    expected = target
    assert float((out - expected)[..., core].abs().max()) < 0.1
    far = _blob(h, w, 8, 22, 6.0) < 0.01
    assert _detail_kept(out, target, detail, far) > 0.95
    # The gate spreads over the mid band's support, several cells around the
    # change; beyond it the carry detail is untouched.
    assert 0.0 < receipt["detail_gate_active_fraction"] < 0.75


def test_camera_translation_does_not_mix_displaced_detail_or_double_edges():
    h, w = 20, 32
    scene = _scene(6, h, w)
    carry = scene + _textured_bar(h, w, 10)
    target = scene + _textured_bar(h, w, 14)
    out, _ = _reconcile(carry, target)
    expected = target
    # Columns covered by either bar position, plus a one-cell margin.
    region = torch.zeros(h, w, dtype=torch.bool)
    region[:, 9:20] = True
    error = (out - expected)[..., region].abs()
    assert float(error.max()) < 0.12
    # The carry-only bar position keeps no carry texture or edge.
    old = torch.zeros(h, w, dtype=torch.bool)
    old[:, 10:14] = True
    assert float((out - expected)[..., old].abs().mean()) < 0.03


def test_subcell_motion_keeps_native_detail():
    h, w = 24, 32
    xx = torch.arange(w).float().view(1, w)
    yy = torch.arange(h).float().view(h, 1)
    smooth = lambda shift: torch.sin(2 * math.pi * (xx - shift) / 16) * torch.cos(2 * math.pi * yy / 12)  # noqa: E731
    detail = _checker(h, w) * 0.3
    carry = _scene(7, h, w) + smooth(0.0) + detail
    target = _scene(7, h, w) + smooth(0.25)
    out, receipt = _reconcile(carry, target)
    assert receipt["detail_gate_weight_mean"] < 0.05
    assert _detail_kept(out, target, detail, torch.ones(h, w, dtype=torch.bool)) > 0.9


def test_disoccluded_background_drops_the_carried_object():
    h, w = 24, 32
    scene = _scene(8, h, w)
    occluder = 1.5 * _blob(h, w, 12, 10, 2.5) * (1 + 0.3 * _checker(h, w))
    carry, target = scene + occluder, scene.clone()
    out, _ = _reconcile(carry, target)
    expected = target
    core = _blob(h, w, 12, 10, 2.5) > 0.3
    assert float((out - expected)[..., core].abs().max()) < 0.15
    assert float((out - expected)[..., core].abs().mean()) < 0.05 * float(occluder[core].abs().mean())


def test_scene_cut_in_returned_prefix_is_followed():
    h, w = 24, 32
    gen = torch.Generator().manual_seed(9)
    carry = torch.randn(1, 24, 3, h, w, generator=gen)
    target = torch.randn(1, 24, 3, h, w, generator=gen) + 0.5
    out, receipt = _reconcile(carry, target)
    assert receipt["detail_gate_full_fraction"] > 0.6
    assert float((out - target).square().mean().sqrt()) < 0.35 * float(_centered(target).square().mean().sqrt())


def test_target_band_identity_global_offset_and_frame_independence():
    gen = torch.Generator().manual_seed(5)
    carry = torch.randn(1, 24, 7, 16, 20, generator=gen)
    same, receipt = _reconcile(carry, carry.clone())
    assert torch.equal(same, carry)
    assert receipt["carry_to_target_rms"] == 0.0

    # A target-wide latent offset is adopted without counting as disagreement.
    offset = carry + torch.randn(1, 24, 7, 1, 1, generator=gen)
    offset_only, receipt = _reconcile(carry, offset)
    assert torch.allclose(offset_only, offset, atol=1e-5)
    assert receipt["detail_gate_active_fraction"] == 0.0

    target = carry.clone()
    target[:, :, 6] += _blob(16, 20, 8, 10, 3.0)
    moved, _ = _reconcile(carry, target)
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
    huge = torch.zeros(1, 24, 1, 8, 8)
    huge[..., 0, 0] = 1.0e20
    with pytest.raises(ValueError, match="non-finite prefix"):
        reconcile_target_band(huge, -huge)
    with pytest.raises(ValueError, match="increasing gate"):
        reconcile_target_band(carry, carry, gate=(0.7, 0.35))


@pytest.mark.parametrize("mode", ["native_source_carry", "native_source_carry_target_band"])
def test_projection_mode_selects_target_band_owner(mode):
    model = SimpleNamespace(model_options={})
    configure_source_prefix_projection(model, mode, None, "progressive_uniform_source")
    owner = model.model_options["h3_flow_source_prefix_carry_v1"]
    assert isinstance(owner, SourcePrefixCarry)
    assert owner.target_band is (mode == "native_source_carry_target_band")


def _smooth(h, w):
    yy, xx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    return torch.sin(2 * math.pi * xx / 16) + 0.5 * torch.cos(2 * math.pi * yy / 12)


def test_difference_only_above_the_band_keeps_carry_detail():
    # Bound of the guarantee: the gate sees mid-band structure only. Shared
    # layout with replaced fine texture keeps the carry's texture.
    h, w = 24, 32
    smooth, checker = _smooth(h, w), _checker(h, w)
    carry = (smooth + 0.5 * checker).expand(1, 24, 3, h, w).clone()
    target = (smooth - 0.5 * checker).expand(1, 24, 3, h, w).clone()
    out, receipt = _reconcile(carry, target)
    assert receipt["detail_gate_active_fraction"] == 0.0
    assert float((out - carry).square().mean().sqrt()) < 0.01


def test_varying_difference_mean_is_not_conserved_exactly_at_borders():
    h, w = 24, 32
    carry = (3 * _smooth(h, w)).expand(1, 24, 3, h, w).clone()
    target = carry.clone()
    target[..., 0, 0] += 1.0
    out, receipt = _reconcile(carry, target)
    assert receipt["detail_gate_active_fraction"] == 0.0
    target_shift = float((target - carry).mean())
    output_shift = float((out - carry).mean())
    assert 0.0 < output_shift < target_shift
