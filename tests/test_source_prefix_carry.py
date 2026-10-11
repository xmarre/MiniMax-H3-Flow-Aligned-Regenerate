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
