from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.comfy_compat import _validate_vdn_target_sparse_compat, validate_h3_model
from h3_flow_regenerate.keyless_compat import (
    KEYLESS_CONTRACT_KEY,
    KeylessCompatibilityError,
    validate_keyless_contract,
)
from h3_flow_regenerate.mixed_grid import MixedGridPlan, _validate_keyless_mixed_measure
from h3_flow_regenerate.partitioned_scheduler import (
    PartitionedPreflightUnsupported,
    _validate_partitioned_keyless_compat,
)


class _Contract:
    api = 1
    architecture = "h3_keyless_core50_v1"
    core_blocks = 50
    token_refiner = "native_qkv"
    token_refiner_blocks = 2
    heads = 56
    head_dim = 128
    inner_dim = 7168
    hidden_size = 5376
    routing_source = "value"
    retrieval_source = "raw_projected_value"
    routing_norm = "rmsnorm"
    routing_norm_epsilon = 1e-5
    rope_policy = "h3_split_half_96_v1"
    qv_order = "q_effective;v"
    projection_attr = "qv_proj"
    checkpoint_format_version = 1

    def __init__(self, **changes):
        for name, value in changes.items():
            setattr(self, name, value)

    def identity(self):
        return (
            self.api,
            self.architecture,
            self.checkpoint_format_version,
            self.qv_order,
            self.heads,
            self.head_dim,
            self.inner_dim,
            self.routing_source,
            self.retrieval_source,
            self.rope_policy,
            "test-provenance",
        )


def _attention(*, fake_qkv: bool = False):
    value = SimpleNamespace(
        qv_proj=SimpleNamespace(weight=SimpleNamespace(shape=(14336, 5376))),
        q_norm=object(),
        route_norm=object(),
        head_dim=128,
    )
    if fake_qkv:
        value.qkv_proj = object()
    return value


def _keyless_diffusion(*, contract=None, fake_qkv: bool = False):
    block = SimpleNamespace(attn=_attention(fake_qkv=fake_qkv))
    value = SimpleNamespace(
        patch_size=(1, 2, 2),
        latents_dim=24,
        audio_latents_dim=32,
        sigma_shift_video=12.0,
        sigma_shift_audio=3.0,
        blocks=[block for _ in range(50)],
        token_refiner=SimpleNamespace(blocks=[object(), object()]),
    )
    setattr(value, KEYLESS_CONTRACT_KEY, _Contract() if contract is None else contract)
    return value


def _native_diffusion():
    return SimpleNamespace(
        patch_size=(1, 2, 2),
        latents_dim=24,
        audio_latents_dim=32,
        sigma_shift_video=12.0,
        sigma_shift_audio=3.0,
    )


def _wrapped_model(diffusion, *, keyless_base: bool):
    MiniMaxH3 = type("MiniMaxH3", (), {})
    base_type = type("KeylessBaseModel", (MiniMaxH3,), {}) if keyless_base else MiniMaxH3
    base = base_type()
    base.diffusion_model = diffusion
    return SimpleNamespace(model=base)


def test_keyless_contract_accepts_canonical_core50_qv_shape():
    diffusion = _keyless_diffusion()
    assert validate_keyless_contract(diffusion) is getattr(diffusion, KEYLESS_CONTRACT_KEY)


def test_keyless_contract_rejects_fake_dead_qkv_projection():
    with pytest.raises(KeylessCompatibilityError, match="fake/dead K path"):
        validate_keyless_contract(_keyless_diffusion(fake_qkv=True))


def test_keyless_contract_rejects_malformed_public_identity():
    with pytest.raises(KeylessCompatibilityError, match="routing_source"):
        validate_keyless_contract(_keyless_diffusion(contract=_Contract(routing_source="key")))


def test_validate_h3_model_accepts_canonical_keyless_base_subclass():
    diffusion = _keyless_diffusion()
    assert validate_h3_model(_wrapped_model(diffusion, keyless_base=True)) is diffusion


def test_validate_h3_model_preserves_exact_native_base_admission():
    diffusion = _native_diffusion()
    assert validate_h3_model(_wrapped_model(diffusion, keyless_base=False)) is diffusion


def test_validate_h3_model_rejects_malformed_explicit_keyless_advertisement_even_on_native_base():
    diffusion = _keyless_diffusion(contract=_Contract(routing_source="key"))
    with pytest.raises(KeylessCompatibilityError, match="routing_source"):
        validate_h3_model(_wrapped_model(diffusion, keyless_base=False))


def test_validate_h3_model_rejects_keyless_contract_on_unrelated_outer_base():
    diffusion = _keyless_diffusion()
    unrelated = type("UnrelatedBase", (), {})()
    unrelated.diffusion_model = diffusion
    with pytest.raises(TypeError, match="native/Keyless MiniMax H3 contract"):
        validate_h3_model(SimpleNamespace(model=unrelated))


def _mixed_plan(*, attention_measure: bool):
    prefix = torch.zeros((1, 24, 1, 4, 4), dtype=torch.float32)
    return MixedGridPlan(
        prefix=prefix,
        temporal=2,
        source_h=2,
        source_w=2,
        prefix_noise=prefix.clone(),
        attention_measure=attention_measure,
    )


def test_mixed_grid_keyless_hidden_sequence_is_not_rejected_without_kv_reduction():
    _validate_keyless_mixed_measure(_keyless_diffusion(), _mixed_plan(attention_measure=False))


def test_mixed_grid_keyless_rejects_legacy_post_projection_kv_reduction():
    with pytest.raises(RuntimeError, match="exact selected V domain"):
        _validate_keyless_mixed_measure(_keyless_diffusion(), _mixed_plan(attention_measure=True))


def test_partitioned_keyless_falls_back_before_qkv_owned_sampler_route():
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=_keyless_diffusion()))
    with pytest.raises(PartitionedPreflightUnsupported, match="current API-4 partition route is QKV-only"):
        _validate_partitioned_keyless_compat(patcher)


def test_partitioned_preflight_guard_does_not_change_ordinary_h3():
    patcher = SimpleNamespace(model=SimpleNamespace(diffusion_model=_native_diffusion()))
    _validate_partitioned_keyless_compat(patcher)


def _model_patcher_with_vdn(diffusion, *, keyless_base: bool):
    patcher = _wrapped_model(diffusion, keyless_base=keyless_base)
    owner = SimpleNamespace(_vdn_forward=True, _vdn_external_sequence_api=4)
    patcher.object_patches = {"diffusion_model.blocks.0.attn.forward": owner}
    return patcher


def test_keyless_rejects_current_qkv_vdn_owner_before_target_sparse_or_mixed_runtime():
    patcher = _model_patcher_with_vdn(_keyless_diffusion(), keyless_base=True)
    with pytest.raises(RuntimeError, match="Keyless-specific value-derived branch/checkpoint"):
        _validate_vdn_target_sparse_compat(patcher, 50, minimum_api=4)


def test_native_vdn_api_check_is_unchanged():
    patcher = _model_patcher_with_vdn(_native_diffusion(), keyless_base=False)
    _validate_vdn_target_sparse_compat(patcher, 50, minimum_api=4)
