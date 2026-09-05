from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.comfy_compat import _validate_vdn_target_sparse_compat
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.target_sparse import (
    TARGET_SPARSE_CONTRACT_KEY,
    VDN_EXTERNAL_SEQUENCE_API_VERSION,
    VDN_EXTERNAL_SEQUENCE_KEY,
    VDN_EXTERNAL_SEQUENCE_MODE,
    TargetSparsePlan,
    make_target_sparse_block_wrapper,
    target_sparse_contract,
)


def _plan():
    return TargetSparsePlan(
        target_t=1,
        target_h=4,
        target_w=4,
        source_h=2,
        source_w=2,
        selected_video_rows=torch.tensor([0, 3], dtype=torch.long),
        anchor_video_rows=torch.tensor([3], dtype=torch.long),
        protected_video_rows=torch.tensor([0], dtype=torch.long),
    )


def _layout():
    return SimpleNamespace(
        segments=[(0, 2, "text"), (2, 6, "video")],
        seq_len=6,
    )


def test_target_sparse_block_publishes_vdn_external_sequence_contract():
    plan = _plan()
    full_img = torch.randn(6, 3)
    rope = torch.randn(1, 6, 1, 1, 1, 1)
    seen = {}

    def previous(args, _extra):
        seen.update(args["transformer_options"][VDN_EXTERNAL_SEQUENCE_KEY])
        return {"img": args["img"]}

    wrapper = make_target_sparse_block_wrapper(0, 2, H3FlowMetrics(), previous=previous)
    out = wrapper(
        {
            "img": full_img,
            "rope_freqs": rope,
            "mod_segments": [(0, 2, 0), (2, 6, 0)],
            "layout": _layout(),
            "transformer_options": {TARGET_SPARSE_CONTRACT_KEY: target_sparse_contract(plan)},
        },
        {"original_block": None},
    )

    assert out["img"].shape[0] == 4
    assert seen == {
        "api": VDN_EXTERNAL_SEQUENCE_API_VERSION,
        "mode": VDN_EXTERNAL_SEQUENCE_MODE,
        "full_sequence_rows": 6,
        "reduced_sequence_rows": 4,
    }


def test_target_sparse_block_refuses_to_overwrite_an_existing_vdn_contract():
    plan = _plan()
    wrapper = make_target_sparse_block_wrapper(
        0,
        1,
        H3FlowMetrics(),
        previous=lambda args, _extra: {"img": args["img"]},
    )
    options = {
        TARGET_SPARSE_CONTRACT_KEY: target_sparse_contract(plan),
        VDN_EXTERNAL_SEQUENCE_KEY: {"api": 999},
    }

    with pytest.raises(RuntimeError, match="existing VDN external-sequence contract"):
        wrapper(
            {
                "img": torch.randn(6, 3),
                "rope_freqs": torch.randn(1, 6, 1, 1, 1, 1),
                "mod_segments": [(0, 2, 0), (2, 6, 0)],
                "layout": _layout(),
                "transformer_options": options,
            },
            {"original_block": None},
        )


def _vdn_owner(api=None):
    def owner(*_args, **_kwargs):
        raise AssertionError("owner should not execute during compatibility validation")

    owner._vdn_forward = True
    if api is not None:
        owner._vdn_external_sequence_api = api
    return owner


def test_target_sparse_accepts_vdn_with_reduced_sequence_capability():
    model = SimpleNamespace(
        object_patches={"diffusion_model.blocks.0.attn.forward": _vdn_owner(VDN_EXTERNAL_SEQUENCE_API_VERSION)}
    )
    _validate_vdn_target_sparse_compat(model, 1)


def test_target_sparse_fails_before_sampling_on_old_vdn_attention_patch():
    model = SimpleNamespace(object_patches={"diffusion_model.blocks.0.attn.forward": _vdn_owner()})
    with pytest.raises(RuntimeError, match="VDN-H3 attention without the required"):
        _validate_vdn_target_sparse_compat(model, 1)


def test_target_sparse_ignores_unrelated_attention_object_patch():
    model = SimpleNamespace(object_patches={"diffusion_model.blocks.0.attn.forward": lambda *args, **kwargs: None})
    _validate_vdn_target_sparse_compat(model, 1)
