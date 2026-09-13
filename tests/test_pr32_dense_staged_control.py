from types import SimpleNamespace

import torch

import h3_flow_regenerate.pr32_dense_staged_control  # noqa: F401
from h3_flow_regenerate import comfy_compat, target_sparse
from h3_flow_regenerate.metrics import H3FlowMetrics


def _dense_plan():
    rows = torch.arange(4, dtype=torch.long)
    return target_sparse.TargetSparsePlan(
        target_t=1,
        target_h=4,
        target_w=4,
        source_h=2,
        source_w=2,
        selected_video_rows=rows,
        anchor_video_rows=torch.tensor([0], dtype=torch.long),
        protected_video_rows=torch.tensor([0], dtype=torch.long),
        dense_collar_video_rows=torch.tensor([1, 2, 3], dtype=torch.long),
        dense_collar_t=1,
    )


def _layout():
    return SimpleNamespace(
        segments=[(0, 2, "text"), (2, 4, "audio"), (4, 8, "video")],
        signature=(2, 1, 4, 4, 1),
        seq_len=8,
    )


def _args(plan, *, img=None):
    if img is None:
        img = torch.arange(24, dtype=torch.float32).reshape(8, 3)
    return {
        "img": img,
        "rope_freqs": torch.zeros(1, 8, 1, 1, 1, 1),
        "mod_segments": [(0, 2, 1), (2, 4, 2), (4, 8, torch.arange(4))],
        "layout": _layout(),
        "transformer_options": {target_sparse.TARGET_SPARSE_CONTRACT_KEY: target_sparse.target_sparse_contract(plan)},
    }


def test_pr32_dense_identity_keeps_native_sequence_and_omits_vdn_reduced_contract():
    plan = _dense_plan()
    metrics = H3FlowMetrics()
    seen = []

    def previous(args, _extra):
        seen.append(args)
        assert args["img"].shape[0] == 8
        assert args["rope_freqs"].shape[1] == 8
        assert target_sparse.VDN_EXTERNAL_SEQUENCE_KEY not in args["transformer_options"]
        return {"img": args["img"] + 1.0, "sentinel": "kept"}

    first = comfy_compat.make_target_sparse_block_wrapper(0, 2, metrics, previous=previous)
    first_out = first(_args(plan), {"original_block": None})
    assert first_out["img"].shape == (8, 3)
    assert first_out["sentinel"] == "kept"

    last = comfy_compat.make_target_sparse_block_wrapper(1, 2, metrics, previous=previous)
    last_out = last(_args(plan, img=first_out["img"]), {"original_block": None})
    assert last_out["img"].shape == (8, 3)
    assert last_out["sentinel"] == "kept"
    assert len(seen) == 2

    event = [row for row in metrics.events if row.kind == "target_sparse_transformer"][-1]
    assert event.fields["selected_video_rows"] == 4
    assert event.fields["full_video_rows"] == 4
    assert event.fields["reduced_sequence_rows"] == 8
    assert event.fields["full_sequence_rows"] == 8
    assert event.fields["video_row_fraction"] == 1.0
    assert event.fields["vdn_external_sequence_mode"] == "native_full_grid"
    assert event.fields["dense_identity"] is True

    lift = [row for row in metrics.events if row.kind == "target_sparse_lift"][-1]
    assert lift.fields["interpolation"] == "identity_full_grid"
    assert lift.fields["retained_rows_overwritten_exactly"] == 4
    assert lift.fields["dense_identity"] is True


def test_pr32_partial_sparse_plan_still_delegates_to_original_reduced_contract():
    plan = _dense_plan()
    partial = target_sparse.TargetSparsePlan(
        target_t=plan.target_t,
        target_h=plan.target_h,
        target_w=plan.target_w,
        source_h=plan.source_h,
        source_w=plan.source_w,
        selected_video_rows=torch.tensor([0, 3], dtype=torch.long),
        anchor_video_rows=torch.tensor([0], dtype=torch.long),
        protected_video_rows=torch.tensor([0], dtype=torch.long),
        dense_collar_video_rows=torch.tensor([3], dtype=torch.long),
        dense_collar_t=1,
    )
    metrics = H3FlowMetrics()
    seen = []

    def previous(args, _extra):
        seen.append(args)
        contract = args["transformer_options"][target_sparse.VDN_EXTERNAL_SEQUENCE_KEY]
        assert contract["full_sequence_rows"] == 8
        assert contract["reduced_sequence_rows"] == 6
        return {"img": args["img"]}

    wrapper = comfy_compat.make_target_sparse_block_wrapper(0, 2, metrics, previous=previous)
    out = wrapper(_args(partial), {"original_block": None})
    assert out["img"].shape[0] == 6
    assert len(seen) == 1
