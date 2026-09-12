from __future__ import annotations

import math

import torch

from h3_flow_regenerate import runtime
from h3_flow_regenerate.pr32_suffix_rope_proxy import (
    _RopePlan,
    _block_replacement,
    _model_options_with_suffix_rope_proxy,
    _source_coordinate_target_frame,
    flow_predict_wrapper_with_suffix_rope_proxy,
)


class _FakeModel:
    blocks = (object(), object(), object())


def test_source_coordinate_target_frame_uses_source_native_ratios_at_target_density():
    frame, report = _source_coordinate_target_frame(
        source_hw=(40, 52),
        target_hw=(56, 74),
    )
    assert tuple(frame.shape) == (28 * 37, 2)
    source_area = math.sqrt(40 * 52)
    target_area = math.sqrt(56 * 74)
    source_h_ratio = 40 / source_area
    source_w_ratio = 52 / source_area
    target_h_ratio = 56 / target_area
    target_w_ratio = 74 / target_area
    grid = frame.reshape(28, 37, 2)

    expected_h0 = 32.0 * (1.0 - source_h_ratio) / 2.0
    expected_w0 = 32.0 * (1.0 - source_w_ratio) / 2.0
    expected_h_last = 32.0 * ((1.0 - source_h_ratio) / 2.0 + source_h_ratio * 27 / 28)
    expected_w_last = 32.0 * ((1.0 - source_w_ratio) / 2.0 + source_w_ratio * 36 / 37)
    assert math.isclose(float(grid[0, 0, 0]), expected_h0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(grid[-1, 0, 0]), expected_h_last, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(grid[0, 0, 1]), expected_w0, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(float(grid[0, -1, 1]), expected_w_last, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(
        report["suffix_rope_proxy_h_coordinate_scale"],
        source_h_ratio / target_h_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(
        report["suffix_rope_proxy_w_coordinate_scale"],
        source_w_ratio / target_w_ratio,
        rel_tol=0.0,
        abs_tol=1e-12,
    )
    assert math.isclose(report["suffix_rope_proxy_h_coordinate_scale"], 1.0082080720186268, abs_tol=1e-12)
    assert math.isclose(report["suffix_rope_proxy_w_coordinate_scale"], 0.9918587519318383, abs_tol=1e-12)


def test_block_replacement_changes_only_rope_argument_and_preserves_chain():
    sentinel = object()
    seen = {}

    class Context:
        def rope_for(self, args):
            seen["input"] = args["rope_freqs"]
            return sentinel

    def previous(args, extra):
        seen["forwarded"] = args
        return {"img": args["img"]}

    replacement = _block_replacement(previous, Context())
    args = {"img": torch.tensor([1.0]), "rope_freqs": object(), "other": 7}
    out = replacement(args, {"original_block": lambda forwarded: {"img": forwarded["img"]}})

    assert out["img"].item() == 1.0
    assert seen["forwarded"]["rope_freqs"] is sentinel
    assert seen["forwarded"]["other"] == 7
    assert args["rope_freqs"] is seen["input"]


def test_model_options_proxy_clones_nested_patch_maps():
    prior = lambda args, extra: extra["original_block"](args)  # noqa: E731
    original_dit = {("double_block", 1): prior}
    original = {"transformer_options": {"patches_replace": {"dit": original_dit}, "keep": object()}}
    contract = {"prefix_t": 12, "shapes": ((1, 24, 62, 56, 74), (1, 32, 2, 348))}
    plan = _RopePlan(source_hw=(40, 52), target_hw=(56, 74), prefix_t=12)

    proxied, _context = _model_options_with_suffix_rope_proxy(
        original,
        native_model=_FakeModel(),
        contract=contract,
        plan=plan,
    )

    assert proxied is not original
    assert proxied["transformer_options"] is not original["transformer_options"]
    assert proxied["transformer_options"]["patches_replace"] is not original["transformer_options"]["patches_replace"]
    assert proxied["transformer_options"]["patches_replace"]["dit"] is not original_dit
    assert original_dit == {("double_block", 1): prior}
    assert set(proxied["transformer_options"]["patches_replace"]["dit"]) == {
        ("double_block", 0),
        ("double_block", 1),
        ("double_block", 2),
    }


def test_pr32_overlay_is_the_registered_runtime_predict_wrapper():
    assert runtime.flow_predict_wrapper is flow_predict_wrapper_with_suffix_rope_proxy
    assert getattr(runtime.flow_predict_wrapper, "_h3_pr32_suffix_rope_proxy", False) is True
