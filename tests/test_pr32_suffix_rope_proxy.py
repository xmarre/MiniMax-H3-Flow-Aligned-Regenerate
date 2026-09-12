from __future__ import annotations

import math

import torch

from h3_flow_regenerate import runtime
from h3_flow_regenerate.pr32_suffix_rope_proxy import (
    _RopePlan,
    _block_replacement,
    _model_options_with_suffix_rope_proxy,
    _source_extent_target_frame,
    flow_predict_wrapper_with_suffix_rope_proxy,
)


class _NativeFrameGrid:
    @staticmethod
    def _frame_grid(lat_h: int, lat_w: int):
        sqrt_area = math.sqrt(lat_h * lat_w)
        ratio_h = lat_h / sqrt_area
        ratio_w = lat_w / sqrt_area
        nh, nw = lat_h // 2, lat_w // 2
        h = (torch.arange(nh, dtype=torch.float64) * (ratio_h / nh) + (1.0 - ratio_h) / 2.0) * 32.0
        w = (torch.arange(nw, dtype=torch.float64) * (ratio_w / nw) + (1.0 - ratio_w) / 2.0) * 32.0
        hh, ww = torch.meshgrid(h, w, indexing="ij")
        frame = torch.stack((hh.reshape(-1), ww.reshape(-1)), dim=-1)
        return frame, (nh, nw)


class _FakeModel:
    blocks = (object(), object(), object())


def test_source_extent_target_frame_matches_source_endpoints():
    frame, report = _source_extent_target_frame(
        _NativeFrameGrid,
        source_hw=(40, 52),
        target_hw=(56, 74),
    )
    assert tuple(frame.shape) == (28 * 37, 2)
    grid = frame.reshape(28, 37, 2)
    assert math.isclose(float(grid[0, 0, 0]), report["suffix_rope_proxy_source_h_endpoints"][0])
    assert math.isclose(float(grid[-1, 0, 0]), report["suffix_rope_proxy_source_h_endpoints"][1])
    assert math.isclose(float(grid[0, 0, 1]), report["suffix_rope_proxy_source_w_endpoints"][0])
    assert math.isclose(float(grid[0, -1, 1]), report["suffix_rope_proxy_source_w_endpoints"][1])
    assert math.isclose(report["suffix_rope_proxy_h_extent_ratio"], 0.9932716736, rel_tol=0.0, abs_tol=1e-9)
    assert math.isclose(report["suffix_rope_proxy_w_extent_ratio"], 0.9802009670, rel_tol=0.0, abs_tol=1e-9)


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
