from __future__ import annotations

from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff
from h3_flow_regenerate.target_sparse_node import H3ProgressiveMixedGridHandoff, H3ProgressiveTargetSparseHandoff


def _patch_kwargs():
    return {
        "model": object(),
        "trajectory": object(),
        "source_mode": "scale",
        "source_scale": 0.7,
        "source_width": 864,
        "source_height": 640,
        "handoff_coordinate": 0.35,
        "handoff_selection": "fixed",
        "guidance_mode": "direction",
        "direction_weight": 0.25,
        "acceleration_weight": 0.0,
        "consistency_weight": 0.0,
        "low_frequency_cutoff": 0.25,
        "temporal_weight": 0.2,
        "handoff_transfer": "bicubic",
        "learned_upscaler": None,
    }


def test_target_sparse_node_sets_opt_in_exact_prefix_mode(monkeypatch):
    captured = {}

    def fake_patch_flow_model(model, **kwargs):
        captured.update(kwargs)
        return model, object()

    monkeypatch.setattr("h3_flow_regenerate.target_sparse_node.patch_flow_model", fake_patch_flow_model)
    kwargs = _patch_kwargs()
    model = kwargs["model"]

    patched, metrics = H3ProgressiveTargetSparseHandoff().patch(**kwargs)

    assert patched is model
    assert metrics is captured["metrics"]
    progressive = captured["progressive"]
    assert isinstance(progressive, ProgressiveTargetInputConfig)
    assert progressive.exact_prefix_mode == "target_sparse_lifter"
    assert progressive.source_scale == 0.7
    assert progressive.transfer_mode == "bicubic"
    assert captured["capture_enabled"] is True
    assert captured["capture_forecasts"] is False
    assert captured["clear_guidance_conditioning_signature"] is True
    assert captured["clear_guidance_run_id"] is True


def test_target_sparse_node_pixel_mode_preserves_existing_source_geometry_semantics(monkeypatch):
    captured = {}

    def fake_patch_flow_model(model, **kwargs):
        captured.update(kwargs)
        return model, object()

    monkeypatch.setattr("h3_flow_regenerate.target_sparse_node.patch_flow_model", fake_patch_flow_model)
    kwargs = _patch_kwargs()
    kwargs.update(source_mode="pixels", source_width=672, source_height=480)

    H3ProgressiveTargetSparseHandoff().patch(**kwargs)

    progressive = captured["progressive"]
    assert progressive.exact_prefix_mode == "target_sparse_lifter"
    assert progressive.source_scale is None
    assert progressive.source_latent_h is not None
    assert progressive.source_latent_w is not None


def test_suffix_dc_bridge_is_exposed_only_on_continuum_specific_progressive_nodes():
    target_inputs = H3ProgressiveTargetInputHandoff.INPUT_TYPES()
    sparse_inputs = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()
    mixed_inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()
    assert "suffix_dc_bridge" not in target_inputs["required"]
    for inputs in (sparse_inputs, mixed_inputs):
        bridge = inputs["required"]["suffix_dc_bridge"]
        assert bridge[0] == "BOOLEAN"
        assert bridge[1]["default"] is True
    assert mixed_inputs["required"]["handoff_transfer"][0] == ["learned_3d"]
    assert sparse_inputs["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]
