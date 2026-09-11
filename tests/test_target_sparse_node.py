from __future__ import annotations

from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff
from h3_flow_regenerate.target_sparse_node import H3ProgressiveMixedGridHandoff, H3ProgressiveTargetSparseHandoff


class _LearnedProvider:
    api_version = 1
    kind = "minimax_h3_learned_latent_upscaler"
    model_name = "minimax_h3_latent_upscaler_3d_bf16.safetensors"
    device = "cuda"
    inference_device = "cuda"
    precision = "bf16"
    offload_after_upscale = False

    def upscale_clean_video(self, video, *, target_h, target_w):
        return video


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


def test_generic_target_input_ui_defaults_match_shipped_workflow():
    required = H3ProgressiveTargetInputHandoff.INPUT_TYPES()["required"]

    assert required["source_mode"][1]["default"] == "scale"
    assert required["source_scale"][1]["default"] == 0.70
    assert required["source_width"][1]["default"] == 864
    assert required["source_height"][1]["default"] == 640
    assert required["handoff_coordinate"][1]["default"] == 0.35
    assert required["handoff_selection"][1]["default"] == "fixed"
    assert required["guidance_mode"][1]["default"] == "direction+temporal"
    assert required["direction_weight"][1]["default"] == 0.25
    assert required["acceleration_weight"][1]["default"] == 0.25
    assert required["consistency_weight"][1]["default"] == 0.25
    assert required["low_frequency_cutoff"][1]["default"] == 0.25
    assert required["temporal_weight"][1]["default"] == 0.20
    assert required["handoff_transfer"][0] == ["bicubic", "learned_3d"]
    assert required["handoff_transfer"][1]["default"] == "learned_3d"


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
    assert progressive.suffix_geometric_bridge is False
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


def test_target_sparse_inherits_learned_transfer_default():
    required = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()["required"]

    assert required["handoff_transfer"][0] == ["bicubic", "learned_3d"]
    assert required["handoff_transfer"][1]["default"] == "learned_3d"


def test_target_sparse_direct_call_uses_inherited_learned_transfer_default(monkeypatch):
    captured = {}

    def fake_patch_flow_model(model, **kwargs):
        captured.update(kwargs)
        return model, object()

    monkeypatch.setattr("h3_flow_regenerate.target_sparse_node.patch_flow_model", fake_patch_flow_model)
    kwargs = _patch_kwargs()
    kwargs.pop("handoff_transfer")
    kwargs["learned_upscaler"] = _LearnedProvider()

    H3ProgressiveTargetSparseHandoff().patch(**kwargs)

    progressive = captured["progressive"]
    assert progressive.exact_prefix_mode == "target_sparse_lifter"
    assert progressive.transfer_mode == "learned_3d"
    assert progressive.learned_upscaler is kwargs["learned_upscaler"]
    assert progressive.suffix_geometric_bridge is False


def test_mixed_grid_ui_defaults_match_canonical_workflow():
    inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()
    required = inputs["required"]

    assert required["source_mode"][1]["default"] == "scale"
    assert required["source_scale"][1]["default"] == 0.70
    assert required["source_width"][1]["default"] == 864
    assert required["source_height"][1]["default"] == 640
    assert required["handoff_coordinate"][1]["default"] == 0.35
    assert required["handoff_selection"][1]["default"] == "fixed"
    assert required["guidance_mode"][1]["default"] == "direction+temporal"
    assert required["direction_weight"][1]["default"] == 0.25
    assert required["acceleration_weight"][1]["default"] == 0.25
    assert required["consistency_weight"][1]["default"] == 0.25
    assert required["low_frequency_cutoff"][1]["default"] == 0.25
    assert required["temporal_weight"][1]["default"] == 0.20
    assert required["handoff_transfer"] == (["learned_3d"], required["handoff_transfer"][1])
    assert required["handoff_transfer"][1]["default"] == "learned_3d"
    assert required["suffix_dc_bridge"][1]["default"] is True
    assert inputs["optional"]["suffix_geometric_bridge"][1]["default"] is True
    profile = inputs["optional"]["attention_measure_profile"]
    assert profile[0] == ["weighted_measure_v1", "legacy_representative_v1", "off"]
    assert profile[1]["default"] == "weighted_measure_v1"


def test_mixed_grid_direct_call_defaults_to_learned_transfer_and_seam_repair(monkeypatch):
    captured = {}

    def fake_patch_flow_model(model, **kwargs):
        captured.update(kwargs)
        return model, object()

    monkeypatch.setattr("h3_flow_regenerate.target_sparse_node.patch_flow_model", fake_patch_flow_model)
    kwargs = _patch_kwargs()
    kwargs.pop("handoff_transfer")
    kwargs["learned_upscaler"] = _LearnedProvider()
    kwargs.update(
        guidance_mode="direction+temporal",
        acceleration_weight=0.25,
        consistency_weight=0.25,
    )

    H3ProgressiveMixedGridHandoff().patch(**kwargs)

    progressive = captured["progressive"]
    assert progressive.exact_prefix_mode == "mixed_grid_low_suffix"
    assert progressive.transfer_mode == "learned_3d"
    assert progressive.suffix_dc_bridge is True
    assert progressive.suffix_geometric_bridge is True
    # Direct calls that omit the new field intentionally retain serialized legacy semantics.
    assert progressive.attention_measure_profile is None
    assert progressive.learned_upscaler is kwargs["learned_upscaler"]


def test_mixed_grid_explicit_weighted_profile_reaches_runtime_config(monkeypatch):
    captured = {}

    def fake_patch_flow_model(model, **kwargs):
        captured.update(kwargs)
        return model, object()

    monkeypatch.setattr("h3_flow_regenerate.target_sparse_node.patch_flow_model", fake_patch_flow_model)
    kwargs = _patch_kwargs()
    kwargs.update(
        handoff_transfer="learned_3d",
        learned_upscaler=_LearnedProvider(),
        attention_measure_profile="weighted_measure_v1",
    )

    H3ProgressiveMixedGridHandoff().patch(**kwargs)

    progressive = captured["progressive"]
    assert progressive.exact_prefix_mode == "mixed_grid_low_suffix"
    assert progressive.attention_measure_profile == "weighted_measure_v1"
    assert progressive.suffix_geometric_bridge is True


def test_suffix_dc_bridge_is_exposed_only_on_continuum_specific_progressive_nodes():
    target_inputs = H3ProgressiveTargetInputHandoff.INPUT_TYPES()
    sparse_inputs = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()
    mixed_inputs = H3ProgressiveMixedGridHandoff.INPUT_TYPES()
    assert "suffix_dc_bridge" not in target_inputs["required"]
    assert "attention_measure_profile" not in target_inputs.get("optional", {})
    assert "attention_measure_profile" not in sparse_inputs.get("optional", {})
    assert "attention_measure_profile" in mixed_inputs["optional"]
    for inputs in (sparse_inputs, mixed_inputs):
        bridge = inputs["required"]["suffix_dc_bridge"]
        assert bridge[0] == "BOOLEAN"
        assert bridge[1]["default"] is True
    assert mixed_inputs["required"]["handoff_transfer"][0] == ["learned_3d"]
    assert sparse_inputs["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]
