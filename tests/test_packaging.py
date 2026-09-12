from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_package_import_is_safe_without_comfy(monkeypatch):
    import builtins
    import importlib
    import sys

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "comfy" or name.startswith("comfy."):
            raise ImportError("blocked comfy import")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    for name in list(sys.modules):
        if name == "h3_flow_regenerate" or name.startswith("h3_flow_regenerate."):
            sys.modules.pop(name, None)
    package = importlib.import_module("h3_flow_regenerate")
    assert package is not None


def test_top_level_package_exports_comfyui_node_mappings_when_loaded_as_package():
    import importlib.util
    import sys

    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location(
        "flow_package_test",
        root / "__init__.py",
        submodule_search_locations=[str(root)],
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        assert "H3FlowTrajectory" in module.NODE_CLASS_MAPPINGS
        assert "H3ProgressiveHandoff" in module.NODE_CLASS_MAPPINGS
        assert "H3ProgressiveTargetInputHandoff" in module.NODE_CLASS_MAPPINGS
        assert "H3ProgressiveTargetSparseHandoff" in module.NODE_CLASS_MAPPINGS
        assert "H3ProgressiveMixedGridHandoff" in module.NODE_CLASS_MAPPINGS
        assert "H3ContinuumAudioBoundaryPipelineDiagnostic" in module.NODE_CLASS_MAPPINGS
    finally:
        sys.modules.pop(spec.name, None)


def test_pyproject_runtime_dependencies_stay_minimal():
    import tomllib

    root = Path(__file__).resolve().parents[1]
    data = tomllib.loads((root / "pyproject.toml").read_text())
    dependencies = data["project"]["dependencies"]
    assert dependencies == ["torch>=2.5", "numpy>=1.25"]


def test_workflows_are_valid_json():
    root = Path(__file__).resolve().parents[1]
    workflows = root / "workflows"
    for path in workflows.glob("*.json"):
        with path.open("r", encoding="utf-8") as handle:
            assert isinstance(json.load(handle), dict)


def test_progressive_target_input_schema_contains_expected_controls():
    from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff

    required = H3ProgressiveTargetInputHandoff.INPUT_TYPES()["required"]
    assert {
        "model",
        "trajectory",
        "source_mode",
        "source_scale",
        "source_width",
        "source_height",
        "handoff_coordinate",
        "handoff_selection",
        "guidance_mode",
        "direction_weight",
        "acceleration_weight",
        "consistency_weight",
        "low_frequency_cutoff",
        "temporal_weight",
        "handoff_transfer",
    }.issubset(required)
    names = list(required)
    assert names.index("temporal_weight") > names.index("low_frequency_cutoff")


def test_target_input_progressive_defaults_to_learned_handoff_with_bicubic_control():
    from h3_flow_regenerate.nodes import H3ProgressiveHandoff, H3ProgressiveTargetInputHandoff
    from h3_flow_regenerate.target_sparse_node import H3ProgressiveTargetSparseHandoff

    for node in (H3ProgressiveTargetInputHandoff, H3ProgressiveTargetSparseHandoff):
        target_schema = node.INPUT_TYPES()
        assert target_schema["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]
        assert target_schema["required"]["handoff_transfer"][1]["default"] == "learned_3d"
        assert target_schema["optional"]["learned_upscaler"] == ("H3_LATENT_UPSCALER",)
    assert "handoff_transfer" not in H3ProgressiveHandoff.INPUT_TYPES()["required"]


def test_target_sparse_compat_node_is_explicitly_experimental_dense_control():
    from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff
    from h3_flow_regenerate.target_sparse_node import H3ProgressiveTargetSparseHandoff

    target_schema = H3ProgressiveTargetInputHandoff.INPUT_TYPES()
    sparse_schema = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()
    assert "suffix_dc_bridge" not in target_schema["required"]
    sparse_required = sparse_schema["required"].copy()
    bridge = sparse_required.pop("suffix_dc_bridge")
    assert bridge[0] == "BOOLEAN"
    assert bridge[1]["default"] is False
    assert sparse_required == target_schema["required"]
    assert sparse_schema["optional"] == target_schema["optional"]
    assert H3ProgressiveTargetSparseHandoff.CATEGORY.endswith("/experimental")
    assert H3ProgressiveTargetSparseHandoff.EXACT_PREFIX_MODE == "fallback"
    assert "full target-grid" in H3ProgressiveTargetSparseHandoff.DESCRIPTION


def test_metrics_json_output_node_saves_unique_json_and_refreshes_after_sampler(monkeypatch, tmp_path):
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.nodes import H3MetricsJSON

    allocations = 0

    def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
        nonlocal allocations
        del image_width, image_height
        allocations += 1
        subfolder = "bench"
        folder = tmp_path / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder), filename_prefix, allocations, subfolder, filename_prefix

    class _FolderPaths:
        get_output_directory = staticmethod(lambda: str(tmp_path))
        get_save_image_path = staticmethod(get_save_image_path)

    monkeypatch.setitem(__import__("sys").modules, "folder_paths", _FolderPaths)
    node = H3MetricsJSON()
    metrics = H3FlowMetrics()
    metrics.increment("model_calls", 3)
    metrics.event("probe", value=7)

    ui_1 = node.save(metrics, "metrics/test")
    ui_2 = node.save(metrics, "metrics/test")
    assert allocations == 2
    first = ui_1["ui"]["text"][0]
    second = ui_2["ui"]["text"][0]
    assert first != second
    assert Path(first).exists()
    assert Path(second).exists()
    payload = json.loads(Path(first).read_text())
    assert payload["counters"]["model_calls"] == 3
    assert payload["events"][0]["kind"] == "probe"


def test_metrics_json_output_requires_metrics():
    from h3_flow_regenerate.nodes import H3MetricsJSON

    with pytest.raises(TypeError):
        H3MetricsJSON().save(None)
