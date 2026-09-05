import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def test_package_import_without_comfyui():
    import h3_flow_regenerate

    assert h3_flow_regenerate.H3FlowTrajectory.api_version == 1


def test_project_declares_apache_license_and_tracks_license_file():
    root = Path(__file__).parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    license_text = (root / "LICENSE").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert 'license = "Apache-2.0"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'License = "https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/main/LICENSE"' in pyproject
    assert license_text.startswith("Apache License\n                           Version 2.0, January 2004")
    assert "TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION" in license_text
    assert "END OF TERMS AND CONDITIONS" in license_text
    assert "[Apache License 2.0](LICENSE)" in readme
    assert "Copyright 2026 xmarre." in readme


def test_custom_node_root_registration_smoke():
    root = Path(__file__).parents[1] / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        "h3_flow_custom_node",
        root,
        submodule_search_locations=[str(root.parent)],
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert "H3ProgressiveHandoff" in module.NODE_CLASS_MAPPINGS
    assert "H3ProgressiveTargetSparseHandoff" in module.NODE_CLASS_MAPPINGS
    assert "H3RefineTargetGeometry" in module.NODE_CLASS_MAPPINGS
    assert "H3RuntimeMetricsProbe" in module.NODE_CLASS_MAPPINGS


def test_progressive_nodes_expose_all_selectable_guidance_controls():
    from h3_flow_regenerate.nodes import H3ProgressiveHandoff, H3ProgressiveTargetInputHandoff
    from h3_flow_regenerate.target_sparse_node import H3ProgressiveTargetSparseHandoff

    for node in (H3ProgressiveHandoff, H3ProgressiveTargetInputHandoff, H3ProgressiveTargetSparseHandoff):
        required = node.INPUT_TYPES()["required"]
        assert {
            "guidance_mode",
            "direction_weight",
            "acceleration_weight",
            "consistency_weight",
            "low_frequency_cutoff",
            "temporal_weight",
        }.issubset(required)
        names = list(required)
        assert names.index("temporal_weight") > names.index("low_frequency_cutoff")


def test_target_input_progressive_exposes_optional_learned_handoff_without_changing_default():
    from h3_flow_regenerate.nodes import H3ProgressiveHandoff, H3ProgressiveTargetInputHandoff
    from h3_flow_regenerate.target_sparse_node import H3ProgressiveTargetSparseHandoff

    for node in (H3ProgressiveTargetInputHandoff, H3ProgressiveTargetSparseHandoff):
        target_schema = node.INPUT_TYPES()
        assert target_schema["required"]["handoff_transfer"][0] == ["bicubic", "learned_3d"]
        assert target_schema["required"]["handoff_transfer"][1]["default"] == "bicubic"
        assert target_schema["optional"]["learned_upscaler"] == ("H3_LATENT_UPSCALER",)
    assert "handoff_transfer" not in H3ProgressiveHandoff.INPUT_TYPES()["required"]


def test_target_sparse_node_is_explicitly_experimental_and_keeps_target_input_schema():
    from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff
    from h3_flow_regenerate.target_sparse_node import H3ProgressiveTargetSparseHandoff

    assert H3ProgressiveTargetSparseHandoff.INPUT_TYPES() == H3ProgressiveTargetInputHandoff.INPUT_TYPES()
    assert H3ProgressiveTargetSparseHandoff.CATEGORY.endswith("/experimental")
    assert "Exact Native Masked video prefixes stay on the target grid" in H3ProgressiveTargetSparseHandoff.DESCRIPTION


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
        return str(folder), "metrics", allocations, subfolder, filename_prefix

    monkeypatch.setitem(
        sys.modules,
        "folder_paths",
        SimpleNamespace(
            get_output_directory=lambda: str(tmp_path),
            get_save_image_path=get_save_image_path,
        ),
    )

    metrics = H3FlowMetrics()
    metrics.event("trajectory_commit", samples=8)
    output = H3MetricsJSON().render(metrics, "bench/metrics")

    saved = tmp_path / "bench" / "metrics_00001_.json"
    assert saved.exists()
    initial = saved.read_text(encoding="utf-8")
    assert '"trajectory_commit"' in initial
    assert '"guidance"' not in initial

    metrics.increment("transformer_actual_nfe", 7)
    metrics.event("guidance", correction_rms=0.125)
    assert '"guidance"' not in saved.read_text(encoding="utf-8")

    repeated = H3MetricsJSON().render(metrics, "bench/other-prefix")
    assert allocations == 1
    assert metrics.autosave_path == saved
    assert not (tmp_path / "bench" / "metrics_00002_.json").exists()
    assert repeated["ui"]["text"] == ["Saving metrics JSON: bench/metrics_00001_.json"]

    metrics.event("sampler_wall", elapsed_ms=123.0, failed=False, progressive=False)
    final = saved.read_text(encoding="utf-8")
    assert '"guidance"' in final
    assert '"correction_rms": 0.125' in final
    assert '"transformer_actual_nfe": 7' in final
    assert '"sampler_wall"' in final
    assert output["result"] != (metrics.to_json(),)
    assert output["ui"]["text"] == ["Saving metrics JSON: bench/metrics_00001_.json"]
