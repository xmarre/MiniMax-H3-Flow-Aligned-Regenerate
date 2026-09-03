import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace


def test_package_import_without_comfyui():
    import h3_flow_regenerate

    assert h3_flow_regenerate.H3FlowTrajectory.api_version == 1


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


def test_progressive_nodes_expose_all_selectable_guidance_controls():
    from h3_flow_regenerate.nodes import H3ProgressiveHandoff, H3ProgressiveTargetInputHandoff

    for node in (H3ProgressiveHandoff, H3ProgressiveTargetInputHandoff):
        required = node.INPUT_TYPES()["required"]
        assert {
            "guidance_mode",
            "direction_weight",
            "acceleration_weight",
            "consistency_weight",
            "low_frequency_cutoff",
        }.issubset(required)


def test_metrics_json_output_node_saves_unique_json_and_refreshes_after_sampler(monkeypatch, tmp_path):
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.nodes import H3MetricsJSON

    def get_save_image_path(filename_prefix, output_dir, image_width=0, image_height=0):
        del image_width, image_height
        subfolder = "bench"
        folder = tmp_path / subfolder
        folder.mkdir(parents=True, exist_ok=True)
        return str(folder), "metrics", 1, subfolder, filename_prefix

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

    metrics.event("sampler_wall", elapsed_ms=123.0, failed=False, progressive=False)
    final = saved.read_text(encoding="utf-8")
    assert '"guidance"' in final
    assert '"correction_rms": 0.125' in final
    assert '"transformer_actual_nfe": 7' in final
    assert '"sampler_wall"' in final
    assert output["result"] != (metrics.to_json(),)
    assert output["ui"]["text"] == ["Saving metrics JSON: bench/metrics_00001_.json"]
