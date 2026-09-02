import importlib.util
from pathlib import Path


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
