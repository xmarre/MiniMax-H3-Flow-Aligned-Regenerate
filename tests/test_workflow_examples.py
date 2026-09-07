from __future__ import annotations

import json
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "workflows" / "examples"


def _load(name: str):
    with (EXAMPLE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def _assert_api_links_resolve(workflow: dict[str, dict]) -> None:
    node_ids = set(workflow)
    for node in workflow.values():
        for value in node.get("inputs", {}).values():
            if not (isinstance(value, list) and len(value) == 2 and isinstance(value[0], str)):
                continue
            assert value[0] in node_ids
            assert isinstance(value[1], int) and value[1] >= 0


def _assert_common_api_chain(workflow: dict[str, dict], patch_class: str) -> None:
    classes = {node["class_type"] for node in workflow.values()}
    assert {
        "UNETLoader",
        "CLIPLoader",
        "VAELoader",
        "MiniMaxH3ImageToVideo",
        "H3FlowTrajectory",
        patch_class,
        "RandomNoise",
        "KSamplerSelect",
        "BasicScheduler",
        "BasicGuider",
        "SamplerCustomAdvanced",
        "VAEDecode",
        "VAEDecodeAudio",
        "CreateVideo",
        "SaveVideo",
    } <= classes
    assert "LoraLoaderModelOnly" not in classes

    patch_id = next(node_id for node_id, node in workflow.items() if node["class_type"] == patch_class)
    scheduler = next(node for node in workflow.values() if node["class_type"] == "BasicScheduler")
    guider = next(node for node in workflow.values() if node["class_type"] == "BasicGuider")
    sampler = next(node for node in workflow.values() if node["class_type"] == "KSamplerSelect")

    assert scheduler["inputs"]["model"] == [patch_id, 0]
    assert guider["inputs"]["model"] == [patch_id, 0]
    assert sampler["inputs"]["sampler_name"] == "res_multistep"
    _assert_api_links_resolve(workflow)


def _assert_canvas_links_resolve(workflow: dict, patch_class: str) -> None:
    assert workflow["version"] == 0.4
    nodes = {node["id"]: node for node in workflow["nodes"]}
    assert workflow["last_node_id"] >= max(nodes)
    link_ids = {link[0] for link in workflow["links"]}
    assert workflow["last_link_id"] >= max(link_ids)
    assert len(link_ids) == len(workflow["links"])

    for link_id, origin_id, origin_slot, target_id, target_slot, link_type in workflow["links"]:
        origin = nodes[origin_id]
        target = nodes[target_id]
        assert 0 <= origin_slot < len(origin["outputs"])
        assert 0 <= target_slot < len(target["inputs"])
        output = origin["outputs"][origin_slot]
        input_ = target["inputs"][target_slot]
        assert link_id in (output.get("links") or [])
        assert input_["link"] == link_id
        assert output["type"] == link_type == input_["type"]

    for node in nodes.values():
        for input_ in node["inputs"]:
            if input_.get("link") is not None:
                assert input_["link"] in link_ids
        for output in node["outputs"]:
            for link_id in output.get("links") or []:
                assert link_id in link_ids

    classes = {node["type"] for node in nodes.values()}
    assert patch_class in classes
    assert "LoraLoaderModelOnly" not in classes
    patch_id = next(node_id for node_id, node in nodes.items() if node["type"] == patch_class)
    scheduler = next(node for node in nodes.values() if node["type"] == "BasicScheduler")
    guider = next(node for node in nodes.values() if node["type"] == "BasicGuider")
    sampler = next(node for node in nodes.values() if node["type"] == "KSamplerSelect")

    scheduler_link = next(link for link in workflow["links"] if link[0] == scheduler["inputs"][0]["link"])
    guider_link = next(link for link in workflow["links"] if link[0] == guider["inputs"][0]["link"])
    assert scheduler_link[1:3] == [patch_id, 0]
    assert guider_link[1:3] == [patch_id, 0]
    assert sampler["widgets_values"][0] == "res_multistep"


def test_progressive_target_input_api_example_is_complete():
    workflow = _load("progressive-target-input.api.json")
    _assert_common_api_chain(workflow, "H3ProgressiveTargetInputHandoff")

    patch = next(node for node in workflow.values() if node["class_type"] == "H3ProgressiveTargetInputHandoff")
    assert patch["inputs"]["source_mode"] == "scale"
    assert patch["inputs"]["source_scale"] == 0.7
    assert patch["inputs"]["handoff_transfer"] == "bicubic"


def test_progressive_source_input_api_example_is_complete():
    workflow = _load("progressive-source-input.api.json")
    _assert_common_api_chain(workflow, "H3ProgressiveHandoff")

    patch = next(node for node in workflow.values() if node["class_type"] == "H3ProgressiveHandoff")
    assert patch["inputs"]["target_mode"] == "scale"
    assert patch["inputs"]["scale"] == 1.2


def test_progressive_target_input_canvas_workflow_is_loadable_shape():
    workflow = _load("progressive-target-input.workflow.json")
    _assert_canvas_links_resolve(workflow, "H3ProgressiveTargetInputHandoff")
    patch = next(node for node in workflow["nodes"] if node["type"] == "H3ProgressiveTargetInputHandoff")
    assert patch["widgets_values"][0:2] == ["scale", 0.7]
    assert patch["widgets_values"][-1] == "bicubic"


def test_progressive_source_input_canvas_workflow_is_loadable_shape():
    workflow = _load("progressive-source-input.workflow.json")
    _assert_canvas_links_resolve(workflow, "H3ProgressiveHandoff")
    patch = next(node for node in workflow["nodes"] if node["type"] == "H3ProgressiveHandoff")
    assert patch["widgets_values"][0:2] == ["scale", 1.2]
