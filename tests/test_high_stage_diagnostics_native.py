from __future__ import annotations

import ast
import os
from pathlib import Path
from types import ModuleType

import pytest

from h3_flow_regenerate.high_stage_diagnostics import HIGH_STAGE_DIAGNOSTIC_KEY, make_high_stage_diagnostic_contract


@pytest.fixture(scope="module")
def native_copy_nested_dicts():
    root = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "comfy"))
    path = root / "comfy/patcher_extension.py"
    if not path.is_file():
        if os.environ.get("COMFYUI_ROOT"):
            raise FileNotFoundError(path)
        pytest.skip("native clone oracle runs in source-contract CI with COMFYUI_ROOT")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "copy_nested_dicts"]
    if len(nodes) != 1:
        raise AssertionError(f"pinned ComfyUI copy_nested_dicts changed: found {len(nodes)} definitions")
    module = ModuleType("native_copy_nested_dicts_oracle")
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module.copy_nested_dicts


def test_non_dict_diagnostic_holder_keeps_identity_through_pinned_comfy_clone(native_copy_nested_dicts):
    contract = make_high_stage_diagnostic_contract(
        prefix_t=1,
        shapes=[(1, 24, 3, 4, 4), (1, 32, 2, 3)],
        phases=((0, "single"),),
        sampler="sample_res_multistep",
    )
    holder = contract["holder"]
    options = {"transformer_options": {HIGH_STAGE_DIAGNOSTIC_KEY: contract}}
    cloned = native_copy_nested_dicts(options)
    cloned_contract = cloned["transformer_options"][HIGH_STAGE_DIAGNOSTIC_KEY]
    assert cloned_contract is not contract
    assert cloned_contract["holder"] is holder
