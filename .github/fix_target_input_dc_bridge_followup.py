from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    p = Path(path)
    text = p.read_text()
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one match, found {count}: {old[:100]!r}")
    p.write_text(text.replace(old, new, 1))


# Internal config defaults to bridge-off so the generic Target Input path cannot
# acquire Continuum behavior implicitly. Tests that directly construct a
# Mixed-Grid config must therefore opt into the node's default-on behavior.
replace_once(
    "tests/test_mixed_grid_seam_diagnostics.py",
    '''        transfer_mode="learned_3d",\n        learned_upscaler=provider,\n    )\n\n    def execute(noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):\n''',
    '''        transfer_mode="learned_3d",\n        learned_upscaler=provider,\n        suffix_dc_bridge=True,\n    )\n\n    def execute(noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):\n''',
)

# Target-Sparse intentionally differs from generic Target Input by one
# Continuum-only control. Preserve every shared input contract while asserting
# that the generic node itself does not expose the bridge.
replace_once(
    "tests/test_packaging.py",
    '''def test_target_sparse_node_is_explicitly_experimental_and_keeps_target_input_schema():\n    from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff\n    from h3_flow_regenerate.target_sparse_node import H3ProgressiveTargetSparseHandoff\n\n    assert H3ProgressiveTargetSparseHandoff.INPUT_TYPES() == H3ProgressiveTargetInputHandoff.INPUT_TYPES()\n    assert H3ProgressiveTargetSparseHandoff.CATEGORY.endswith("/experimental")\n    assert "Exact Native Masked video prefixes stay on the target grid" in H3ProgressiveTargetSparseHandoff.DESCRIPTION\n''',
    '''def test_target_sparse_node_is_explicitly_experimental_and_only_adds_continuum_bridge_schema():\n    from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff\n    from h3_flow_regenerate.target_sparse_node import H3ProgressiveTargetSparseHandoff\n\n    target_schema = H3ProgressiveTargetInputHandoff.INPUT_TYPES()\n    sparse_schema = H3ProgressiveTargetSparseHandoff.INPUT_TYPES()\n    assert "suffix_dc_bridge" not in target_schema["required"]\n    sparse_required = sparse_schema["required"].copy()\n    bridge = sparse_required.pop("suffix_dc_bridge")\n    assert bridge[0] == "BOOLEAN"\n    assert bridge[1]["default"] is True\n    assert sparse_required == target_schema["required"]\n    assert sparse_schema["optional"] == target_schema["optional"]\n    assert H3ProgressiveTargetSparseHandoff.CATEGORY.endswith("/experimental")\n    assert "Exact Native Masked video prefixes stay on the target grid" in H3ProgressiveTargetSparseHandoff.DESCRIPTION\n''',
)

print("Updated tests for generic Target Input / Continuum-specific bridge separation")
