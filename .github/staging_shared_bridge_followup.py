from pathlib import Path

path = Path("tests/test_target_sparse.py")
text = path.read_text()
start_marker = "def test_target_sparse_runtime_splits_two_target_grid_lifetimes_without_resizing_exact_prefix():"
end_marker = "\n\ndef test_target_sparse_high_failure_invalidates_committed_low_trajectory():"
start = text.index(start_marker)
end = text.index(end_marker, start)
block = text[start:end]
old = '''        ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            handoff_coordinate=0.3,\n            exact_prefix_mode="target_sparse_lifter",\n        ),\n'''
new = '''        ProgressiveTargetInputConfig(\n            source_latent_h=4,\n            source_latent_w=4,\n            handoff_coordinate=0.3,\n            exact_prefix_mode="target_sparse_lifter",\n            suffix_dc_bridge=False,\n        ),\n'''
if block.count(old) != 1:
    raise RuntimeError(f"expected one target-sparse structural-test config, found {block.count(old)}")
block = block.replace(old, new, 1)
path.write_text(text[:start] + block + text[end:])
