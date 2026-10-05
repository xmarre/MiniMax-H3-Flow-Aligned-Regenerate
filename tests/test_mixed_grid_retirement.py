from __future__ import annotations

import json
from pathlib import Path

import h3_flow_regenerate.nodes as nodes
import h3_flow_regenerate.target_sparse_node as target_sparse_node

ROOT = Path(__file__).resolve().parents[1]


def test_target_input_remains_the_standard_progressive_node():
    assert nodes.NODE_CLASS_MAPPINGS["H3ProgressiveTargetInputHandoff"] is nodes.H3ProgressiveTargetInputHandoff
    assert nodes.NODE_DISPLAY_NAME_MAPPINGS["H3ProgressiveTargetInputHandoff"] == (
        "MiniMax H3 Progressive Handoff (Target Input)"
    )


def test_mixed_grid_id_is_preserved_but_not_remapped():
    mixed = target_sparse_node.H3ProgressiveMixedGridHandoff
    assert target_sparse_node.NODE_CLASS_MAPPINGS["H3ProgressiveMixedGridHandoff"] is mixed
    assert mixed is not nodes.H3ProgressiveTargetInputHandoff
    assert mixed.EXACT_PREFIX_MODE == "mixed_grid_low_suffix"
    assert mixed.CATEGORY.endswith("/deprecated")
    assert target_sparse_node.NODE_DISPLAY_NAME_MAPPINGS["H3ProgressiveMixedGridHandoff"].endswith("[Deprecated]")
    assert "Deprecated compatibility path" in mixed.DESCRIPTION


def test_canonical_overlay_uses_partitioned_exact_prefix_and_retains_target_input_control():
    payload = json.loads((ROOT / "workflows" / "progressive-handoff.overlay.json").read_text(encoding="utf-8"))

    chain = payload["placement"]["chain"]
    assert "H3PartitionedExactPrefixDiagnosticHandoff" in chain
    assert "H3ProgressiveMixedGridHandoff" not in chain

    exact = payload["exact_prefix_contract"]
    assert exact["mode"] == "same_grid_target_control"
    assert exact["low_probe_high_video_grid"] == "target"
    assert exact["clean_and_residual_handoff"] == "identity"
    assert exact["audio"]["stored_overlap_ticks"] == 16
    assert exact["audio"]["effective_overlap_ticks"] == 0
    assert exact["video"]["stored_overlap_tokens"] == 6
    assert exact["video"]["effective_overlap_tokens"] == 0
    assert exact["final_exact_restore"] is True

    control = payload["generic_target_input_control"]
    assert control["node"] == "H3ProgressiveTargetInputHandoff"
    assert control["production_exact_prefix_recommended"] is False
    fallback = control["exact_prefix_fallback"]
    assert fallback["mode"] == "single_target_grid_sampler"
    assert fallback["learned_upscaler_calls"] == 0
    assert fallback["geometry_boundary"] is False
    assert fallback["audio_guided_overlap_ticks"] == 4


def test_overlay_marks_mixed_grid_as_compatibility_only_without_aliasing():
    payload = json.loads((ROOT / "workflows" / "progressive-handoff.overlay.json").read_text(encoding="utf-8"))
    retired = payload["deprecated_compatibility"]

    assert retired["node_id"] == "H3ProgressiveMixedGridHandoff"
    assert retired["status"] == "compatibility-only for one release"
    assert retired["remapped_to_target_input"] is False
    assert retired["production_recommended"] is False
    assert retired["release_gate"] is False
    assert retired["compatibility_render_required"] is False
