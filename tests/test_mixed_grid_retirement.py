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


def test_canonical_overlay_uses_target_input_and_records_exact_prefix_fallback():
    payload = json.loads((ROOT / "workflows" / "progressive-handoff.overlay.json").read_text(encoding="utf-8"))

    chain = payload["placement"]["chain"]
    assert "H3ProgressiveTargetInputHandoff" in chain
    assert "H3ProgressiveMixedGridHandoff" not in chain

    exact = payload["exact_prefix_contract"]
    assert exact["mode"] == "conservative target-grid fallback"
    assert exact["private_low_grid_sampler"] is False
    assert exact["handoff_probe"] is False
    assert exact["learned_upscaler_calls"] == 0
    assert exact["geometry_boundary"] is False
    assert exact["history_boundary"] is False
    assert exact["video_mask_modified_during_sampling"] is False
    assert exact["audio_guided_overlap"]["default_ticks"] == 4
    assert exact["audio_guided_overlap"]["ramp"] == [0.203125, 0.40234375, 0.6015625, 0.80078125]
    assert exact["audio_guided_overlap"]["final_exact_restore"] is True


def test_overlay_marks_mixed_grid_as_compatibility_only_without_aliasing():
    payload = json.loads((ROOT / "workflows" / "progressive-handoff.overlay.json").read_text(encoding="utf-8"))
    retired = payload["deprecated_compatibility"]

    assert retired["node_id"] == "H3ProgressiveMixedGridHandoff"
    assert retired["status"] == "compatibility-only for one release"
    assert retired["remapped_to_target_input"] is False
    assert retired["production_recommended"] is False
    assert retired["release_gate"] is False
    assert retired["compatibility_render_required"] is False
