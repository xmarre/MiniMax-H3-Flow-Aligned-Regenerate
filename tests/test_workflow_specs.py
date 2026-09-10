import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def _load(path):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_resolution_shift_overlay_targets_refine_not_continuum():
    overlay = _load("workflows/resolution-shift-only.overlay.json")

    assert overlay["refine"]["enabled"] is True
    assert overlay["refine"]["sigmas"] == "H3ResolutionAwareSigmas.sigmas -> refine.sigmas"
    assert "do not insert H3ResolutionAwareSigmas into Continuum" in overlay["continuum"]["sigmas"]
    assert overlay["target_geometry"]["outputs"] == {
        "target_width": "H3ResolutionAwareSigmas.target_width",
        "target_height": "H3ResolutionAwareSigmas.target_height",
    }
    assert "MiniMax H3 Latent Upscaler + Refine (3D) execution" not in overlay["model_chain"]["remove_or_bypass"]


def test_resolution_shift_matrix_preserves_base_and_refine():
    matrix = _load("workflows/benchmark-matrix.json")["resolution_shift_smoke"]

    assert matrix["topology"]["continuum"].startswith("unchanged")
    assert matrix["common"]["refine_settings"]["enabled"] is True
    assert matrix["common"]["refine_settings"]["scale"] == 1.2
    assert matrix["topology"]["forbidden_placement"] == "H3ResolutionAwareSigmas must not feed Continuum.sigmas"
    assert matrix["runs"][0]["id"] == "E0-refine-control"
    assert matrix["runs"][1]["id"] == "E1-refine-resolution-aware"


def test_progressive_overlay_defines_canonical_mixed_grid_defaults_and_preserves_history():
    overlay = _load("workflows/progressive-handoff.overlay.json")
    defaults = overlay["canonical_defaults"]
    provider = overlay["latent_upscaler_provider"]
    historical = overlay["historical_learned_transfer_ab"]

    assert overlay["schema_version"] == 4
    assert overlay["placement"]["chain"][-2] == "H3ProgressiveMixedGridHandoff"
    assert defaults == {
        "source_mode": "scale",
        "source_scale": 0.7,
        "source_width": 864,
        "source_height": 640,
        "handoff_coordinate": 0.35,
        "handoff_selection": "fixed",
        "guidance_mode": "direction+temporal",
        "direction_weight": 0.25,
        "acceleration_weight": 0.25,
        "consistency_weight": 0.25,
        "low_frequency_cutoff": 0.25,
        "temporal_weight": 0.2,
        "handoff_transfer": "learned_3d",
        "suffix_dc_bridge": True,
        "suffix_geometric_bridge": True,
        "weight_semantics": (
            "With guidance_mode=direction+temporal, acceleration_weight and consistency_weight are staged values "
            "only; apply_guidance does not use them unless the corresponding guidance mode is selected."
        ),
    }
    assert provider["node"] == "MinimaxH3LatentUpscaler3DProvider"
    assert provider["widgets"] == {
        "model_name": "minimax_h3_latent_upscaler_3d_bf16.safetensors",
        "device": "cuda",
        "precision": "bf16",
        "offload_after_upscale": False,
    }
    assert historical["control_widget"] == {"handoff_transfer": "bicubic"}
    assert historical["treatment_widget"] == {"handoff_transfer": "learned_3d"}
    assert historical["strict_d14_pair"]["only_intended_difference"] == "handoff_transfer"
    assert historical["strict_d14_pair"]["latent_transition"] == "46x46 -> 56x56"
