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


def test_progressive_overlay_defaults_to_bicubic_and_defines_strict_learned_ab():
    overlay = _load("workflows/progressive-handoff.overlay.json")
    widgets = overlay["recommended_quality_operating_point"]["widgets"]
    experiment = overlay["learned_transfer_experiment"]

    assert overlay["schema_version"] == 4
    assert widgets["handoff_transfer"] == "bicubic"
    assert experiment["provider_node"] == "MinimaxH3LatentUpscaler3DProvider"
    assert experiment["control_widget"] == {"handoff_transfer": "bicubic"}
    assert experiment["treatment_widget"] == {"handoff_transfer": "learned_3d"}
    assert experiment["strict_d14_pair"]["only_intended_difference"] == "handoff_transfer"
    assert experiment["strict_d14_pair"]["latent_transition"] == "46x46 -> 56x56"
