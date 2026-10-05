import runpy
import sys
from pathlib import Path

import pytest
import torch

from h3_flow_regenerate.handoff import deterministic_video_noise, refine_h3_patch_lattice_residual

sys.path.insert(0, str(Path(__file__).parents[1] / "tools"))
REPLAY = runpy.run_path(str(Path(__file__).parents[1] / "tools/analyze_handoff_residual_evidence.py"))


def fixture():
    seed, sigma, temporal = 1144, 0.8, 22
    noise = deterministic_video_noise(
        (1, 24, temporal, 8, 8), seed=seed + 0x48334C4F574C52, device=torch.device("cpu"), dtype=torch.float32
    )
    residual = noise.clone()
    residual[:, :, 12:17, 4, :] += 0.25
    target, _ = refine_h3_patch_lattice_residual(residual, target_h=12, target_w=12, seed=seed + 0x4833464C4F57)
    clean = torch.ones(1, 24, 7, 12, 12)
    manifest = {
        "seed": seed,
        "sigma": sigma,
        "metadata": {"window": {"prefix_t": 12, "window_start_t": 10, "window_stop_t": 17, "temporal": temporal}},
    }
    values = {
        "pre_high_exact_restored": clean,
        "first_high_sampler_input": (1 - sigma) * clean + sigma * target[:, :, 10:17],
    }
    return manifest, values


def test_saved_sampler_state_reconstructs_historical_residual_before_counterfactual():
    manifest, values = fixture()
    report = REPLAY["analyze_residual_transport"](manifest, values, source_hw=(8, 8))
    assert report["historical_reconstruction_max_abs_error"] < 2e-6
    assert report["source_drift_rms"] == pytest.approx(0.25 / 8**0.5, abs=1e-6)
    assert report["candidate_minus_historical_residual_rms"] > 0.02
    assert report["rng_temporal_length"] == 22
    assert report["production_sampling_rerun"] is False
    assert report["rendered_acceptance"] is False


@pytest.mark.parametrize("corruption", ["input", "seed", "geometry", "temporal"])
def test_inconsistent_contract_fails_before_reporting_a_counterfactual(corruption):
    manifest, values = fixture()
    shape = (8, 8)
    if corruption == "input":
        values["first_high_sampler_input"][:, :, 2:, 2, 2] += 0.01
    elif corruption == "seed":
        manifest["seed"] += 1
    elif corruption == "geometry":
        shape = (6, 8)
    else:
        manifest["metadata"]["window"]["temporal"] = 23
    with pytest.raises(ValueError, match="not reproduced"):
        REPLAY["analyze_residual_transport"](manifest, values, source_hw=shape)


def test_coarse_reconstruction_is_not_independent_evidence_of_source_values():
    manifest, values = fixture()
    # A valid coarse-mode change is indistinguishable from a different source
    # residual without saved source bytes. It must remain explicitly inferred.
    values["first_high_sampler_input"][:, :, 2:, 4, 4] += 0.01
    report = REPLAY["analyze_residual_transport"](manifest, values, source_hw=(8, 8))
    assert report["independent_source_residual_bytes_available"] is False
    assert report["recovered_source_residual_status"] == "inferred_from_saved_target_v1_coarse_projection"
