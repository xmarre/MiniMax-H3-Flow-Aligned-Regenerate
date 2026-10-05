#!/usr/bin/env python3
"""Compare historical Gaussian refinement and dense drift on a verified stage bundle.

This reconstructs only the exported generated tokens using the historical v1
coarse projection. It verifies that reconstruction against saved sampler input
before evaluating the alternative operator. It does not rerun H3 or a decoder.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from decode_native_boundary_evidence import load_bundle

from h3_flow_regenerate.handoff import (
    H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
    _patch_phase_group_mean,
    deterministic_video_noise,
    refine_h3_flow_residual,
    refine_h3_patch_lattice_residual,
)


def rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt())


def analyze_residual_transport(
    manifest: dict,
    values: dict[str, torch.Tensor],
    *,
    source_hw: tuple[int, int],
    noise_scale: float = 1.0,
    source_noise_dtype: torch.dtype = torch.float32,
) -> dict:
    window = manifest["metadata"]["window"]
    prefix, stop, temporal = window["prefix_t"], window["window_stop_t"], window["temporal"]
    offset = prefix - window["window_start_t"]
    sigma = float(manifest["sigma"])
    if not 0 < sigma < 1:
        raise ValueError("residual evidence requires a finite interior sigma")
    clean = values["pre_high_exact_restored"][:, :, offset:]
    saved = values["first_high_sampler_input"][:, :, offset:]
    residual = (saved - (1 - sigma) * clean) / sigma
    sh, sw = source_hw
    th, tw = residual.shape[-2:]
    if min(sh, sw) <= 0 or sh % 2 or sw % 2 or sh > th or sw > tw:
        raise ValueError("historical residual projection requires positive non-shrinking even geometry")
    recovered = torch.empty(*residual.shape[:-2], sh, sw)
    for py in range(2):
        for px in range(2):
            mean, cy, cx = _patch_phase_group_mean(residual[..., py::2, px::2], sh // 2, sw // 2)
            recovered[..., py::2, px::2] = mean * (cy[:, None] * cx[None, :]).sqrt()
    initial = deterministic_video_noise(
        (1, 24, temporal, sh, sw),
        seed=int(manifest["seed"]) + 0x48334C4F574C52,
        device=torch.device("cpu"),
        dtype=source_noise_dtype,
    ).float()
    full_residual = initial * noise_scale
    full_residual[:, :, prefix:stop] = recovered
    innovation_seed = int(manifest["seed"]) + 0x4833464C4F57
    old, _ = refine_h3_patch_lattice_residual(full_residual, target_h=th, target_w=tw, seed=innovation_seed)
    reconstruction_error = old[:, :, prefix:stop] - residual
    if rms(reconstruction_error) > 1e-5 or float(reconstruction_error.abs().max()) > 1e-4:
        raise ValueError("saved residual is not reproduced by the asserted historical v1 noise contract")
    candidate, receipt = refine_h3_flow_residual(
        full_residual, initial, target_h=th, target_w=tw, seed=innovation_seed, noise_scale=noise_scale
    )
    delta = candidate[:, :, prefix:stop] - old[:, :, prefix:stop]
    initial_window = initial[:, :, prefix:stop] * noise_scale
    drift = recovered - initial_window
    return {
        "policy": "offline_verified_residual_operator_comparison_v1",
        "historical_noise_policy": H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
        "recovered_source_residual_status": "inferred_from_saved_target_v1_coarse_projection",
        "independent_source_residual_bytes_available": False,
        "source_hw": list(source_hw),
        "target_hw": [th, tw],
        "generated_token_range": [prefix, stop],
        "rng_temporal_length": temporal,
        "source_noise_dtype": str(source_noise_dtype),
        "model_noise_scale": noise_scale,
        "historical_reconstruction_rms_error": rms(reconstruction_error),
        "historical_reconstruction_max_abs_error": float(reconstruction_error.abs().max()),
        "source_drift_rms": rms(drift),
        "source_residual_vs_initial_cosine": float(
            torch.nn.functional.cosine_similarity(recovered.flatten(), initial_window.flatten(), dim=0)
        ),
        "candidate_minus_historical_residual_rms": rms(delta),
        "candidate_minus_historical_state_rms": rms(delta * sigma),
        "candidate_receipt_full_temporal_scope": receipt,
        "unexported_source_tokens_used_for_comparison": False,
        "production_sampling_rerun": False,
        "rendered_acceptance": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--source-hw", type=int, nargs=2, required=True)
    parser.add_argument("--historical-policy", choices=[H3_HANDOFF_NOISE_SOURCE_RESIDUAL], required=True)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument("--source-noise-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest, values = load_bundle(args.bundle)
    report = analyze_residual_transport(
        manifest,
        values,
        source_hw=tuple(args.source_hw),
        noise_scale=args.noise_scale,
        source_noise_dtype=getattr(torch, args.source_noise_dtype),
    )
    report["manifest_sha256"] = hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest()
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
