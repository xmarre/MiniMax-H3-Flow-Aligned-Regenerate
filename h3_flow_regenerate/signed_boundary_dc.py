"""Signed per-channel DC receipts across one exact-prefix continuation boundary.

Diagnostic only. Each receipt reduces an existing video latent to per-channel
spatial means of the native temporal tokens around the boundary and moves only
that small table to the CPU. No tensor reference is retained.

The ledger attributes the final boundary step to the stages that produce it.
With ``s`` the signed step ``mean_hw(token[p]) - mean_hw(token[p-1])`` of a
stage, the final step telescopes exactly:

    s_final = s_source + (s_provider - s_source) + (s_restored - s_provider)
              + (s_final - s_restored)

Each term's projection onto ``s_final / |s_final|`` is its signed contribution
to the final step's magnitude; the contributions sum to ``|s_final|``. Spatial
means are comparable across the source and target grids because a resize does
not define a different spatial mean. They are latent-domain quantities: the
linear preview luma below is an approximation through the latent format's RGB
preview factors, not a VAE decode.
"""

from __future__ import annotations

import math
from typing import Any

import torch

SIGNED_BOUNDARY_DC_POLICY = "signed_boundary_dc_v1"
SIGNED_BOUNDARY_DC_TOKENS_BEFORE = 3
SIGNED_BOUNDARY_DC_TOKENS_AFTER = 4
SIGNED_BOUNDARY_DC_STAGES = ("source_low_exact_context", "provider_native", "pre_high_exact_restored", "final")
_LUMA = (0.299, 0.587, 0.114)


def _preview_luma(means: torch.Tensor, factors: torch.Tensor | None) -> list[float] | None:
    if factors is None:
        return None
    rgb = means.transpose(0, 1) @ factors
    weights = torch.tensor(_LUMA, dtype=torch.float32)
    return [float(v) for v in (rgb @ weights).tolist()]


def latent_preview_factors(latent_format: Any, channels: int) -> torch.Tensor | None:
    """Return the latent format's ``[channels, 3]`` RGB preview factors, if it has them."""

    factors = getattr(latent_format, "latent_rgb_factors", None)
    if factors is None:
        return None
    try:
        table = torch.tensor(factors, dtype=torch.float32)
    except (TypeError, ValueError):
        return None
    if table.ndim != 2 or tuple(table.shape) != (int(channels), 3) or not bool(torch.isfinite(table).all()):
        return None
    return table


def measure_signed_boundary_dc(
    video: torch.Tensor,
    prefix_t: int,
    *,
    preview_factors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Signed per-channel spatial means of tokens ``p-3 .. p+3`` of batch item 0."""

    if not torch.is_tensor(video) or video.ndim != 5 or not video.is_floating_point():
        raise ValueError("signed boundary DC expects a floating-point BxCxTxHxW video")
    temporal = int(video.shape[2])
    prefix_t = int(prefix_t)
    if not 0 < prefix_t < temporal:
        raise ValueError("signed boundary DC requires a non-empty prefix shorter than the video")
    start = max(0, prefix_t - SIGNED_BOUNDARY_DC_TOKENS_BEFORE)
    stop = min(temporal, prefix_t + SIGNED_BOUNDARY_DC_TOKENS_AFTER)
    with torch.no_grad():
        means = video[0, :, start:stop].mean(dim=(-2, -1), dtype=torch.float32).cpu()
    channels = int(means.shape[0])
    if preview_factors is not None and tuple(preview_factors.shape) != (channels, 3):
        preview_factors = None
    boundary = prefix_t - start
    step = means[:, boundary] - means[:, boundary - 1]
    pair_steps = means[:, 1:] - means[:, :-1]
    luma = _preview_luma(means, preview_factors)
    return {
        "policy": SIGNED_BOUNDARY_DC_POLICY,
        "tokens": list(range(start, stop)),
        "prefix_t": prefix_t,
        "batch_index": 0,
        "batch_size": int(video.shape[0]),
        "grid_hw": [int(video.shape[-2]), int(video.shape[-1])],
        "channel_means": [[float(v) for v in means[:, k].tolist()] for k in range(stop - start)],
        "boundary_step": [float(v) for v in step.tolist()],
        "boundary_step_norm": float(step.norm().item()),
        "boundary_step_rms": float(step.square().mean().sqrt().item()),
        "pair_step_rms": [float(v) for v in pair_steps.square().mean(dim=0).sqrt().tolist()],
        "preview_luma": luma,
        "preview_luma_boundary_step": (luma[boundary] - luma[boundary - 1]) if luma is not None else None,
        "preview": "linear_latent_rgb_preview_factors" if luma is not None else "unavailable",
        "output_mutated": False,
        "extra_h3_nfe": 0,
        "extra_vae_calls": 0,
    }


def signed_boundary_dc_ledger(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attribute the final boundary step to source, provider, exact restore and high."""

    missing = [stage for stage in SIGNED_BOUNDARY_DC_STAGES if stage not in receipts]
    if missing:
        return {"policy": SIGNED_BOUNDARY_DC_POLICY, "status": "incomplete", "missing_stages": missing}
    steps = {stage: torch.tensor(receipts[stage]["boundary_step"], dtype=torch.float64) for stage in receipts}
    if len({int(step.numel()) for step in steps.values()}) != 1:
        return {"policy": SIGNED_BOUNDARY_DC_POLICY, "status": "channel_mismatch"}
    source, provider, restored, final = (steps[stage] for stage in SIGNED_BOUNDARY_DC_STAGES)
    components = {
        "source_low": source,
        "provider_transfer": provider - source,
        "exact_prefix_restore": restored - provider,
        "target_high": final - restored,
    }
    final_norm = float(final.norm().item())
    direction = final / final_norm if final_norm > 0.0 else None
    luma = {stage: receipts[stage].get("preview_luma_boundary_step") for stage in SIGNED_BOUNDARY_DC_STAGES}
    luma_components = None
    if all(value is not None for value in luma.values()):
        luma_components = {
            "source_low": luma["source_low_exact_context"],
            "provider_transfer": luma["provider_native"] - luma["source_low_exact_context"],
            "exact_prefix_restore": luma["pre_high_exact_restored"] - luma["provider_native"],
            "target_high": luma["final"] - luma["pre_high_exact_restored"],
        }
    rows = {}
    for name, vector in components.items():
        rows[name] = {
            "rms": float(vector.square().mean().sqrt().item()),
            "signed_contribution_to_final_norm": (
                float(torch.dot(vector, direction).item()) if direction is not None else None
            ),
            "preview_luma_step": luma_components[name] if luma_components is not None else None,
        }
    return {
        "policy": SIGNED_BOUNDARY_DC_POLICY,
        "status": "complete",
        "stages": list(SIGNED_BOUNDARY_DC_STAGES),
        "domains": {stage: receipts[stage].get("domain") for stage in SIGNED_BOUNDARY_DC_STAGES},
        "final_step_norm": final_norm,
        "final_step_rms": final_norm / math.sqrt(max(1, int(final.numel()))),
        "components": rows,
        "attribution": "telescoping_signed_projection_on_final_step",
        "output_mutated": False,
        "extra_h3_nfe": 0,
        "extra_vae_calls": 0,
    }


__all__ = [
    "SIGNED_BOUNDARY_DC_POLICY",
    "SIGNED_BOUNDARY_DC_STAGES",
    "latent_preview_factors",
    "measure_signed_boundary_dc",
    "signed_boundary_dc_ledger",
]
