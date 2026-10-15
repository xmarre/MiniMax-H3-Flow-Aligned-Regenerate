"""Signed per-channel DC receipts across one exact-prefix continuation boundary.

Diagnostic only. Each receipt reduces an existing video latent to per-channel
spatial means of the native temporal tokens around the boundary and moves only
that small table to the CPU. No tensor reference is retained.

The ledger attributes the final boundary step to the stages that produce it.
With ``s_k`` the signed step ``mean_hw(token[p]) - mean_hw(token[p-1])`` of the
k-th recorded stage, in pipeline order, the final step telescopes exactly:

    s_final = s_0 + (s_1 - s_0) + ... + (s_final - s_{n-1})

Each term is the change that one stage introduces. Its projection onto
``s_final / |s_final|`` is its signed contribution to the final step's
magnitude; the contributions sum to ``|s_final|``. Spatial means are comparable
across the source and target grids because a resize does not define a different
spatial mean. They are latent-domain quantities: the linear preview luma is an
approximation through the latent format's RGB preview factors, not a VAE decode.
"""

from __future__ import annotations

import math
import time
from typing import Any

import torch

SIGNED_BOUNDARY_DC_POLICY = "signed_boundary_dc_v1"
SIGNED_BOUNDARY_DC_TOKENS_BEFORE = 3
SIGNED_BOUNDARY_DC_TOKENS_AFTER = 4
# Pipeline order. A stage maps to the name of the change it introduces relative
# to the previous recorded stage. Optional stages are recorded only when the
# configuration performs them.
SIGNED_BOUNDARY_DC_STAGES = (
    ("source_low_native", "source_low"),
    ("source_low_exact_context", "source_prefix_write"),
    ("provider_native", "provider_transfer"),
    ("provider_boundary_pair_replaced", "provider_boundary_pair_replacement"),
    ("pre_high_exact_restored", "pre_high_correction_and_exact_restore"),
    ("final", "target_high"),
)
SIGNED_BOUNDARY_DC_OPTIONAL_STAGES = frozenset({"provider_boundary_pair_replaced"})
_LUMA = (0.299, 0.587, 0.114)


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


def signed_token_means(video: torch.Tensor, start: int, stop: int) -> torch.Tensor:
    """Per-channel float32 spatial means ``[C, stop - start]`` of batch item 0, on the CPU."""

    if not torch.is_tensor(video) or video.ndim != 5 or not video.is_floating_point():
        raise ValueError("signed token means expect a floating-point BxCxTxHxW video")
    if not 0 <= int(start) < int(stop) <= int(video.shape[2]):
        raise ValueError("signed token means require a non-empty in-range token span")
    with torch.no_grad():
        return video[0, :, int(start) : int(stop)].mean(dim=(-2, -1), dtype=torch.float32).cpu()


def _preview_luma(means: torch.Tensor, factors: torch.Tensor | None) -> list[float] | None:
    if factors is None or tuple(factors.shape) != (int(means.shape[0]), 3):
        return None
    rgb = means.transpose(0, 1) @ factors
    return [float(v) for v in (rgb @ torch.tensor(_LUMA, dtype=torch.float32)).tolist()]


def measure_signed_boundary_dc(
    video: torch.Tensor,
    prefix_t: int,
    *,
    preview_factors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Signed per-channel spatial means of tokens ``p-3 .. p+3`` of batch item 0."""

    started = time.perf_counter()
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("signed boundary DC expects a BxCxTxHxW video")
    temporal = int(video.shape[2])
    prefix_t = int(prefix_t)
    if not 0 < prefix_t < temporal:
        raise ValueError("signed boundary DC requires a non-empty prefix shorter than the video")
    start = max(0, prefix_t - SIGNED_BOUNDARY_DC_TOKENS_BEFORE)
    stop = min(temporal, prefix_t + SIGNED_BOUNDARY_DC_TOKENS_AFTER)
    means = signed_token_means(video, start, stop)
    boundary = prefix_t - start
    step = means[:, boundary] - means[:, boundary - 1]
    pair_steps = means[:, 1:] - means[:, :-1]
    luma = _preview_luma(means, preview_factors)
    return {
        "policy": SIGNED_BOUNDARY_DC_POLICY,
        "status": "measured",
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
        "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        "output_mutated": False,
        "extra_h3_nfe": 0,
        "extra_vae_calls": 0,
    }


def measure_signed_prefix_dc(
    carried: torch.Tensor,
    reconciled: torch.Tensor,
    *,
    tokens: int = SIGNED_BOUNDARY_DC_TOKENS_BEFORE,
) -> dict[str, Any]:
    """Signed per-channel DC that prefix reconciliation adds to the last carried tokens."""

    if not torch.is_tensor(carried) or not torch.is_tensor(reconciled) or carried.shape != reconciled.shape:
        raise ValueError("signed prefix DC requires paired prefixes")
    temporal = int(carried.shape[2])
    start = max(0, temporal - int(tokens))
    before = signed_token_means(carried, start, temporal)
    after = signed_token_means(reconciled, start, temporal)
    delta = after - before
    return {
        "policy": SIGNED_BOUNDARY_DC_POLICY,
        "status": "measured",
        "tokens": list(range(start, temporal)),
        "carried_channel_means": [[float(v) for v in before[:, k].tolist()] for k in range(temporal - start)],
        "reconciled_minus_carried": [[float(v) for v in delta[:, k].tolist()] for k in range(temporal - start)],
        "reconciled_minus_carried_rms": [float(v) for v in delta.square().mean(dim=0).sqrt().tolist()],
        "output_mutated": False,
    }


def signed_boundary_dc_ledger(receipts: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Attribute the final boundary step to the recorded pipeline stages."""

    order = [(stage, change) for stage, change in SIGNED_BOUNDARY_DC_STAGES if stage in receipts]
    missing = [
        stage
        for stage, _change in SIGNED_BOUNDARY_DC_STAGES
        if stage not in receipts and stage not in SIGNED_BOUNDARY_DC_OPTIONAL_STAGES
    ]
    failed = [stage for stage, _change in order if receipts[stage].get("status") != "measured"]
    if missing or failed:
        return {
            "policy": SIGNED_BOUNDARY_DC_POLICY,
            "status": "incomplete",
            "missing_stages": missing,
            "failed_stages": failed,
        }
    steps = [torch.tensor(receipts[stage]["boundary_step"], dtype=torch.float64) for stage, _change in order]
    if len({int(step.numel()) for step in steps}) != 1:
        return {"policy": SIGNED_BOUNDARY_DC_POLICY, "status": "channel_mismatch"}
    final = steps[-1]
    final_norm = float(final.norm().item())
    direction = final / final_norm if final_norm > 0.0 else None
    lumas = [receipts[stage].get("preview_luma_boundary_step") for stage, _change in order]
    components = {}
    previous_step = torch.zeros_like(final)
    previous_luma = 0.0
    for (stage, change), step, luma in zip(order, steps, lumas, strict=True):
        if stage == "source_low_exact_context" and receipts[stage].get("shadow_clean_selected"):
            change = "source_prefix_write_and_shadow_clean_selection"
        vector = step - previous_step
        norm = float(vector.norm().item())
        components[change] = {
            "stage": stage,
            "rms": norm / math.sqrt(max(1, int(vector.numel()))),
            "norm": norm,
            "signed_contribution_to_final_norm": (
                float(torch.dot(vector, direction).item()) if direction is not None else None
            ),
            "cosine_to_final": (
                float(torch.dot(vector, direction).item()) / norm if direction is not None and norm > 0.0 else None
            ),
            "preview_luma_step": (luma - previous_luma if luma is not None and previous_luma is not None else None),
        }
        previous_step = step
        previous_luma = luma
    component_norm_sum = sum(row["norm"] for row in components.values())
    return {
        "policy": SIGNED_BOUNDARY_DC_POLICY,
        "status": "complete",
        "stages": [stage for stage, _change in order],
        "domains": {stage: receipts[stage].get("domain") for stage, _change in order},
        "final_step_norm": final_norm,
        "final_step_rms": final_norm / math.sqrt(max(1, int(final.numel()))),
        "components": components,
        "component_norm_sum_over_final_norm": component_norm_sum / final_norm if final_norm > 0.0 else None,
        "attribution": "telescoping_signed_projection_on_final_step",
        "output_mutated": False,
        "extra_h3_nfe": 0,
        "extra_vae_calls": 0,
    }


__all__ = [
    "SIGNED_BOUNDARY_DC_OPTIONAL_STAGES",
    "SIGNED_BOUNDARY_DC_POLICY",
    "SIGNED_BOUNDARY_DC_STAGES",
    "latent_preview_factors",
    "measure_signed_boundary_dc",
    "measure_signed_prefix_dc",
    "signed_boundary_dc_ledger",
    "signed_token_means",
]
