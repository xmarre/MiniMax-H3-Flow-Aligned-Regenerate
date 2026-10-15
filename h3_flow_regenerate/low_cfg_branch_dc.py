"""Per-evaluation signed boundary DC of the low stage's CFG branches.

Diagnostic only. While the low and probe sampler lifetimes run, a
``sampler_post_cfg_function`` observer reduces the conditional, unconditional and
guided clean predictions, and the sampler input, to per-channel spatial means of
the last protected token and the first two generated tokens. It returns the
guided prediction unchanged and retains no tensor.

This distinguishes a tone that the conditional branch selects against the
protected context (prompt or conditioning driven, amplified by CFG) from one
that both branches select (the context is not followed), and shows at which
sigma the boundary step appears. Guiders that do not call ComfyUI's CFG
function leave the receipt empty.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch

from .geometry import unpack_streams
from .signed_boundary_dc import SIGNED_BOUNDARY_DC_POLICY, latent_preview_factors

LOW_CFG_BRANCH_DC_POLICY = "low_cfg_branch_boundary_dc_v1"
_BRANCHES = ("input", "cond", "uncond", "guided")
_LUMA = (0.299, 0.587, 0.114)


def _sigma(value: Any) -> float | None:
    if torch.is_tensor(value):
        return float(value.detach().flatten()[0].item()) if value.numel() else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def measure_cfg_branch_dc(
    args: dict[str, Any],
    shapes: list[tuple[int, ...]],
    prefix_t: int,
    *,
    preview_factors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Reduce one CFG evaluation's branches to tokens ``p-1 .. p+1`` of batch item 0."""

    tensors = {
        "input": args.get("input"),
        "cond": args.get("cond_denoised"),
        "uncond": args.get("uncond_denoised"),
        "guided": args.get("denoised"),
    }
    present = [name for name in _BRANCHES if torch.is_tensor(tensors[name])]
    start, stop = int(prefix_t) - 1, int(prefix_t) + 2
    with torch.no_grad():
        reductions = []
        for name in present:
            video = unpack_streams(tensors[name], shapes)[0]
            if not 0 <= start < stop <= int(video.shape[2]):
                raise ValueError("low CFG branch DC requires the boundary tokens")
            reductions.append(video[0, :, start:stop].mean(dim=(-2, -1), dtype=torch.float32))
        table = torch.stack(reductions).cpu() if reductions else None
    luma = torch.tensor(_LUMA, dtype=torch.float32)
    row: dict[str, Any] = {
        "sigma": _sigma(args.get("sigma")),
        "cond_scale": _sigma(args.get("cond_scale")),
        "tokens": list(range(start, stop)),
    }
    for name in _BRANCHES:
        if name not in present:
            row[name] = None
            continue
        means = table[present.index(name)]
        step = means[:, 1] - means[:, 0]
        entry = {
            "channel_means": [[float(v) for v in means[:, k].tolist()] for k in range(stop - start)],
            "boundary_step": [float(v) for v in step.tolist()],
            "boundary_step_rms": float(step.square().mean().sqrt().item()),
        }
        if preview_factors is not None and tuple(preview_factors.shape) == (int(means.shape[0]), 3):
            preview = (means.transpose(0, 1) @ preview_factors) @ luma
            entry["preview_luma"] = [float(v) for v in preview.tolist()]
            entry["preview_luma_boundary_step"] = float(preview[1] - preview[0])
        row[name] = entry
    return row


@contextmanager
def capture_low_cfg_branch_dc(
    model_options: dict[str, Any],
    shapes: list[tuple[int, ...]],
    prefix_t: int,
    *,
    stage: str,
    rows: list[dict[str, Any]],
    latent_format: Any = None,
):
    """Observe every CFG evaluation in this block; the guided prediction is returned unchanged."""

    previous = model_options.get("sampler_post_cfg_function")
    factors = latent_preview_factors(latent_format, int(shapes[0][1]))

    def observe(args):
        started = time.perf_counter()
        try:
            row = measure_cfg_branch_dc(args, shapes, prefix_t, preview_factors=factors)
            row["status"] = "measured"
        except Exception as exc:  # diagnostics must never stop sampling
            row = {"status": "error", "error": repr(exc)}
        row.update(stage=stage, evaluation=len(rows), elapsed_ms=(time.perf_counter() - started) * 1000.0)
        rows.append(row)
        return args["denoised"]

    model_options["sampler_post_cfg_function"] = [*(previous or ()), observe]
    try:
        yield
    finally:
        if previous is None:
            model_options.pop("sampler_post_cfg_function", None)
        else:
            model_options["sampler_post_cfg_function"] = previous


def low_cfg_branch_dc_receipt(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize the observed evaluations, first and last, per branch."""

    measured = [row for row in rows if row.get("status") == "measured"]
    summary = {}
    for name in _BRANCHES:
        steps = [
            (row["stage"], row["sigma"], row[name].get("preview_luma_boundary_step"), row[name]["boundary_step_rms"])
            for row in measured
            if row.get(name) is not None
        ]
        summary[name] = (
            {
                "first": {
                    "stage": steps[0][0],
                    "sigma": steps[0][1],
                    "preview_luma_step": steps[0][2],
                    "rms": steps[0][3],
                },
                "last": {
                    "stage": steps[-1][0],
                    "sigma": steps[-1][1],
                    "preview_luma_step": steps[-1][2],
                    "rms": steps[-1][3],
                },
            }
            if steps
            else None
        )
    return {
        "policy": LOW_CFG_BRANCH_DC_POLICY,
        "boundary_dc_policy": SIGNED_BOUNDARY_DC_POLICY,
        "status": "measured" if measured else ("not_observed" if not rows else "error"),
        "evaluations": len(rows),
        "failed_evaluations": len(rows) - len(measured),
        "summary": summary,
        "rows": rows,
        "output_mutated": False,
        "extra_h3_nfe": 0,
        "extra_vae_calls": 0,
    }


__all__ = [
    "LOW_CFG_BRANCH_DC_POLICY",
    "capture_low_cfg_branch_dc",
    "low_cfg_branch_dc_receipt",
    "measure_cfg_branch_dc",
]
