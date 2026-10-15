"""Per-evaluation signed boundary DC of the low stage's CFG branches.

Diagnostic only. While the low and probe sampler lifetimes run, a
``sampler_post_cfg_function`` observer reduces the conditional, unconditional and
guided clean predictions, and the sampler input, to per-channel spatial means of
the last protected token ``p-1`` and the first two generated tokens. It returns
the guided prediction unchanged and retains no tensor.

Core's inpaint wrapper discards every prediction for the protected rows after
CFG and returns the protected reference instead. The causal step of a branch is
therefore its generated token ``p`` against that protected reference token
``p-1`` (``anchored_step``), not against the branch's own discarded prediction
of ``p-1``. The step within the prediction (``predicted_prefix_step``) is kept
as a separate, discarded-branch diagnostic.

The unconditional tensor may be an accumulator rather than a model evaluation.
The receipt distinguishes definitely skipped requests from unverified candidate
branches; it never infers a physical H3 call from tensor presence.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any

import torch

from .geometry import unpack_streams
from .signed_boundary_dc import SIGNED_BOUNDARY_DC_POLICY, latent_preview_factors

LOW_CFG_BRANCH_DC_POLICY = "low_cfg_branch_boundary_dc_v3"
_BRANCHES = ("input", "cond", "uncond", "guided")
_LUMA = (0.299, 0.587, 0.114)


def _scalar(value: Any) -> float | None:
    if torch.is_tensor(value):
        return float(value.detach().flatten()[0].item()) if value.numel() else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _luma(means: torch.Tensor, factors: torch.Tensor | None) -> torch.Tensor | None:
    if factors is None or tuple(factors.shape) != (int(means.shape[0]), 3):
        return None
    return (means.transpose(0, 1) @ factors) @ torch.tensor(_LUMA, dtype=torch.float32)


def _unconditional_provenance(args: dict[str, Any]) -> tuple[bool, str]:
    """Whether a possibly real unconditional prediction is present, not whether H3 ran.

    Core may supply an all-zero accumulator when no effective uncond item ran.
    The CFG hook has no proof of physical branch execution. Reject the cases
    provably skipped and label all other candidate tensors unverified.
    """

    conditions = args.get("uncond")
    if conditions is None:
        return False, "not_requested"
    if isinstance(conditions, (list, tuple)):
        if not conditions:
            return False, "empty_conditioning"
        sigma = _scalar(args.get("sigma"))
        if sigma is not None:
            potentially_active = False
            for cond in conditions:
                if not isinstance(cond, dict) or "default" in cond:
                    potentially_active = True
                    break
                try:
                    if "timestep_start" in cond and sigma > float(cond["timestep_start"]):
                        continue
                    if "timestep_end" in cond and sigma < float(cond["timestep_end"]):
                        continue
                except (TypeError, ValueError):
                    potentially_active = True
                    break
                potentially_active = True
                break
            if not potentially_active:
                return False, "out_of_timestep_range"
    if not torch.is_tensor(args.get("uncond_denoised")):
        return False, "prediction_tensor_missing"
    return True, "candidate_tensor_execution_unverified"


def protected_reference_means(reference_video: torch.Tensor, prefix_t: int) -> torch.Tensor:
    """Per-channel float32 spatial means of the protected token ``p-1`` of batch item 0, on the CPU."""

    if not torch.is_tensor(reference_video) or reference_video.ndim != 5:
        raise ValueError("low CFG branch DC requires a BxCxTxHxW protected reference")
    if not 0 < int(prefix_t) <= int(reference_video.shape[2]):
        raise ValueError("low CFG branch DC requires the protected boundary token")
    with torch.no_grad():
        return reference_video[0, :, int(prefix_t) - 1].mean(dim=(-2, -1), dtype=torch.float32).cpu()


def measure_cfg_branch_dc(
    args: dict[str, Any],
    shapes: list[tuple[int, ...]],
    prefix_t: int,
    reference: torch.Tensor,
    *,
    preview_factors: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Reduce one CFG evaluation's branches to tokens ``p-1 .. p+1`` of batch item 0."""

    uncond_candidate, uncond_reason = _unconditional_provenance(args)
    tensors = {
        "input": args.get("input"),
        "cond": args.get("cond_denoised"),
        "uncond": args.get("uncond_denoised") if uncond_candidate else None,
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
    reference = reference.to(torch.float32)
    reference_luma = _luma(reference.view(-1, 1), preview_factors)
    row: dict[str, Any] = {
        "sigma": _scalar(args.get("sigma")),
        "cond_scale": _scalar(args.get("cond_scale")),
        "tokens": list(range(start, stop)),
        "uncond_requested": args.get("uncond") is not None,
        "uncond_candidate": uncond_candidate,
        "uncond_evaluated": None if uncond_candidate else False,
        "uncond_provenance": uncond_reason,
    }
    for name in _BRANCHES:
        if name not in present:
            row[name] = None
            continue
        means = table[present.index(name)]
        anchored = means[:, 1] - reference
        predicted = means[:, 1] - means[:, 0]
        entry = {
            "channel_means": [[float(v) for v in means[:, k].tolist()] for k in range(stop - start)],
            "anchored_step": [float(v) for v in anchored.tolist()],
            "anchored_step_rms": float(anchored.square().mean().sqrt().item()),
            "predicted_prefix_step_rms": float(predicted.square().mean().sqrt().item()),
        }
        luma = _luma(means, preview_factors)
        if luma is not None and reference_luma is not None:
            entry["preview_luma"] = [float(v) for v in luma.tolist()]
            entry["preview_luma_anchored_step"] = float(luma[1] - reference_luma[0])
            entry["preview_luma_predicted_prefix_step"] = float(luma[1] - luma[0])
        row[name] = entry
    return row


@contextmanager
def capture_low_cfg_branch_dc(
    model_options: dict[str, Any],
    shapes: list[tuple[int, ...]],
    prefix_t: int,
    reference_video: torch.Tensor,
    *,
    stage: str,
    rows: list[dict[str, Any]],
    latent_format: Any = None,
):
    """Observe every CFG evaluation in this block; the guided prediction is returned unchanged.

    ``reference_video`` is the sampler's ``latent_image`` video, whose protected
    rows Core's inpaint wrapper returns in place of every prediction.
    """

    previous = model_options.get("sampler_post_cfg_function")
    factors = latent_preview_factors(latent_format, int(shapes[0][1]))
    try:
        reference = protected_reference_means(reference_video, prefix_t)
        reference_error = None
    except Exception as exc:  # diagnostics must never stop sampling
        reference, reference_error = None, repr(exc)

    def observe(args):
        started = time.perf_counter()
        try:
            if reference is None:
                raise RuntimeError(f"protected reference unavailable: {reference_error}")
            row = measure_cfg_branch_dc(args, shapes, prefix_t, reference, preview_factors=factors)
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


def low_cfg_branch_dc_receipt(
    rows: list[dict[str, Any]], *, counter_deltas: dict[str, int | None] | None = None
) -> dict[str, Any]:
    """Summarize the observed evaluations, first and last, per branch."""

    measured = [row for row in rows if row.get("status") == "measured"]
    summary = {}
    for name in _BRANCHES:
        entries = [(row, row[name]) for row in measured if row.get(name) is not None]
        summary[name] = (
            {
                position: {
                    "stage": row["stage"],
                    "sigma": row["sigma"],
                    "anchored_step_rms": entry["anchored_step_rms"],
                    "preview_luma_anchored_step": entry.get("preview_luma_anchored_step"),
                }
                for position, (row, entry) in (("first", entries[0]), ("last", entries[-1]))
            }
            if entries
            else None
        )
    counter_deltas = counter_deltas or {}
    stage_counts: dict[str, dict[str, Any]] = {}
    for stage in ("low", "probe"):
        observed = sum(row.get("stage") == stage for row in rows)
        logical = counter_deltas.get(f"sampler_logical_calls_{stage}")
        actual = counter_deltas.get(f"transformer_actual_nfe_{stage}")
        forecast = counter_deltas.get(f"spectrum_forecast_calls_{stage}")
        stage_counts[stage] = {
            "cfg_hook_observations": observed,
            "measured": sum(row.get("stage") == stage and row.get("status") == "measured" for row in rows),
            "logical_sampler_calls": logical,
            "actual_transformer_nfe": actual,
            "spectrum_forecasts": forecast,
            "logical_calls_without_cfg_observation": max(0, logical - observed) if logical is not None else None,
        }
    return {
        "policy": LOW_CFG_BRANCH_DC_POLICY,
        "boundary_dc_policy": SIGNED_BOUNDARY_DC_POLICY,
        "status": "measured" if measured else ("not_observed" if not rows else "error"),
        "protected_reference": "sampler_latent_image_protected_token",
        "evaluations": len(rows),
        "failed_evaluations": len(rows) - len(measured),
        "uncond_candidate_rows": sum(1 for row in measured if row.get("uncond_candidate")),
        "uncond_verified_evaluations": None,
        "observation_scope": "sampler_post_cfg_function_calls_not_logical_sampler_steps",
        "stage_counts": stage_counts,
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
    "protected_reference_means",
]
