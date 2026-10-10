"""Experimental last-prefix residual transport for uniform-source continuation.

The provider's target-grid prefix rendering can differ from the authoritative
prefix. This opt-in intervention adds the last prefix's exact-minus-learned
residual to the suffix, weighted by learned change and residual persistence
fitted on prefix pairs. A separate prefix-pair check must improve before it
applies. Weights may be fractional.

Prefix calibration cannot establish suffix motion or occlusion safety. Fine
detail lost by projection can move while learned change stays small; transport
can then freeze stale detail or create an overlay. Rendered GPU quality is
unvalidated. The scheduler leaves this disabled unless explicitly requested.
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

UNIFORM_SOURCE_DETAIL_TRANSPORT_POLICY = "prefix_calibrated_static_detail_transport_v1"
UNIFORM_SOURCE_DETAIL_TRANSPORT_BINS = 16
UNIFORM_SOURCE_DETAIL_TRANSPORT_POOL = 3
# Latent frame 0 of an H3 clip is a single-frame VAE token, so its upscaler
# error is not representative of the multi-frame tokens that follow it.
UNIFORM_SOURCE_DETAIL_TRANSPORT_FIRST_CALIBRATION_FRAME = 1
UNIFORM_SOURCE_DETAIL_TRANSPORT_MIN_CALIBRATION_FRAMES = 3


def _pool(value: torch.Tensor) -> torch.Tensor:
    """Average a ``B x H x W`` map over a small spatial neighbourhood."""

    pool = UNIFORM_SOURCE_DETAIL_TRANSPORT_POOL
    return F.avg_pool2d(
        value.unsqueeze(1),
        kernel_size=pool,
        stride=1,
        padding=pool // 2,
        count_include_pad=False,
    ).squeeze(1)


def _local_change(frame: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Neighbourhood-maximum RMS change between two ``B x C x H x W`` frames.

    The maximum, not the mean, so a location next to moving content is treated as
    changed and does not receive the anchor frame's detail across a motion edge.
    """

    pool = UNIFORM_SOURCE_DETAIL_TRANSPORT_POOL
    change = (frame - anchor).square().mean(dim=1).sqrt()
    return F.max_pool2d(change.unsqueeze(1), kernel_size=pool, stride=1, padding=pool // 2).squeeze(1)


def _transport_change(learned: torch.Tensor, anchor: int, frame: int) -> torch.Tensor:
    """Change that limits how far frame ``anchor``'s error can be carried to ``frame``.

    The anchor's error only describes static content, so motion at the anchor
    (against the frame before it) counts as much as change since the anchor.
    """

    since_anchor = _local_change(learned[:, :, frame], learned[:, :, anchor])
    at_anchor = _local_change(learned[:, :, anchor], learned[:, :, anchor - 1])
    return torch.maximum(since_anchor, at_anchor)


def _pair_statistics(
    learned: torch.Tensor,
    residual: torch.Tensor,
    pairs: list[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return change, ``<D_t, D_a>`` and ``|D_a|^2`` per location for each pair."""

    changes = []
    numerators = []
    denominators = []
    for anchor, frame in pairs:
        changes.append(_transport_change(learned, anchor, frame))
        numerators.append(_pool((residual[:, :, frame] * residual[:, :, anchor]).sum(dim=1)))
        denominators.append(_pool(residual[:, :, anchor].square().sum(dim=1)))
    return torch.stack(changes), torch.stack(numerators), torch.stack(denominators)


def _monotone_nonincreasing(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Weighted pool-adjacent-violators fit of a non-increasing sequence."""

    blocks: list[list[float]] = []
    for value, weight in zip(values.tolist(), weights.tolist(), strict=True):
        blocks.append([value * weight, weight, 1.0])
        while len(blocks) > 1:
            previous, current = blocks[-2], blocks[-1]
            previous_mean = previous[0] / max(previous[1], 1e-30)
            current_mean = current[0] / max(current[1], 1e-30)
            if current_mean <= previous_mean:
                break
            blocks[-2] = [previous[0] + current[0], previous[1] + current[1], previous[2] + current[2]]
            blocks.pop()
    fitted: list[float] = []
    for total, weight, count in blocks:
        fitted.extend([total / max(weight, 1e-30)] * int(count))
    return torch.tensor(fitted, dtype=values.dtype, device=values.device)


class _PersistenceCurve:
    """Monotone persistence weight as a function of local learned change."""

    def __init__(self, centers: torch.Tensor, persistence: torch.Tensor) -> None:
        self.centers = centers
        self.persistence = persistence
        # Suppress bins with persistence at or below one half. Intermediate
        # persistence still produces fractional weights; this cannot guarantee
        # that stale detail or overlays are absent from the unknown suffix.
        self.weights = (2.0 * persistence - 1.0).clamp(0.0, 1.0)

    @classmethod
    def fit(
        cls,
        change: torch.Tensor,
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> _PersistenceCurve | None:
        flat_change = change.reshape(-1)
        flat_numerator = numerator.reshape(-1)
        flat_denominator = denominator.reshape(-1)
        if flat_change.numel() < UNIFORM_SOURCE_DETAIL_TRANSPORT_BINS:
            return None
        quantiles = torch.linspace(0.0, 1.0, UNIFORM_SOURCE_DETAIL_TRANSPORT_BINS + 1, device=flat_change.device)
        edges = torch.unique(torch.quantile(flat_change, quantiles))
        if edges.numel() < 2:
            return None
        index = torch.bucketize(flat_change, edges[1:-1], right=True)
        bins = int(edges.numel() - 1)
        numerator_sum = torch.zeros(bins, dtype=torch.float64, device=flat_change.device)
        denominator_sum = torch.zeros_like(numerator_sum)
        change_sum = torch.zeros_like(numerator_sum)
        count = torch.zeros_like(numerator_sum)
        numerator_sum.index_add_(0, index, flat_numerator.to(torch.float64))
        denominator_sum.index_add_(0, index, flat_denominator.to(torch.float64))
        change_sum.index_add_(0, index, flat_change.to(torch.float64))
        count.index_add_(0, index, torch.ones_like(flat_change, dtype=torch.float64))
        occupied = (count > 0) & (denominator_sum > 0)
        if int(occupied.sum().item()) < 1:
            return None
        centers = change_sum[occupied] / count[occupied]
        raw = (numerator_sum[occupied] / denominator_sum[occupied]).clamp(0.0, 1.0)
        persistence = _monotone_nonincreasing(raw, denominator_sum[occupied])
        return cls(centers.to(torch.float32), persistence.to(torch.float32))

    def __call__(self, change: torch.Tensor) -> torch.Tensor:
        centers = self.centers.to(change.device)
        weights = self.weights.to(change.device)
        if centers.numel() == 1:
            result = torch.full_like(change, float(weights[0]))
        else:
            upper = torch.searchsorted(centers, change.contiguous()).clamp(1, centers.numel() - 1)
            lower = upper - 1
            span = (centers[upper] - centers[lower]).clamp_min(1e-12)
            position = ((change - centers[lower]) / span).clamp(0.0, 1.0)
            result = weights[lower] + (weights[upper] - weights[lower]) * position
        # Changes larger than any calibrated prefix change have no persistence
        # evidence; fade the last calibrated weight to zero by twice that change.
        last_center = centers[-1].clamp_min(1e-12)
        beyond = ((2.0 * last_center - change) / last_center).clamp(0.0, 1.0)
        return torch.where(change > last_center, float(weights[-1]) * beyond, result)

    def as_receipt(self) -> dict[str, list[float]]:
        return {
            "change_centers": [float(value) for value in self.centers.tolist()],
            "persistence": [float(value) for value in self.persistence.tolist()],
            "transport_weights": [float(value) for value in self.weights.tolist()],
        }


def _transport_error_ratio(
    curve: _PersistenceCurve,
    change: torch.Tensor,
    learned_residual_pairs: tuple[torch.Tensor, torch.Tensor],
) -> float:
    """Held-out ``|D_t - w D_a|^2 / |D_t|^2`` over the given pairs."""

    target, anchor = learned_residual_pairs
    weight = curve(change).unsqueeze(2)
    before = target.square().sum()
    after = (target - weight * anchor).square().sum()
    return float((after / before.clamp_min(1e-30)).item())


def apply_uniform_source_detail_transport(
    learned_clean: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    prefix_t: int,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Carry the exact prefix's static detail into the learned suffix.

    ``learned_clean`` is the learned provider's full target-grid clean clip,
    including its own rendering of the prefix frames. The returned tensor equals
    ``learned_clean`` on the prefix frames and adds ``w * (E_last - L_last)`` on
    every suffix frame.
    """

    if learned_clean.ndim != 5 or exact_prefix.ndim != 5:
        raise ValueError("uniform-source detail transport expects BxCxTxHxW tensors")
    prefix_t = int(prefix_t)
    temporal = int(learned_clean.shape[2])
    if not 0 < prefix_t < temporal:
        raise ValueError("uniform-source detail transport requires a valid prefix boundary")
    if tuple(exact_prefix.shape) != tuple(learned_clean[:, :, :prefix_t].shape):
        raise ValueError("uniform-source detail transport prefix geometry differs from the learned clip")
    if not learned_clean.is_floating_point() or not exact_prefix.is_floating_point():
        raise TypeError("uniform-source detail transport expects floating-point tensors")

    receipt: dict[str, Any] = {
        "policy": UNIFORM_SOURCE_DETAIL_TRANSPORT_POLICY,
        "applied": False,
        "prefix_t": prefix_t,
        "suffix_tokens": temporal - prefix_t,
        "authoritative_prefix_modified": False,
        "suffix_motion_safety_established": False,
        "rendered_quality_validated": False,
        "extra_h3_nfe": 0,
        "extra_provider_calls": 0,
        "extra_vae_calls": 0,
        "extra_sampler_lifetimes": 0,
        "extra_history_boundaries": 0,
    }
    first = UNIFORM_SOURCE_DETAIL_TRANSPORT_FIRST_CALIBRATION_FRAME
    if prefix_t - first < UNIFORM_SOURCE_DETAIL_TRANSPORT_MIN_CALIBRATION_FRAMES:
        receipt["reason"] = "prefix_too_short_for_calibration"
        return learned_clean, receipt

    with torch.no_grad():
        learned = learned_clean.detach().to(torch.float32)
        residual = exact_prefix.detach().to(device=learned.device, dtype=torch.float32) - learned[:, :, :prefix_t]
        if not bool(torch.isfinite(learned).all().item()) or not bool(torch.isfinite(residual).all().item()):
            raise RuntimeError("uniform-source detail transport received non-finite tensors")

        # Every anchor needs its preceding frame for the anchor-motion term.
        pairs = [(anchor, frame) for anchor in range(first + 1, prefix_t) for frame in range(anchor + 1, prefix_t)]
        fit_pairs = [pair for pair in pairs if pair[1] % 2 == 0]
        holdout_pairs = [pair for pair in pairs if pair[1] % 2 == 1]
        if not fit_pairs or not holdout_pairs:
            receipt["reason"] = "prefix_too_short_for_calibration"
            return learned_clean, receipt
        fit_change, fit_numerator, fit_denominator = _pair_statistics(learned, residual, fit_pairs)
        fit_curve = _PersistenceCurve.fit(fit_change, fit_numerator, fit_denominator)
        if fit_curve is None:
            receipt["reason"] = "calibration_degenerate"
            return learned_clean, receipt
        holdout_change = torch.stack([_transport_change(learned, anchor, frame) for anchor, frame in holdout_pairs])
        holdout_target = torch.stack([residual[:, :, frame] for _anchor, frame in holdout_pairs])
        holdout_anchor = torch.stack([residual[:, :, anchor] for anchor, _frame in holdout_pairs])
        holdout_ratio = _transport_error_ratio(fit_curve, holdout_change, (holdout_target, holdout_anchor))
        receipt["holdout_error_ratio"] = holdout_ratio
        receipt["fit_pairs"] = len(fit_pairs)
        receipt["holdout_pairs"] = len(holdout_pairs)
        if not math.isfinite(holdout_ratio) or holdout_ratio >= 1.0:
            receipt["reason"] = "holdout_not_improved"
            return learned_clean, receipt

        change, numerator, denominator = _pair_statistics(learned, residual, pairs)
        curve = _PersistenceCurve.fit(change, numerator, denominator)
        if curve is None:
            receipt["reason"] = "calibration_degenerate"
            return learned_clean, receipt

        anchor_residual = residual[:, :, prefix_t - 1]
        suffix = learned[:, :, prefix_t:]
        suffix_change = torch.stack(
            [_transport_change(learned, prefix_t - 1, frame) for frame in range(prefix_t, temporal)],
            dim=1,
        )
        weight = curve(suffix_change)
        delta = weight.unsqueeze(1) * anchor_residual.unsqueeze(2)
        transported = learned_clean.clone()
        transported[:, :, prefix_t:] = (suffix + delta).to(dtype=learned_clean.dtype)
        if not bool(torch.isfinite(transported).all().item()):
            raise RuntimeError("uniform-source detail transport produced non-finite values")

        per_token_weight = weight.mean(dim=(0, 2, 3))
        receipt.update(
            applied=True,
            reason="holdout_improved",
            anchor_residual_rms=float(anchor_residual.square().mean().sqrt().item()),
            delta_rms=float(delta.square().mean().sqrt().item()),
            first_suffix_weight_mean=float(per_token_weight[0].item()),
            last_suffix_weight_mean=float(per_token_weight[-1].item()),
            suffix_weight_mean=float(weight.mean().item()),
            suffix_weight_above_half_fraction=float((weight > 0.5).float().mean().item()),
            **curve.as_receipt(),
        )
    return transported, receipt


__all__ = [
    "UNIFORM_SOURCE_DETAIL_TRANSPORT_POLICY",
    "apply_uniform_source_detail_transport",
]
