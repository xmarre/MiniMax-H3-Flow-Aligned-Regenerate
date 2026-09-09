"""Exact-prefix representation reconciliation for Mixed-Grid Continuum.

The learned 3D upscaler produces a coherent target-grid prefix and suffix, but
Mixed-Grid must discard that learned prefix and restore Continuum's authoritative
prefix.  The overlap therefore gives us an exact, same-frame measurement of the
representation residual introduced by that replacement.

This module transfers only the zero-spatial-mean part of the last overlap
residual onto the first generated suffix token.  The existing DC bridge owns the
spatial-mean component.  When both bridges are enabled their sum is exactly the
full same-frame residual, so the exact-prefix -> corrected-suffix transition
matches the upscaler's native learned-prefix -> learned-suffix transition before
fresh target-grid refinement.

No prefix value, later suffix token, audio value, noise, mask, conditioning, VAE
state, or model evaluation is changed here.  Disabled and already-matched paths
return the original tensor object.
"""

from __future__ import annotations

import math

import torch


def _rms(value: torch.Tensor) -> float:
    result = float(value.float().square().mean().sqrt().detach().cpu().item())
    if not math.isfinite(result):
        raise RuntimeError("representation bridge produced a non-finite RMS")
    return result


def _validate_pair(learned: torch.Tensor, exact_prefix: torch.Tensor) -> int:
    if not torch.is_tensor(learned) or not torch.is_tensor(exact_prefix):
        raise TypeError("representation bridge expects torch.Tensor inputs")
    if learned.ndim != 5 or exact_prefix.ndim != 5:
        raise ValueError("representation bridge expects BxCxTxHxW tensors")
    if not learned.is_floating_point() or not exact_prefix.is_floating_point():
        raise TypeError("representation bridge expects floating-point tensors")
    if learned.shape[:2] != exact_prefix.shape[:2] or learned.shape[-2:] != exact_prefix.shape[-2:]:
        raise ValueError("representation bridge learned/exact geometry differs")
    prefix_t = int(exact_prefix.shape[2])
    if prefix_t < 1 or prefix_t >= int(learned.shape[2]):
        raise ValueError("representation bridge requires a non-empty prefix and suffix")
    if not bool(torch.isfinite(learned).all().item()) or not bool(torch.isfinite(exact_prefix).all().item()):
        raise RuntimeError("representation bridge inputs contain NaN or Inf")
    return prefix_t


def disabled_suffix_representation_bridge_metrics(*, prefix_t: int, requested: bool = False) -> dict:
    p = int(prefix_t)
    if p < 1:
        raise ValueError("representation bridge prefix length must be positive")
    return {
        "suffix_representation_bridge_version": 1,
        "suffix_representation_bridge_requested": bool(requested),
        "suffix_representation_bridge_enabled": False,
        "suffix_representation_bridge_accepted": False,
        "suffix_representation_bridge_reason": "disabled" if not requested else "not_applied",
        "suffix_representation_bridge_prefix_t": p,
        "suffix_representation_bridge_corrected_tokens": 0,
        "suffix_representation_bridge_delta_rms": 0.0,
        "suffix_representation_bridge_dc_rms": 0.0,
        "suffix_representation_bridge_structural_rms": 0.0,
        "suffix_representation_bridge_centered_error_before": 0.0,
        "suffix_representation_bridge_centered_error_after": 0.0,
        "suffix_representation_bridge_centered_error_ratio": 1.0,
    }


@torch.no_grad()
def apply_suffix_representation_bridge(
    learned_clean_video: torch.Tensor,
    exact_prefix: torch.Tensor,
    *,
    requested: bool = True,
) -> tuple[torch.Tensor, dict]:
    """Transplant the measured overlap representation residual to suffix token 0.

    The correction is not a temporal fade or image-space blend.  Let ``L_p`` be
    the learned upscaler's last prefix token, ``E_p`` the exact token that will
    replace it, and ``L_s`` the first learned suffix token.  The measured overlap
    residual is ``D = E_p - L_p``.  This function applies ``D - mean_spatial(D)``
    to ``L_s``; the independent DC bridge may apply ``mean_spatial(D)``.

    Consequently, with both bridges active, ``(L_s + D) - E_p == L_s - L_p``:
    exact-prefix restoration preserves the upscaler's native boundary transition
    algebraically instead of fitting an affine camera transform after the fact.
    """

    prefix_t = _validate_pair(learned_clean_video, exact_prefix)
    if not isinstance(requested, bool):
        raise TypeError("representation bridge requested flag must be boolean")
    if not requested:
        return learned_clean_video, disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t)

    learned_last = learned_clean_video[:, :, prefix_t - 1].float()
    exact_last = exact_prefix[:, :, -1].to(device=learned_clean_video.device, dtype=torch.float32)
    suffix_first = learned_clean_video[:, :, prefix_t].float()
    delta = exact_last - learned_last
    dc = delta.mean(dim=(-2, -1), keepdim=True)
    structural = delta - dc

    native_transition = suffix_first - learned_last
    hard_transition = suffix_first - exact_last
    native_centered = native_transition - native_transition.mean(dim=(-2, -1), keepdim=True)
    hard_centered = hard_transition - hard_transition.mean(dim=(-2, -1), keepdim=True)
    before_error = _rms(hard_centered - native_centered)

    metrics = disabled_suffix_representation_bridge_metrics(prefix_t=prefix_t, requested=True)
    metrics.update(
        suffix_representation_bridge_delta_rms=_rms(delta),
        suffix_representation_bridge_dc_rms=_rms(dc),
        suffix_representation_bridge_structural_rms=_rms(structural),
        suffix_representation_bridge_centered_error_before=before_error,
    )
    if before_error == 0.0 or not bool(torch.count_nonzero(structural).item()):
        metrics.update(
            suffix_representation_bridge_reason="structural_overlap_already_matched",
            suffix_representation_bridge_centered_error_after=before_error,
            suffix_representation_bridge_centered_error_ratio=1.0,
        )
        return learned_clean_video, metrics

    corrected = learned_clean_video.clone()
    corrected[:, :, prefix_t].add_(structural.to(dtype=corrected.dtype))
    if not bool(torch.isfinite(corrected).all().item()):
        raise RuntimeError("representation bridge produced NaN or Inf")
    if not torch.equal(corrected[:, :, :prefix_t], learned_clean_video[:, :, :prefix_t]):
        raise RuntimeError("representation bridge modified prefix values")
    if not torch.equal(corrected[:, :, prefix_t + 1 :], learned_clean_video[:, :, prefix_t + 1 :]):
        raise RuntimeError("representation bridge modified unmeasured later suffix tokens")

    corrected_transition = corrected[:, :, prefix_t].float() - exact_last
    corrected_centered = corrected_transition - corrected_transition.mean(dim=(-2, -1), keepdim=True)
    after_error = _rms(corrected_centered - native_centered)
    ratio = after_error / max(before_error, 1e-12)
    metrics.update(
        suffix_representation_bridge_centered_error_after=after_error,
        suffix_representation_bridge_centered_error_ratio=ratio,
    )
    if not after_error < before_error:
        metrics.update(suffix_representation_bridge_reason="projection_did_not_reduce_measured_overlap_error")
        return learned_clean_video, metrics

    metrics.update(
        suffix_representation_bridge_enabled=True,
        suffix_representation_bridge_accepted=True,
        suffix_representation_bridge_reason="exact_overlap_structural_residual_transplanted",
        suffix_representation_bridge_corrected_tokens=1,
    )
    return corrected, metrics
