"""Couple progressive flow noise on the ordinary half-pixel image grid.

Image interpolation is not a white-noise refinement: applying it directly to
Gaussian noise changes variance and correlations. Whiten its coarse analysis
rows, retain those modes, and put new Gaussian innovation in their nullspace.
Model drift uses a distinct, constant-preserving image right inverse.
"""

from __future__ import annotations

import functools
import math

import torch
import torch.nn.functional as F


@functools.lru_cache(maxsize=32)
def _axis_operators(source_n: int, target_n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return CPU float64 image analysis A, white analysis Q and drift lift P."""
    if not 0 < source_n <= target_n:
        raise ValueError("image residual refinement must not shrink an axis")
    if source_n == target_n:
        identity = torch.eye(source_n, dtype=torch.float64)
        return identity, identity, identity
    # Resize the rows of an identity image; columns are independent basis
    # functions. The unchanged column axis makes this the actual separable
    # antialiased bicubic image operator, including its boundary convention.
    analysis = F.interpolate(
        torch.eye(target_n, dtype=torch.float64)[None, None],
        size=(source_n, target_n),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )[0, 0]
    gram = analysis @ analysis.T
    values, vectors = torch.linalg.eigh(gram)
    if not bool(torch.isfinite(values).all()) or float(values.min()) <= 1e-12:
        raise ValueError("image noise analysis is rank deficient")
    inverse = (vectors / values) @ vectors.T
    white = (vectors / values.sqrt()) @ vectors.T @ analysis
    bicubic = F.interpolate(
        torch.eye(source_n, dtype=torch.float64)[None, None],
        size=(target_n, source_n),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    )[0, 0]
    # Correct ordinary interpolation only where it fails coarse consistency.
    # P*1=1 because both image resizes preserve constants; A*P=I.
    lift = bicubic + analysis.T @ inverse @ (torch.eye(source_n, dtype=torch.float64) - analysis @ bicubic)
    return analysis, white, lift


def _operators(source, target_hw):
    return tuple(
        tuple(value.to(device=source.device, dtype=torch.float32) for value in _axis_operators(s, t))
        for s, t in zip(source.shape[-2:], target_hw, strict=True)
    )


def _map(value, y, x):
    return torch.matmul(torch.matmul(y, value), x.T)


def _validate(source, innovation):
    if (
        not torch.is_tensor(source)
        or not torch.is_tensor(innovation)
        or source.ndim != 5
        or innovation.ndim != 5
        or not source.is_floating_point()
        or not innovation.is_floating_point()
        or source.shape[:-2] != innovation.shape[:-2]
        or source.device != innovation.device
    ):
        raise ValueError("image residual refinement requires matching floating BxCxT source and innovation")
    if not bool(torch.isfinite(source).all()) or not bool(torch.isfinite(innovation).all()):
        raise ValueError("image residual refinement requires finite inputs")
    if min(source.shape) < 1:
        raise ValueError("image residual refinement requires nonempty inputs")


def refine_image_residual(source, innovation):
    """Retain Q target=source; high innovation has exactly zero Q projection.

    For iid unit Gaussian source and independent iid unit Gaussian innovation,
    the target covariance is Q.T*Q + (I-Q.T*Q) = I. A measured model residual is
    not claimed to be Gaussian. This guarantees its coarse projection only.
    """
    _validate(source, innovation)
    (ay, qy, _py), (ax, qx, _px) = _operators(source, innovation.shape[-2:])
    del ay, ax
    if source.shape == innovation.shape:
        target = source.clone()
    else:
        coarse_innovation = _map(innovation.float(), qy, qx)
        target = (_map(source.float() - coarse_innovation, qy.T, qx.T) + innovation.float()).to(source.dtype)
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("image residual refinement produced non-finite target values")
    error = _map(target.float(), qy, qx) - source.float()
    return target, {
        "identity": source.shape == innovation.shape,
        "noise_lattice": "half_pixel_latent_v1",
        "analysis": "whitened_antialiased_bicubic_image_v1",
        "projection_rms_error": float(error.square().mean().sqrt()),
        "projection_max_abs_error": float(error.abs().max()),
        "innovation_coarse_projection_zero": True,
        "gaussian_marginal_if_source_standard": True,
        "extra_h3_nfe": 0,
    }


def transport_image_flow_residual(source, initial, innovation, *, noise_scale=1.0):
    """Refine Gaussian initial noise, then retain image-consistent model drift.

    This decomposition requires a deterministic low sampler. Stochastic low
    samplers must refine their complete measured residual instead, since their
    difference from initial noise contains additional stochastic increments.
    """
    _validate(source, innovation)
    if (
        not torch.is_tensor(initial)
        or initial.shape != source.shape
        or not initial.is_floating_point()
        or not bool(torch.isfinite(initial).all())
    ):
        raise ValueError("image flow transfer requires finite original source noise of matching shape")
    if not math.isfinite(float(noise_scale)) or float(noise_scale) <= 0:
        raise ValueError("image flow transfer requires a finite positive model noise scale")
    initial = initial.to(device=source.device, dtype=torch.float32)
    gaussian, report = refine_image_residual(initial, innovation)
    drift = source.float() - float(noise_scale) * initial
    (ay, _qy, py), (ax, _qx, px) = _operators(source, innovation.shape[-2:])
    target_drift = _map(drift, py, px)
    target = (float(noise_scale) * gaussian + target_drift).to(source.dtype)
    if report["identity"]:
        target = source.clone()
    drift_error = _map(target_drift, ay, ax) - drift
    if not bool(torch.isfinite(target).all()):
        raise RuntimeError("image flow transfer produced non-finite target residual")
    report["gaussian_projection_rms_error"] = report.pop("projection_rms_error")
    report["gaussian_projection_max_abs_error"] = report.pop("projection_max_abs_error")
    report.pop("gaussian_marginal_if_source_standard")
    report.update(
        coarse_projection_scope="initial_gaussian_component",
        normalized_gaussian_component_standard_if_initial_standard=True,
        gaussian_component_variance_if_initial_standard=float(noise_scale) ** 2,
        gaussian_noise_scale=float(noise_scale),
        drift_lattice="half_pixel_latent_v1",
        drift_lift="constant_preserving_image_right_inverse_v1",
        drift_projection_rms_error=float(drift_error.square().mean().sqrt()),
        drift_projection_max_abs_error=float(drift_error.abs().max()),
        source_drift_rms=float(drift.square().mean().sqrt()),
        target_drift_rms=float(target_drift.square().mean().sqrt()),
    )
    return target, report
