"""Output-neutral, bounded runtime provenance receipts for Flow diagnostics."""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Any

import torch

_EFFECTIVE_SEED_MASK = (1 << 63) - 1


def effective_seed(seed: int) -> int:
    """Return the exact seed domain used by Flow's CPU noise generator."""
    return int(seed) & _EFFECTIVE_SEED_MASK


def tensor_provenance(value: torch.Tensor, *, samples: int = 64) -> dict[str, Any]:
    """Fingerprint a tensor from a bounded fixed sample without modifying it."""
    if not isinstance(value, torch.Tensor):
        raise TypeError("tensor provenance requires a torch.Tensor")
    if isinstance(samples, bool) or int(samples) < 1:
        raise ValueError("tensor provenance samples must be positive")

    detached = value.detach()
    numel = int(detached.numel())
    report: dict[str, Any] = {
        "shape": [int(v) for v in detached.shape],
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "numel": numel,
        "sample_count": 0,
        "sample_sha256": hashlib.sha256(b"").hexdigest(),
        "sample_finite": True,
        "sample_mean": 0.0,
        "sample_rms": 0.0,
        "sample_min": 0.0,
        "sample_max": 0.0,
    }
    if numel == 0:
        return report

    count = min(int(samples), numel)
    flat = detached.reshape(-1)
    if count == 1:
        indices = torch.zeros(1, device=flat.device, dtype=torch.long)
    else:
        positions = torch.arange(count, device=flat.device, dtype=torch.long)
        indices = torch.div(positions * (numel - 1), count - 1, rounding_mode="floor")
    sample = flat.index_select(0, indices).to(device="cpu", dtype=torch.float32)
    values = [float(v) for v in sample.tolist()]
    payload = b"".join(struct.pack("<f", v) for v in values)
    finite = all(math.isfinite(v) for v in values)
    report.update(
        sample_count=count,
        sample_sha256=hashlib.sha256(payload).hexdigest(),
        sample_finite=finite,
    )
    if finite:
        mean = math.fsum(values) / count
        rms = math.sqrt(math.fsum(v * v for v in values) / count)
        report.update(
            sample_mean=mean,
            sample_rms=rms,
            sample_min=min(values),
            sample_max=max(values),
        )
    return report
