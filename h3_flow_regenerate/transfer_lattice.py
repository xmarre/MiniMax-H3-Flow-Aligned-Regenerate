"""The coordinate contract between a physical source carrier and learned transfer."""

from __future__ import annotations

import hashlib
import time

import torch
import torch.nn.functional as F

from .geometry import (
    H3_DENSE_PATCH_CENTER_LATTICE,
    H3_DENSE_TRANSPORT_LATTICES,
    H3_ROPE_BOX_HALF_PIXEL_LATTICE,
    normalize_h3_dense_transport_lattice,
)

H3_TRANSFER_LATTICE = H3_DENSE_PATCH_CENTER_LATTICE
H3_ROPE_BOX_TRANSFER_LATTICE = H3_ROPE_BOX_HALF_PIXEL_LATTICE
H3_TRANSFER_LATTICE_OPTIONS = H3_DENSE_TRANSPORT_LATTICES


class H3PatchLatticeTransferProvider:
    """Select the provider's encoder-to-decoder transport for a physical carrier."""

    def __init__(self, provider, *, lattice: str = H3_TRANSFER_LATTICE):
        lattice = normalize_h3_dense_transport_lattice(lattice)
        if getattr(provider, "h3_patch_lattice_api", None) != 2 or not callable(
            getattr(provider, "upscale_clean_video_h3_patch_lattice", None)
        ):
            raise RuntimeError(
                "Partitioned H3 continuation requires the upscaler provider with h3_patch_lattice_api=2. "
                "Update MiniMax H3 Latent Upscaler-Plus to the matching dense-transfer candidate."
            )
        if lattice != H3_TRANSFER_LATTICE and lattice not in tuple(getattr(provider, "h3_transport_lattices", ())):
            raise RuntimeError(
                f"transfer_lattice={lattice!r} requires an upscaler provider that advertises it in "
                "h3_transport_lattices. Update MiniMax H3 Latent Upscaler-Plus."
            )
        self.provider = provider
        self.lattice = lattice
        self.calls = 0

    def __getattr__(self, name):
        return getattr(self.provider, name)

    def upscale_clean_video(self, video, *, target_h, target_w):
        self.calls += 1
        if self.lattice == H3_TRANSFER_LATTICE:
            # Keep the historical call shape so older providers stay compatible.
            return self.provider.upscale_clean_video_h3_patch_lattice(video, target_h=target_h, target_w=target_w)
        return self.provider.upscale_clean_video_h3_patch_lattice(
            video, target_h=target_h, target_w=target_w, spatial_lattice=self.lattice
        )


def measure_paired_prefix_affine(learned, exact, *, prefix_t, frames=4):
    """Diagnostic affine correspondence of the same temporal prefix frames.

    Fit only pooled CPU features from up to four frames and eight channels. This
    never resamples a production operand or chooses a correction. The fitted
    displacement is in target latent cells, including both scale and shear.
    """
    started = time.perf_counter()
    start = max(0, prefix_t - frames)
    reports = []
    with torch.inference_mode(False), torch.enable_grad():
        a = learned[0, :, start:prefix_t].detach().float().cpu().clone().permute(1, 0, 2, 3)
        b = exact[0, :, start:prefix_t].detach().float().cpu().clone().permute(1, 0, 2, 3)
        source_sha = hashlib.sha256(a.contiguous().numpy().tobytes()).hexdigest()
        exact_sha = hashlib.sha256(b.contiguous().numpy().tobytes()).hexdigest()
        # Patch pooling removes within-patch phase from the affine estimation.
        a, b = F.avg_pool2d(a, 2), F.avg_pool2d(b, 2)
        h, w = b.shape[-2:]
        if min(h, w) < 8:
            return {"status": "unsupported", "reason": "prefix_grid_too_small", "output_mutated": False}
        channels = b.var(dim=(0, 2, 3)).topk(min(8, b.shape[1])).indices
        a, b = a[:, channels], b[:, channels]
        mean, std = b.mean((-1, -2), keepdim=True), b.std((-1, -2), keepdim=True).clamp_min(0.1)
        a, b = (a - mean) / std, (b - mean) / std
        yy, xx = torch.meshgrid(torch.arange(2, h - 2), torch.arange(2, w - 2), indexing="ij")
        x, y = 2 * xx.float() + 0.5, 2 * yy.float() + 0.5
        xc, yc = x - (2 * w - 1) / 2, y - (2 * h - 1) / 2
        for local, frame in enumerate(range(start, prefix_t)):
            params = torch.zeros(6, requires_grad=True)
            optimizer = torch.optim.LBFGS([params], max_iter=60, line_search_fn="strong_wolfe")
            target = b[local, :, 2:-2, 2:-2]

            def sample(params=params, local=local):
                dx = params[0] + params[2] * xc + params[3] * yc
                dy = params[1] + params[4] * xc + params[5] * yc
                grid = torch.stack(((x - dx - 0.5) / (w - 1) - 1, (y - dy - 0.5) / (h - 1) - 1), -1)[None]
                return F.grid_sample(a[local : local + 1], grid, align_corners=True, padding_mode="border")[0]

            zero_loss = float(F.smooth_l1_loss(sample(), target, beta=0.5).detach())

            def closure(optimizer=optimizer, sample=sample, target=target):
                optimizer.zero_grad()
                loss = F.smooth_l1_loss(sample(), target, beta=0.5)
                loss.backward()
                return loss

            optimizer.step(closure)
            fitted = sample().detach()
            p = params.detach().tolist()
            tiles = []
            for row in range(3):
                for col in range(3):
                    ys = slice(row * target.shape[-2] // 3, (row + 1) * target.shape[-2] // 3)
                    xs = slice(col * target.shape[-1] // 3, (col + 1) * target.shape[-1] // 3)
                    tiles.append(float((fitted[:, ys, xs] - target[:, ys, xs]).square().mean().sqrt()))
            reports.append(
                {
                    "frame": frame,
                    "center_dx_dy_cells": p[:2],
                    "affine_displacement_gradients": [p[2:4], p[4:6]],
                    "zero_huber": zero_loss,
                    "affine_huber": float(F.smooth_l1_loss(fitted, target, beta=0.5)),
                    "normalized_tile_rms_3x3": tiles,
                }
            )
    return {
        "status": "measured",
        "frames": reports,
        "channels": channels.tolist(),
        "learned_prefix_sample_sha256": source_sha,
        "exact_prefix_sample_sha256": exact_sha,
        "sign_convention": "sample learned at (x-dx,y-dy) to compare with exact at (x,y)",
        "policy": "same_frame_prefix_affine_diagnostic_v1",
        "output_mutated": False,
        "elapsed_ms": (time.perf_counter() - started) * 1000,
        "extra_h3_nfe": 0,
        "extra_provider_calls": 0,
        "fit_is_diagnostic_not_causal_proof": True,
    }
