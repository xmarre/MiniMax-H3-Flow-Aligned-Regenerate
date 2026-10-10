"""Read-only, confidence-qualified image-space background motion evidence.

Unlike absolute ROI phase matching, this tracks features consecutively through
defocus before estimating cumulative background similarity scale. Requires
OpenCV when invoked; no production paths depend on OpenCV at import time.
"""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch

POLICY = "h3_static_background_feature_trajectory_v1"


def _grayscale_frames(frames: torch.Tensor) -> np.ndarray:
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[-1] < 3:
        raise ValueError("background tracking expects IMAGE [T,H,W,C]")
    if min(frames.shape[1:3]) < 80 or not bool(torch.isfinite(frames[..., :3]).all().item()):
        raise ValueError("background tracking requires finite images of at least 80px on each side")
    rgb = frames[..., :3].detach().to(device="cpu", dtype=torch.float32)
    gray = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    return gray.clamp(0, 1).mul(255).round().to(torch.uint8).numpy()


def track_background_features(
    frames: torch.Tensor,
    frame_labels: list[int],
    *,
    join_frame: int,
    rois: Mapping[str, tuple[float, float, float, float]],
) -> dict:
    """Track static room features around join_frame-1, without modifying inputs.

    Reports apparent image-space scale, not true camera magnification. Motion,
    changing texture, focus changes, or insufficient static regions may bias
    estimates; do not use this function to warp production frames.
    """
    if len(frame_labels) != len(frames) or len(set(frame_labels)) != len(frame_labels):
        raise ValueError("background feature labels must match frames")
    if type(join_frame) is not int or join_frame < 1 or not rois:
        raise ValueError("background tracking requires a join and named static ROIs")
    try:
        import cv2
    except ImportError:
        return {"policy": POLICY, "status": "opencv_unavailable", "images_modified": False}
    gray = _grayscale_frames(frames)
    height, width = gray.shape[1:3]
    idx = {value: index for index, value in enumerate(frame_labels)}
    anchor = join_frame - 1
    if anchor not in idx:
        return {"policy": POLICY, "status": "missing_prefix_anchor", "images_modified": False}
    mask = np.zeros((height, width), dtype=np.uint8)
    for rect in rois.values():
        if len(rect) != 4 or not (
            0 <= rect[0] < rect[2] <= 1 and 0 <= rect[1] < rect[3] <= 1
        ):
            raise ValueError("background tracking expects normalized XYXY ROIs")
        x0, y0, x1, y1 = rect
        mask[round(y0 * height) : round(y1 * height), round(x0 * width) : round(x1 * width)] = 255
    corners = cv2.goodFeaturesToTrack(
        gray[idx[anchor]], 900, 0.006, 8, mask=mask, blockSize=5
    )
    if corners is None or len(corners) < 25:
        return {
            "policy": POLICY,
            "status": "insufficient_initial_landmarks",
            "detected_landmarks": 0 if corners is None else len(corners),
            "images_modified": False,
        }
    original = corners[:, 0, :].copy()
    records = {
        anchor: {
            "status": "anchor",
            "tracked_landmarks": len(corners),
            "cumulative_apparent_scale_percent": 0.0,
        }
    }
    # Separate forward and backward trajectories anchor all measurements to
    # the same exact protected prefix frame.
    for direction in (-1, 1):
        previous = anchor
        positions = corners.copy()
        indices = np.arange(len(corners), dtype=np.int32)
        stop = min(frame_labels) - 1 if direction < 0 else max(frame_labels) + 1
        for frame_no in range(anchor + direction, stop, direction):
            if previous not in idx or frame_no not in idx:
                break
            source, target = gray[idx[previous]], gray[idx[frame_no]]
            forward, forward_status, _ = cv2.calcOpticalFlowPyrLK(
                source, target, positions, None, winSize=(25, 25), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
            )
            if forward is None:
                records[frame_no] = {"status": "no_tracked_landmarks", "tracked_landmarks": 0}
                break
            backward, backward_status, _ = cv2.calcOpticalFlowPyrLK(
                target, source, forward, None, winSize=(25, 25), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.003),
            )
            if backward is None:
                records[frame_no] = {"status": "no_reverse_correspondence", "tracked_landmarks": 0}
                break
            fb_error = np.linalg.norm((positions - backward)[:, 0, :], axis=1)
            keep = (
                (forward_status[:, 0] > 0)
                & (backward_status[:, 0] > 0)
                & np.isfinite(fb_error)
                & (fb_error < 1.2)
            )
            positions = forward[keep]
            indices = indices[keep]
            previous = frame_no
            count = len(positions)
            if count < 25:
                records[frame_no] = {
                    "status": "insufficient_tracked_landmarks",
                    "tracked_landmarks": count,
                }
                break
            initial = original[indices]
            current = positions[:, 0, :]
            matrix, inliers = cv2.estimateAffinePartial2D(
                initial, current, method=cv2.RANSAC, ransacReprojThreshold=2.0,
                maxIters=1500, confidence=0.995, refineIters=10,
            )
            if matrix is None or inliers is None:
                records[frame_no] = {"status": "affine_fit_failed", "tracked_landmarks": count}
                continue
            supported = inliers[:, 0].astype(bool)
            n = int(supported.sum())
            projected = cv2.transform(initial[None], matrix)[0]
            residual = np.linalg.norm(projected - current, axis=1)
            median_error = float(np.median(residual[supported])) if n else float("inf")
            spread = float(np.ptp(initial[supported, 0])) / width if n else 0.0
            left = int(((initial[:, 0] < width * 0.35) & supported).sum())
            right = int(((initial[:, 0] > width * 0.65) & supported).sum())
            valid = (
                n >= 25 and n / count >= 0.30 and left >= 6 and right >= 6
                and spread >= 0.45 and median_error <= 1.5
            )
            row = {
                "status": "measured" if valid else "low_confidence_indeterminate",
                "tracked_landmarks": count,
                "ransac_inliers": n,
                "inlier_fraction": round(n / count, 4),
                "support_left": left,
                "support_right": right,
                "horizontal_support_fraction": round(spread, 4),
                "median_inlier_residual_px": round(median_error, 4),
            }
            if valid:
                scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
                delta_x = current[:, 0] - initial[:, 0]
                row.update({
                    "cumulative_apparent_scale_percent": round((scale - 1) * 100, 5),
                    "rotation_degrees": round(float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0]))), 5),
                    "left_median_dx_px": round(float(np.median(delta_x[initial[:, 0] < width * 0.35])), 4),
                    "right_median_dx_px": round(float(np.median(delta_x[initial[:, 0] > width * 0.65])), 4),
                })
            records[frame_no] = row
    return {
        "policy": POLICY,
        "status": "measured",
        "join_frame": join_frame,
        "anchor_frame": anchor,
        "canvas_hw": [height, width],
        "detected_landmarks": len(corners),
        "method": "sequential_pyramidal_LK_bidirectional_RANSAC_similarity",
        "measurement_is_not_camera_ground_truth": True,
        "regions_xyxy": {name: list(rect) for name, rect in rois.items()},
        "frame_trajectories": {str(frame): value for frame, value in sorted(records.items())},
        "images_modified": False,
        "extra_vae_calls": 0,
        "extra_h3_nfe": 0,
    }
