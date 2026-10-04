#!/usr/bin/env python3
"""Measure selected regions in an existing video or native replay pixel tensor.

Regions must be chosen by inspecting the scene. These measurements do not
classify a frame jump, scene motion, exposure change, or rendered acceptance.
This tool neither samples H3 nor modifies media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def roi_argument(value: str) -> tuple[int, int, int, int]:
    try:
        roi = tuple(int(v) for v in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("ROI requires x1,y1,x2,y2 integers") from exc
    if len(roi) != 4 or not (0 <= roi[0] < roi[2] and 0 <= roi[1] < roi[3]):
        raise argparse.ArgumentTypeError("ROI requires nonempty, nonnegative x1,y1,x2,y2 bounds")
    return roi


def track_region(left, right, roi):
    gray_left = cv2.cvtColor(np.clip(left * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    gray_right = cv2.cvtColor(np.clip(right * 255, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    x1, y1, x2, y2 = roi
    mask = np.zeros_like(gray_left)
    mask[y1:y2, x1:x2] = 255
    points = cv2.goodFeaturesToTrack(gray_left, 200, 0.01, 5, mask=mask, blockSize=5)
    if points is None or len(points) < 8:
        return None
    following, forward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_left, gray_right, points, None, winSize=(31, 31), maxLevel=3
    )
    preceding, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        gray_right, gray_left, following, None, winSize=(31, 31), maxLevel=3
    )
    keep = (
        (forward_status[:, 0] == 1)
        & (backward_status[:, 0] == 1)
        & (np.linalg.norm(preceding[:, 0] - points[:, 0], axis=1) < 0.5)
    )
    if keep.sum() < 8:
        return None
    return points[keep, 0], following[keep, 0]


def measure_motion(left, right, roi):
    tracks = track_region(left, right, roi)
    if tracks is None:
        return {"status": "insufficient_tracks"}
    before, after = tracks
    delta = after - before
    center = np.median(delta, axis=0)
    return {
        "status": "measured",
        "tracked": len(delta),
        "dx": float(center[0]),
        "dy": float(center[1]),
        "vector_mad": float(np.median(np.linalg.norm(delta - center, axis=1))),
    }


def measure_tone(left, right, roi):
    tracks = track_region(left, right, roi)
    if tracks is None:
        return {"status": "insufficient_tracks"}
    before, after = tracks
    affine, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=1.0)
    if affine is None or inliers.sum() < 6:
        return {"status": "insufficient_affine_inliers"}
    height, width = left.shape[:2]
    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
    aligned = cv2.warpAffine(right, affine, (width, height), flags=flags)
    validity = cv2.warpAffine(np.ones((height, width), np.float32), affine, (width, height), flags=flags)
    x1, y1, x2, y2 = roi
    valid = validity[y1:y2, x1:x2] > 0.999
    if not valid.any():
        return {"status": "no_valid_aligned_pixels"}
    weights = np.array([0.2126, 0.7152, 0.0722], np.float32)
    left_luma = (left[y1:y2, x1:x2] @ weights)[valid]
    right_luma = (aligned[y1:y2, x1:x2] @ weights)[valid]
    if left_luma.std() < 1e-8 or right_luma.std() < 1e-8:
        return {"status": "constant_luma"}
    gain, offset = np.linalg.lstsq(np.column_stack([left_luma, np.ones_like(left_luma)]), right_luma, rcond=None)[0]

    def ratio(right_value, left_value):
        return float(right_value / left_value) if abs(left_value) > 1e-8 else None

    return {
        "status": "measured",
        "tracked": len(before),
        "inliers": int(inliers.sum()),
        "valid_pixels": int(valid.sum()),
        "luma_mean_ratio": ratio(right_luma.mean(), left_luma.mean()),
        "luma_std_ratio": ratio(right_luma.std(), left_luma.std()),
        "luma_p05_ratio": ratio(np.percentile(right_luma, 5), np.percentile(left_luma, 5)),
        "affine_luma_gain": float(gain),
        "affine_luma_offset": float(offset),
        "affine_luma_residual_rms": float(np.mean((right_luma - gain * left_luma - offset) ** 2) ** 0.5),
        "correlation": float(np.corrcoef(left_luma, right_luma)[0, 1]),
    }


def frames(args):
    if args.native_frames:
        import torch

        values = torch.load(args.native_frames, map_location="cpu", weights_only=True)
        if not isinstance(values, torch.Tensor) or values.ndim != 4 or values.shape[0] != 3:
            raise ValueError("native pixels must be a CxTxHxW RGB tensor")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("native pixels must be finite floating point")
        if values.min() < 0 or values.max() > 1:
            raise ValueError("native pixels must be finalized RGB in [0,1]")
        for index in range(values.shape[1]):
            if args.stop_frame is not None and args.start_frame + index >= args.stop_frame:
                break
            yield np.ascontiguousarray(values[:, index].permute(1, 2, 0).float().numpy())
        return
    capture = cv2.VideoCapture(str(args.video))
    try:
        if not capture.isOpened():
            raise ValueError("video could not be opened")
        capture.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)
        index = args.start_frame
        while args.stop_frame is None or index < args.stop_frame:
            found, bgr = capture.read()
            if not found:
                break
            yield cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255
            index += 1
    finally:
        capture.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--native-frames", type=Path, help="stage .pt produced by the native replay tool")
    parser.add_argument("--start-frame", type=int, default=0, help="video seek position or native frame-label offset")
    parser.add_argument("--stop-frame", type=int, help="exclusive final frame label")
    parser.add_argument("--motion-roi", type=roi_argument, required=True)
    parser.add_argument("--tone-roi", type=roi_argument, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.start_frame < 0 or (args.stop_frame is not None and args.stop_frame <= args.start_frame):
        parser.error("frame range must be nonnegative and nonempty")
    cv2.setNumThreads(1)
    path = args.video or args.native_frames
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(block)
    report = {
        "policy": "offline_decoded_regional_audit_v1",
        "source": path.name,
        "source_sha256": digest.hexdigest(),
        "source_kind": "native_replay_pixels" if args.native_frames else "encoded_video",
        "opencv_version": cv2.__version__,
        "motion_roi": args.motion_roi,
        "tone_roi": args.tone_roi,
        "native_preceding_temporal_blend_reproduced": False if args.native_frames else None,
        "rendered_acceptance": False,
        "measurements": [],
    }
    previous = None
    for index, current in enumerate(frames(args), args.start_frame):
        height, width = current.shape[:2]
        if any(roi[2] > width or roi[3] > height for roi in (args.motion_roi, args.tone_roi)):
            raise ValueError("ROI exceeds image geometry")
        if previous is not None:
            report["measurements"].append(
                {
                    "pair": [index - 1, index],
                    "motion": measure_motion(previous, current, args.motion_roi),
                    "tone": measure_tone(previous, current, args.tone_roi),
                }
            )
        previous = current
    if not report["measurements"]:
        raise ValueError("audit requires at least two frames")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
