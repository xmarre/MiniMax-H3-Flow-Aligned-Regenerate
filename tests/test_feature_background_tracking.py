"""Independent geometry tracker tests: actual landmarks, false stability and ownership."""

import json

import pytest
import torch

from h3_flow_regenerate.feature_background_tracking import track_background_features


def _synthetic(*, zoom: float = 0.006, blur: bool = True):
    cv2 = pytest.importorskip("cv2")
    np = pytest.importorskip("numpy")
    random = np.random.default_rng(7)
    base = random.integers(32, 225, size=(512, 512, 3), dtype=np.uint8)
    for y in range(10, 490, 21):
        for x in range(10, 490, 29):
            cv2.rectangle(base, (x, y), (x + 4, y + 6), (0, 250, 0), -1)
    frames = []
    labels = list(range(166, 192))
    for frame in labels:
        fractional_scale = zoom * max(0, min(1, (frame - 174) / 16))
        matrix = cv2.getRotationMatrix2D((256, 256), 0, 1 + fractional_scale)
        pixels = cv2.warpAffine(base, matrix, (512, 512), borderMode=cv2.BORDER_REFLECT101)
        if blur and frame >= 175:
            pixels = cv2.GaussianBlur(pixels, (3, 3), 0.8)
        frames.append(pixels)
    return torch.from_numpy(np.stack(frames)).float() / 255, labels


ROOM = {"left": (0.02, 0.15, 0.35, 0.80), "right": (0.65, 0.15, 0.98, 0.80)}


@pytest.mark.parametrize("zoom", [0, 0.006])
def test_feature_tracking_detects_cumulative_background_expansion_with_blur(zoom):
    images, labels = _synthetic(zoom=zoom)
    preserved = images.clone()
    result = track_background_features(images, labels, join_frame=175, rois=ROOM)
    assert result["status"] == "measured"
    assert result["detected_landmarks"] >= 100
    assert result["method"] == "sequential_pyramidal_LK_bidirectional_RANSAC_similarity"
    assert result["measurement_is_not_camera_ground_truth"] is True
    assert result["images_modified"] is False
    assert result["extra_vae_calls"] == result["extra_h3_nfe"] == 0
    assert torch.equal(images, preserved)
    assert result["frame_trajectories"]["174"]["cumulative_apparent_scale_percent"] == 0
    f190 = result["frame_trajectories"]["190"]
    assert f190["status"] == "measured"
    assert f190["ransac_inliers"] >= 80
    assert f190["support_left"] >= 6 and f190["support_right"] >= 6
    expected = 100 * zoom
    assert f190["cumulative_apparent_scale_percent"] == pytest.approx(expected, abs=0.10)
    if zoom:
        assert f190["left_median_dx_px"] < 0
        assert f190["right_median_dx_px"] > 0
        assert result["frame_trajectories"]["170"]["cumulative_apparent_scale_percent"] == pytest.approx(0, abs=0.05)
    json.dumps(result, allow_nan=False)


def test_feature_tracking_never_reports_missing_support_as_zero_scale():
    images = torch.full((12, 128, 128, 3), 0.5)
    result = track_background_features(
        images,
        list(range(169, 181)),
        join_frame=175,
        rois={"left": (0.0, 0.1, 0.35, 0.8), "right": (0.65, 0.1, 1, 0.8)},
    )
    assert result["status"] in ("opencv_unavailable", "insufficient_initial_landmarks")
    assert "cumulative_apparent_scale_percent" not in result


def test_feature_tracking_rejects_bad_frame_ownership():
    frames = torch.zeros(4, 128, 128, 3)
    with pytest.raises(ValueError, match="labels"):
        track_background_features(frames, [172, 173, 174], join_frame=175, rois=ROOM)
    with pytest.raises(ValueError, match="join"):
        track_background_features(frames, [172, 173, 174, 175], join_frame=0, rois=ROOM)
