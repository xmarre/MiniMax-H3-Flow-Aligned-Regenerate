"""Read-only matched-frame photometric evidence for captured continuation stages.

This ledger establishes which saved stage contains a tone trajectory; it does
not infer generative causality from differently conditioned decoded clips.
"""

from __future__ import annotations

import itertools

import torch

STAGE_ORDER = (
    "source_grid",
    "provider_native",
    "provider",
    "pre_high",
    "first_high_before_flow",
    "first_high_after_flow",
    "final",
)
LUMA = (0.2126, 0.7152, 0.0722)


def _region(frame_stack, bounds):
    if bounds is None:
        return frame_stack
    height, width = frame_stack.shape[1:3]
    x0, y0, x1, y1 = bounds
    xa, xb = round(x0 * width), round(x1 * width)
    ya, yb = round(y0 * height), round(y1 * height)
    if xb - xa < 2 or yb - ya < 2:
        raise ValueError("static ROI too small on decoded stage grid")
    return frame_stack[:, ya:yb, xa:xb]


def _trajectory(frames, labels, anchor, bounds=None):
    selected = _region(frames, bounds)
    luma = selected @ torch.tensor(LUMA, dtype=torch.float32)
    means = luma.mean(dim=(1, 2))
    rgb_means = selected.mean(dim=(1, 2))
    anchor_index = labels.index(anchor)
    reference = means[anchor_index]
    return {
        "frame_labels": list(labels),
        "native_hw": list(frames.shape[1:3]),
        "anchor_frame": anchor,
        "anchor_luma": float(reference),
        "luma_mean": means.tolist(),
        "luma_std": luma.std(dim=(1, 2), correction=0).tolist(),
        "rgb_mean": rgb_means.tolist(),
        "luma_change_from_anchor": (means - reference).tolist(),
        "adjacent_luma_change": (means[1:] - means[:-1]).tolist(),
    }


def tone_attribution_ledger(pixels, frame_labels, join_frame, rois, *, production_images=None):
    """Compare stage trajectories on identical global pixel times.

    Prefix anchor is decoder output from the second chunk's discarded context,
    NOT the previously assembled clip. Production IMAGE is optional and, when
    supplied, must be the entire globally indexed assembled timeline.
    """
    labels = list(frame_labels)
    if len(labels) < 3 or labels != list(range(labels[0], labels[-1] + 1)):
        raise ValueError("tone attribution requires contiguous global frame labels")
    anchor = join_frame - 1
    if anchor not in labels or join_frame not in labels:
        raise ValueError("tone attribution requires join and previous decoded context")
    if not all(name in pixels for name in STAGE_ORDER):
        raise ValueError("tone attribution requires all seven full-stage operands")
    if any(tuple(pixels[name].shape[:1]) != (len(labels),) for name in STAGE_ORDER):
        raise ValueError("full-stage decoded frame count differs across operands")

    areas = {"full_frame": None, **dict(rois)}
    regions = {}
    for region_name, bounds in areas.items():
        stage_rows = {name: _trajectory(pixels[name], labels, anchor, bounds) for name in STAGE_ORDER}
        pairs = {}
        for left, right in itertools.pairwise(STAGE_ORDER):
            previous = stage_rows[left]
            current = stage_rows[right]
            pair = {
                "left": left,
                "right": right,
                "frame_labels": labels,
                "same_time_luma_difference": [
                    b - a for a, b in zip(previous["luma_mean"], current["luma_mean"], strict=True)
                ],
                "change_in_join_relative_luma": [
                    b - a
                    for a, b in zip(
                        previous["luma_change_from_anchor"], current["luma_change_from_anchor"], strict=True
                    )
                ],
                "representation_domain": (
                    "different_latent_and_decoder_spatial_grids"
                    if left == "source_grid"
                    else "shared_target_decoder_grid"
                ),
                "causal_attribution_from_this_pair_alone": False,
            }
            pairs[f"{left}_to_{right}"] = pair
        regions[region_name] = {"stages": stage_rows, "sequential_pairs": pairs}

    result = {
        "policy": "full_stage_boundary_tone_ledger_v1",
        "global_frame_labels": labels,
        "join_frame": join_frame,
        "anchor_frame": anchor,
        "anchor_is_discarded_current_chunk_context": True,
        "stage_order": list(STAGE_ORDER),
        "stage_outputs_decoded_with_identical_connected_vae": True,
        "source_grid_cross_representation_comparisons_not_causal": True,
        "independent_prior_chunk_decoder_not_captured_by_bundle": True,
        "raw_production_output_provided": production_images is not None,
        "rendered_defect_causality_established": False,
        "regions": regions,
    }

    if production_images is not None:
        if not isinstance(production_images, torch.Tensor):
            raise TypeError("production_images must be ComfyUI IMAGE tensor")
        if production_images.ndim != 4 or production_images.shape[-1] != 3:
            raise ValueError("production_images must be [frames, height, width, RGB]")
        if labels[0] < 0 or production_images.shape[0] <= labels[-1]:
            raise ValueError("production_images must include globally indexed join and measured frames")
        if tuple(production_images.shape[1:]) != tuple(pixels["final"].shape[1:]):
            raise ValueError("production_images geometry must match final target decode exactly")
        # Slice before transfer. Avoid copying the complete assembled video to CPU.
        actual = production_images[labels[0] : labels[-1] + 1].detach().to(device="cpu", dtype=torch.float32)
        if not bool(torch.isfinite(actual).all()) or not bool(((actual >= 0) & (actual <= 1)).all()):
            raise ValueError("production_images must contain finite RGB values within [0,1]")
        if actual.shape != pixels["final"].shape:
            raise ValueError("production_images frame selection does not match decoded times")
        comparisons = {}
        for region_name, bounds in areas.items():
            prod = _trajectory(actual, labels, anchor, bounds)
            decoded = regions[region_name]["stages"]["final"]
            comparisons[region_name] = {
                "assembled": prod,
                "same_frame_assembled_minus_replayed_final_luma": [
                    b - a for a, b in zip(decoded["luma_mean"], prod["luma_mean"], strict=True)
                ],
                "assembled_minus_replay_join_step": (
                    prod["luma_mean"][labels.index(join_frame)]
                    - prod["luma_mean"][labels.index(anchor)]
                    - decoded["luma_mean"][labels.index(join_frame)]
                    + decoded["luma_mean"][labels.index(anchor)]
                ),
            }
        result["production_image_comparison"] = {
            "image_role": "assembled_output_after_any_continuum_rgb_adjustment",
            "previous_chunk_reference_is_real_assembled_frame": True,
            "comparison_is_not_raw_vae_causality": True,
            "includes_encoder_effects": False,
            "regions": comparisons,
        }
    return result
