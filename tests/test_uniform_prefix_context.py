"""Native conditioning preserves prefix information discarded by projection."""

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import resize_spatial_5d_h3_patch_lattice
from h3_flow_regenerate.partitioned_stage import PartitionedStagePlan
from h3_flow_regenerate.uniform_prefix_context import add_exact_prefix_visual_context


@pytest.fixture
def native():
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    return pytest.importorskip("comfy.ldm.minimax.model")


def plans(prefix):
    exact = PartitionedStagePlan(
        prefix=prefix, prefix_noise=torch.zeros_like(prefix), temporal=7, source_h=4, source_w=4
    )
    projected = resize_spatial_5d_h3_patch_lattice(prefix, 4, 4)
    carrier = PartitionedStagePlan(
        prefix=projected, prefix_noise=torch.zeros_like(projected), temporal=7, source_h=4, source_w=4
    )
    return carrier, exact


@pytest.mark.parametrize("conditioning", [False, True])
def test_exact_context_preserves_native_conditioning_audio_and_video_positions(native, conditioning):
    carrier, exact = plans(torch.randn(1, 24, 2, 8, 8))
    payload = {"seed": 13, "text_token_tags": torch.tensor([0, 1, 2])}
    if conditioning:
        keyframe = {"latent": torch.randn(1, 24, 1, 4, 4), "resolved_frame_index": 9}
        ref = {"kind": "image", "latent": torch.randn(1, 24, 1, 6, 8), "latent_h": 6, "latent_w": 8}
        payload.update(keyframes=[keyframe], refs=[ref], cond_video_latents=[keyframe["latent"], ref["latent"]])
    layout = native.PackedLayout(3, 7, 4, 4, 6, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    original_positions = layout.position_ids.clone()
    original_segments = list(layout.segments)
    prefix_before = exact.prefix.clone()
    extended, augmented, (first, last) = add_exact_prefix_visual_context(native, layout, payload, carrier, exact)
    assert torch.equal(torch.cat((extended.position_ids[:first], extended.position_ids[last:])), original_positions)
    assert last - first == 2 * 8 * 8 // 4
    # Native target-grid prefix positions, including the existing reference timeline origin.
    target_layout = native.PackedLayout(3, 7, 8, 8, 6, refs=payload.get("refs"))
    video_start = target_layout.segments[-1][0]
    assert torch.equal(
        extended.position_ids[first:last], target_layout.position_ids[video_start : video_start + last - first]
    )
    assert not bool(extended.img_update[(extended.img_pos >= first) & (extended.img_pos < last)].any())
    assert augmented["cond_video_latents"][-1] is exact.prefix
    assert augmented["text_token_tags"] is payload["text_token_tags"]
    assert payload.get("refs", []) == augmented["refs"][:-1]
    assert layout.segments == original_segments
    assert torch.equal(layout.position_ids, original_positions)
    assert torch.equal(exact.prefix, prefix_before)


def test_detail_in_projection_nullspace_reaches_native_condition_rows(native):
    # A real physical resize has a nullspace: these distinct exact prefixes
    # produce the same reduced carrier to numerical precision. A lossy-only
    # transformer cannot discriminate them; the new native context can.
    basis = torch.eye(64).reshape(64, 1, 1, 8, 8)
    projection = resize_spatial_5d_h3_patch_lattice(basis, 4, 4).reshape(64, 16).T
    _, _, vh = torch.linalg.svd(projection, full_matrices=True)
    detail = vh[-1].reshape(1, 1, 1, 8, 8).repeat(1, 24, 2, 1, 1)
    zero = torch.zeros_like(detail)
    a, exact_a = plans(zero)
    b, exact_b = plans(detail)
    assert float((a.prefix - b.prefix).abs().max()) < 1e-6
    layout = native.PackedLayout(3, 7, 4, 4, 6)
    _, payload_a, _ = add_exact_prefix_visual_context(native, layout, {"seed": 5}, a, exact_a)
    _, payload_b, _ = add_exact_prefix_visual_context(native, layout, {"seed": 5}, b, exact_b)
    model = SimpleNamespace(patch_size=(1, 2, 2))
    rows_a = native.MiniMaxH3Model._cond_video_rows(model, payload_a, "cpu")
    rows_b = native.MiniMaxH3Model._cond_video_rows(model, payload_b, "cpu")
    assert torch.allclose(rows_b - rows_a, native.patchify_video(detail) * native.VISUAL_COND_TIMESTEP, atol=1e-6)
    assert float((rows_b - rows_a).square().mean()) > 0.01


def test_exact_context_refuses_temporal_ownership_mismatch(native):
    carrier, exact = plans(torch.randn(1, 24, 2, 8, 8))
    mismatch = PartitionedStagePlan(
        prefix=exact.prefix[:, :, :1], prefix_noise=exact.prefix_noise[:, :, :1], temporal=7, source_h=4, source_w=4
    )
    with pytest.raises(RuntimeError, match="temporal ownership"):
        add_exact_prefix_visual_context(native, native.PackedLayout(3, 7, 4, 4, 6), {}, carrier, mismatch)
