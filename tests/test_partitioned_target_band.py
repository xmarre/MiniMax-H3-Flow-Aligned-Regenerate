"""Target-band continuation: geometry, sampler-state layout and handoff helpers."""

from __future__ import annotations

import math

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.partitioned_band import (
    pack_target_band_video,
    target_band_padding_max_abs,
    target_band_source_view,
    target_band_tail,
    target_band_target_preview,
)
from h3_flow_regenerate.partitioned_prefix import (
    PARTITIONED_NATIVE_CARRIER_TARGET,
    PartitionedExactPrefixPlan,
    validate_partitioned_contract,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PartitionedPreflightUnsupported,
    _apply_target_band_head_dc_bridge,
    _target_band_low_stage_inputs,
    _target_band_source_views,
    _validate_target_band_configuration,
)
from h3_flow_regenerate.partitioned_stage import (
    PartitionedStagePlan,
    PartitionedTargetBandGeometry,
    partitioned_mod_segments,
)
from h3_flow_regenerate.partitioned_transformer import _vdn_external_contract

# 2 protected + 2 band tokens on an 8x12 target grid, 3 tail tokens on 4x6.
GEOMETRY = PartitionedTargetBandGeometry(
    protected_t=2, band_t=2, temporal=7, source_h=4, source_w=6, target_h=8, target_w=12
)


@pytest.mark.parametrize("rows", [None, 6, 999])
def test_source_carrier_rejects_an_undeclared_native_row_count(rows):
    plan = PartitionedExactPrefixPlan(7, 5, 3, 2, 3, 3, 4)
    contract = dict(plan.to_contract(), native_carrier_rows_per_frame=rows)
    with pytest.raises(ValueError, match="requires native_carrier_grid"):
        validate_partitioned_contract(contract, sequence_rows=plan.sequence_rows)


@pytest.mark.parametrize("rows", [None, 12.0, "12", True])
def test_target_carrier_requires_integer_native_rows(rows):
    plan = PartitionedExactPrefixPlan(7, 5, 3, 2, 3, 3, 4, native_carrier_grid=PARTITIONED_NATIVE_CARRIER_TARGET)
    contract = dict(plan.to_contract(), native_carrier_rows_per_frame=rows)
    with pytest.raises(ValueError, match="native_carrier_rows_per_frame must be an integer"):
        validate_partitioned_contract(contract, sequence_rows=plan.sequence_rows)


def test_geometry_rows_match_the_partition_contract():
    g = GEOMETRY
    assert (g.head_t, g.prefix_t) == (4, 4)
    assert (g.target_rows, g.source_rows) == (24, 6)
    assert g.protected_rows == 48
    assert g.prefix_rows == 96
    assert g.suffix_rows == 18
    assert g.partitioned_rows == 114
    assert g.native_rows == 168
    plan = PartitionedExactPrefixPlan(
        video_start=5,
        temporal=g.temporal,
        prefix_t=g.prefix_t,
        source_grid_h=2,
        source_grid_w=3,
        target_grid_h=4,
        target_grid_w=6,
        native_carrier_grid=PARTITIONED_NATIVE_CARRIER_TARGET,
    )
    assert plan.sequence_rows == 5 + g.partitioned_rows
    contract = plan.to_contract()
    assert contract["native_carrier_grid"] == "target"
    assert contract["native_carrier_rows_per_frame"] == 24
    assert validate_partitioned_contract(contract, sequence_rows=plan.sequence_rows) == plan
    assert _vdn_external_contract(plan)["native_carrier_rows_per_frame"] == 24
    legacy = PartitionedExactPrefixPlan(5, g.temporal, g.prefix_t, 2, 3, 4, 6)
    assert "native_carrier_grid" not in legacy.to_contract()
    assert "native_carrier_rows_per_frame" not in _vdn_external_contract(legacy)
    assert legacy.to_contract()["semantic_digest"] != contract["semantic_digest"]


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"band_t": 5}, "at least one generated token"),
        ({"band_t": 0}, "positive integer"),
        ({"source_h": 8, "source_w": 12}, "strictly reduced"),
        ({"source_h": 10}, "fit inside"),
        ({"source_w": 5}, "patch-safe"),
    ],
)
def test_geometry_rejects_unrealizable_layouts(overrides, error):
    fields = dict(protected_t=2, band_t=2, temporal=7, source_h=4, source_w=6, target_h=8, target_w=12)
    fields.update(overrides)
    with pytest.raises(ValueError, match=error):
        PartitionedTargetBandGeometry(**fields)


def test_tail_rows_follow_core_patchify_order_inside_the_storage_window():
    """Gathering native target rows at the tail index equals patchifying the stored windows."""
    g = GEOMETRY
    video = torch.randn(1, 24, g.temporal, g.target_h, g.target_w)
    # Core's patchify: [B,C,T,H,W] -> rows ordered (t, h/2, w/2).
    b, c, t, h, w = video.shape
    native = (
        video.reshape(b, c, t, 1, h // 2, 2, w // 2, 2)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(t * (h // 2) * (w // 2), c * 4)
    )
    tail = target_band_tail(video, g)
    tb, tc, tt, th, tw = tail.shape
    expected = (
        tail.reshape(tb, tc, tt, 1, th // 2, 2, tw // 2, 2)
        .permute(0, 2, 4, 6, 1, 3, 5, 7)
        .reshape(tt * (th // 2) * (tw // 2), tc * 4)
    )
    assert torch.equal(native[g.tail_native_rows()], expected)


def test_pack_views_and_padding_round_trip():
    g = GEOMETRY
    head = torch.randn(1, 24, g.head_t, g.target_h, g.target_w)
    tail = torch.randn(1, 24, g.temporal - g.head_t, g.source_h, g.source_w)
    packed = pack_target_band_video(head, tail, g)
    assert packed.shape == (1, 24, g.temporal, g.target_h, g.target_w)
    assert torch.equal(packed[:, :, : g.head_t], head)
    assert torch.equal(target_band_tail(packed, g), tail)
    assert target_band_padding_max_abs(packed, g) == 0.0

    view = target_band_source_view(packed, g)
    assert view.shape == (1, 24, g.temporal, g.source_h, g.source_w)
    assert torch.equal(view[:, :, g.head_t :], tail)

    preview = target_band_target_preview(packed, g)
    assert preview.shape == packed.shape
    assert torch.equal(preview[:, :, : g.head_t], head)

    packed[:, :, -1, -1, -1] = 3.0
    assert target_band_padding_max_abs(packed, g) == 3.0
    with pytest.raises(ValueError, match="does not match the target-band layout"):
        target_band_source_view(packed[:, :, :-1], g)


def _row_labels(mask_video):
    """Per-row protected(0)/generated(1) labels the way Core derives them from a mask."""
    _b, _c, t, h, w = mask_video.shape
    return mask_video[0, 0].reshape(t, h // 2, 2, w // 2, 2).amax(dim=(2, 4)).reshape(-1)


def test_band_modulation_rows_keep_prefix_band_and_tail_labels():
    g = GEOMETRY
    mask = pack_target_band_video(
        torch.cat((torch.zeros(1, 1, 2, 8, 12), torch.ones(1, 1, 2, 8, 12)), dim=2),
        torch.ones(1, 1, 3, 4, 6),
        g,
    )
    labels = _row_labels(mask)
    native = [(0, 5, 3), (5, 5 + g.native_rows, labels)]
    (text, video) = partitioned_mod_segments(native, g, 5, 5 + g.native_rows)
    assert text == (0, 5, 3)
    start, stop, row = video
    assert (start, stop) == (5, 5 + g.partitioned_rows)
    assert row.numel() == g.partitioned_rows
    assert torch.equal(row[: g.protected_rows], torch.zeros(g.protected_rows))
    assert torch.equal(row[g.protected_rows :], torch.ones(g.partitioned_rows - g.protected_rows))

    # Padding is protected, so a padding label leaking into the tail is rejected.
    leaking = labels.clone()
    leaking[g.tail_native_rows()[0]] = 0
    with pytest.raises(RuntimeError, match="uniform protected-prefix and generated"):
        partitioned_mod_segments([(5, 5 + g.native_rows, leaking)], g, 5, 5 + g.native_rows)


def _packed(video, audio):
    return pack_streams((video, audio))


def test_low_stage_inputs_place_band_noise_tail_noise_and_protected_padding():
    g = GEOMETRY
    target_shapes = [(1, 24, g.temporal, g.target_h, g.target_w), (1, 32, 2, 9)]
    source_shapes = [(1, 24, g.temporal, g.source_h, g.source_w), (1, 32, 2, 9)]
    noise = _packed(torch.randn(*target_shapes[0]), torch.randn(*target_shapes[1]))[0]
    source_noise = torch.randn(*source_shapes[0])
    latent = _packed(torch.randn(*target_shapes[0]), torch.randn(*target_shapes[1]))[0]
    source_latent = _packed(torch.randn(*source_shapes[0]), unpack_streams(latent, target_shapes)[1])[0]
    target_mask_video = torch.ones(*target_shapes[0])
    target_mask_video[:, :, :2] = 0
    target_mask_audio = torch.ones(*target_shapes[1])
    mask = _packed(target_mask_video, target_mask_audio)[0]
    source_mask = _packed(torch.ones(*source_shapes[0]), target_mask_audio)[0]

    low_noise, low_latent, low_mask = _target_band_low_stage_inputs(
        g,
        noise=noise,
        source_video_noise=source_noise,
        latent_image=latent,
        source_latent_image=source_latent,
        denoise_mask=mask,
        source_mask=source_mask,
        target_shapes=target_shapes,
        source_shapes=source_shapes,
        seed=11,
    )
    noise_video, noise_audio = unpack_streams(low_noise, target_shapes)
    latent_video, latent_audio = unpack_streams(low_latent, target_shapes)
    mask_video, mask_audio = unpack_streams(low_mask, target_shapes)
    assert torch.equal(noise_audio, unpack_streams(noise, target_shapes)[1])
    assert torch.equal(latent_audio, unpack_streams(latent, target_shapes)[1])
    assert torch.equal(mask_audio, target_mask_audio)
    # Tail noise is the reduced-grid progressive noise; padding carries nothing.
    assert torch.equal(target_band_tail(noise_video, g), source_noise[:, :, g.head_t :])
    assert target_band_padding_max_abs(noise_video, g) == 0.0
    assert target_band_padding_max_abs(latent_video, g) == 0.0
    # Head latents are the caller's target-grid values, untouched.
    assert torch.equal(latent_video[:, :, : g.head_t], unpack_streams(latent, target_shapes)[0][:, :, : g.head_t])
    # Mask: prefix protected, band and tail windows generated, padding protected.
    assert torch.count_nonzero(mask_video[:, :, :2]) == 0
    assert bool((mask_video[:, :, 2 : g.head_t] == 1).all())
    assert bool((target_band_tail(mask_video, g) == 1).all())
    assert target_band_padding_max_abs(mask_video, g) == 0.0

    partial = target_mask_video.clone()
    partial[:, :, 3, 0, 0] = 0.5
    with pytest.raises(PartitionedPreflightUnsupported, match="fully generated band"):
        _target_band_low_stage_inputs(
            g,
            noise=noise,
            source_video_noise=source_noise,
            latent_image=latent,
            source_latent_image=source_latent,
            denoise_mask=_packed(partial, target_mask_audio)[0],
            source_mask=source_mask,
            target_shapes=target_shapes,
            source_shapes=source_shapes,
            seed=11,
        )


def test_source_views_keep_identity_band_tensors_and_reject_padding_writes():
    g = GEOMETRY
    target_shapes = [(1, 24, g.temporal, g.target_h, g.target_w), (1, 32, 2, 9)]
    source_shapes = [(1, 24, g.temporal, g.source_h, g.source_w), (1, 32, 2, 9)]
    head = torch.randn(1, 24, g.head_t, g.target_h, g.target_w)
    raw = pack_target_band_video(head, torch.randn(1, 24, 3, 4, 6), g)
    raw[:, :, -1, -1, -1] = 0.25  # stochastic samplers may leave noise in protected padding
    clean = pack_target_band_video(head * 0.5, torch.randn(1, 24, 3, 4, 6), g)
    audio = torch.randn(*target_shapes[1])
    source_raw, source_clean, band_raw, band_clean, receipt = _target_band_source_views(
        _packed(raw, audio)[0],
        _packed(clean, audio)[0],
        g,
        target_shapes=target_shapes,
        source_shapes=source_shapes,
    )
    assert torch.equal(band_raw, raw[:, :, 2:4])
    assert torch.equal(band_clean, clean[:, :, 2:4])
    raw_view, raw_audio = unpack_streams(source_raw, source_shapes)
    assert torch.equal(raw_audio, audio)
    assert torch.equal(raw_view[:, :, g.head_t :], target_band_tail(raw, g))
    assert torch.equal(unpack_streams(source_clean, source_shapes)[0][:, :, g.head_t :], target_band_tail(clean, g))
    assert receipt["raw_padding_max_abs"] == 0.25
    assert receipt["clean_padding_max_abs"] == 0.0
    assert receipt["low_probe_video_rows"] == g.partitioned_rows

    clean[:, :, -1, -1, -1] = 1.0
    with pytest.raises(RuntimeError, match="outside its reduced-grid storage windows"):
        _target_band_source_views(
            _packed(raw, audio)[0],
            _packed(clean, audio)[0],
            g,
            target_shapes=target_shapes,
            source_shapes=source_shapes,
        )


@pytest.mark.parametrize("enabled", [True, False])
def test_head_dc_bridge_corrects_only_the_first_transferred_tail_token(enabled):
    g = GEOMETRY
    torch.manual_seed(5)
    provider_native = torch.randn(1, 24, g.temporal, g.target_h, g.target_w)
    spliced = provider_native.clone()
    band_clean = torch.randn(1, 24, g.band_t, g.target_h, g.target_w) + 0.4
    spliced[:, :, g.protected_t : g.head_t] = band_clean
    exact_prefix = torch.randn(1, 24, g.protected_t, g.target_h, g.target_w)
    state = torch.randn_like(spliced)
    sigma = 0.8

    mapped, corrected, metrics = _apply_target_band_head_dc_bridge(
        state, spliced, provider_native, exact_prefix, g, sigma=sigma, enabled=enabled
    )
    assert metrics["suffix_dc_bridge_boundary"] == "target_band_head"
    assert metrics["suffix_dc_bridge_prefix_t"] == g.head_t
    untouched = [t for t in range(g.temporal) if t != g.head_t]
    assert torch.equal(mapped[:, :, untouched], state[:, :, untouched])
    assert torch.equal(corrected[:, :, untouched], spliced[:, :, untouched])
    if not enabled:
        assert torch.equal(mapped, state)
        assert metrics["suffix_dc_bridge_corrected_tokens"] == 0
        return
    # The offset is the provider's channel-mean error on the band's last token.
    expected = band_clean[:, :, -1].mean(dim=(-2, -1), keepdim=True) - provider_native[:, :, g.head_t - 1].mean(
        dim=(-2, -1), keepdim=True
    )
    torch.testing.assert_close(
        corrected[:, :, g.head_t] - spliced[:, :, g.head_t], expected.expand_as(spliced[:, :, 0])
    )
    torch.testing.assert_close(
        mapped[:, :, g.head_t] - state[:, :, g.head_t], (1.0 - sigma) * expected.expand_as(state[:, :, 0])
    )
    assert metrics["suffix_dc_bridge_corrected_tokens"] == 1
    assert math.isclose(metrics["suffix_dc_bridge_delta_rms"], float(expected.square().mean().sqrt()), rel_tol=1e-5)


@pytest.mark.parametrize(
    ("override", "error"),
    [
        ({"handoff_transfer_control": "bicubic_same_source_control"}, "learned_3d"),
        ({"vdn_temporal_carrier_policy": "destination_grid_stencil_v1"}, "vdn_temporal_carrier_policy"),
        ({"prefix_transformer_context": "source_carrier_uniform"}, "prefix_transformer_context"),
        ({"low_probe_execution_source": "source_carrier_uniform_only"}, "low_probe_execution_source"),
        ({"guidance_trajectory_source": "source_carrier_uniform_shadow"}, "main partitioned sources"),
        ({"residual_mode": "apply"}, "frame_gauge_residual_mode"),
        ("witness", "capture_boundary_witness"),
    ],
)
def test_runtime_configuration_rejects_unimplemented_band_combinations(override, error):
    from h3_flow_regenerate.boundary_witness import WITNESS_DIRECTORY_OPTION

    fields = dict(
        handoff_transfer_control="learned_3d",
        vdn_temporal_carrier_policy="native_grid_then_map_v1",
        prefix_transformer_context="exact_target_partitioned",
        low_probe_execution_source="main_then_shadow",
        audio_handoff_source="main_partitioned",
        av_handoff_source="main_partitioned",
        guidance_trajectory_source="main_exact_partitioned",
        residual_mode="off",
        model_options={},
    )
    _validate_target_band_configuration(**fields)
    _validate_target_band_configuration(**dict(fields, residual_mode="measure"))
    if override == "witness":
        override = {"model_options": {WITNESS_DIRECTORY_OPTION: "/tmp/witness"}}
    fields.update(override)
    with pytest.raises(PartitionedPreflightUnsupported, match=error):
        _validate_target_band_configuration(**fields)


def test_band_geometry_must_match_its_protected_prefix_owner():
    from types import SimpleNamespace

    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStageRuntime
    from h3_flow_regenerate.partitioned_transformer import partitioned_diffusion_wrapper

    prefix = torch.randn(1, 24, 3, 8, 12)
    owner = PartitionedStagePlan(prefix, 7, 4, 6, torch.randn_like(prefix))
    runtime = PartitionedStageRuntime(plan=owner, metrics=H3FlowMetrics(), target_band=GEOMETRY)
    executor = SimpleNamespace(class_obj=SimpleNamespace(blocks=[object()]))
    with pytest.raises(RuntimeError, match="does not match the protected-prefix stage plan"):
        partitioned_diffusion_wrapper(
            executor,
            [torch.zeros(1, 24, 7, 8, 12), torch.zeros(1, 32, 2, 9)],
            torch.tensor([500.0]),
            torch.zeros(1, 3, 8),
            transformer_options={PARTITIONED_STAGE_KEY: runtime},
        )
