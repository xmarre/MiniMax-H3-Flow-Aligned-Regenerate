"""Native conditioning preserves prefix information discarded by projection."""

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import resize_spatial_5d
from h3_flow_regenerate.partitioned_diagnostics import PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE
from h3_flow_regenerate.partitioned_stage import PartitionedStagePlan, partitioned_positions_with_audio_policy
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
    projected = resize_spatial_5d(prefix, 4, 4, mode="bicubic")
    carrier = PartitionedStagePlan(
        prefix=projected, prefix_noise=torch.zeros_like(projected), temporal=7, source_h=4, source_w=4
    )
    return carrier, exact


@pytest.mark.parametrize("conditioning", [False, True])
def test_exact_context_uses_native_reference_timeline_and_preserves_relative_av_time(native, conditioning):
    carrier, exact = plans(torch.randn(1, 24, 2, 8, 8))
    payload = {"seed": 13, "text_token_tags": torch.tensor([0, 1, 2])}
    if conditioning:
        keyframe = {
            "latent": torch.randn(1, 24, 1, 4, 4),
            "audio_latent": torch.randn(1, 128, 2),
            "resolved_frame_index": 9,
        }
        ref = {"kind": "image", "latent": torch.randn(1, 24, 1, 6, 8), "latent_h": 6, "latent_w": 8}
        av_ref = {
            "kind": "video_audio",
            "latent": torch.randn(1, 24, 2, 6, 8),
            "latent_h": 6,
            "latent_w": 8,
            "latent_t": 2,
            "ref_audio_t": 3,
        }
        payload.update(
            keyframes=[keyframe],
            refs=[ref, av_ref],
            cond_video_latents=[keyframe["latent"], ref["latent"], av_ref["latent"]],
            cond_audio_latents=[keyframe["audio_latent"], torch.randn(1, 128, 3)],
        )
    layout = native.PackedLayout(3, 7, 4, 4, 6, keyframes=payload.get("keyframes"), refs=payload.get("refs"))
    original_positions = layout.position_ids.clone()
    original_segments = list(layout.segments)
    prefix_before = exact.prefix.clone()
    extended, augmented, (first, last) = add_exact_prefix_visual_context(native, layout, payload, carrier, exact)
    expected = native.PackedLayout(3, 7, 4, 4, 6, keyframes=payload.get("keyframes"), refs=augmented["refs"])
    assert extended.segments == expected.segments
    assert torch.equal(extended.position_ids, expected.position_ids)
    assert last - first == 2 * 8 * 8 // 4
    # Audio/video and existing keyframes advance by the same native ref span.
    shift = native._ref_t_span(augmented["refs"][-1])
    assert shift > 0
    old_segments = [s for s in extended.segments if s[:2] != (first, last)]
    for before, after in zip(layout.segments, old_segments, strict=True):
        old = layout.position_ids[before[0] : before[1]]
        new = extended.position_ids[after[0] : after[1]]
        assert torch.equal(new[:, 1:], old[:, 1:])
        delta = shift if before[2] in ("audio", "video", "cond", "cond_audio") else 0.0
        torch.testing.assert_close(new[:, 0], old[:, 0] + delta, rtol=0, atol=1e-12)
    audio_first = next(s[0] for s in extended.segments if s[2] == "audio")
    video_first = extended.segments[-1][0]
    assert extended.position_ids[last - 1, 0] < extended.position_ids[video_first, 0]
    assert extended.position_ids[audio_first, 0] == extended.position_ids[video_first, 0]
    positions, policy = partitioned_positions_with_audio_policy(
        native,
        carrier,
        extended,
        audio_position_domain=PARTITIONED_AUDIO_POSITION_DOMAIN_SOURCE,
        stage_owner_generation=1,
    )
    assert policy.audio_temporal_equal
    assert torch.equal(positions[:, 0], extended.position_ids[:, 0])
    assert not bool(extended.img_update[(extended.img_pos >= first) & (extended.img_pos < last)].any())
    assert augmented["cond_video_latents"][-1] is exact.prefix
    assert augmented["text_token_tags"] is payload["text_token_tags"]
    assert augmented.get("cond_audio_latents") is payload.get("cond_audio_latents")
    assert payload.get("refs", []) == augmented["refs"][:-1]
    assert layout.segments == original_segments
    assert torch.equal(layout.position_ids, original_positions)
    assert torch.equal(exact.prefix, prefix_before)


def test_twelve_token_reference_uses_native_duration_and_target_phase(native):
    prefix = torch.zeros(1, 24, 12, 72, 54)
    exact = PartitionedStagePlan(
        prefix=prefix, prefix_noise=torch.zeros_like(prefix), temporal=62, source_h=50, source_w=38
    )
    projected = torch.zeros(1, 24, 12, 50, 38)
    carrier = PartitionedStagePlan(
        prefix=projected, prefix_noise=torch.zeros_like(projected), temporal=62, source_h=50, source_w=38
    )
    layout = native.PackedLayout(3, 62, 50, 38, 6)
    extended, payload, _ = add_exact_prefix_visual_context(native, layout, {}, carrier, exact)
    expected = native.PackedLayout(3, 62, 50, 38, 6, refs=payload["refs"])
    assert torch.equal(extended.position_ids, expected.position_ids)
    assert native._ref_t_span(payload["refs"][-1]) == 65.0
    assert extended.position_ids[extended.segments[-1][0], 0] - layout.position_ids[layout.segments[-1][0], 0] == 65.0


def test_detail_in_projection_nullspace_reaches_native_condition_rows(native):
    # The actual antialiased uniform image resize has a nullspace: these distinct exact prefixes
    # produce the same reduced carrier to numerical precision. A lossy-only
    # transformer cannot discriminate them; the new native context can.
    basis = torch.eye(64).reshape(64, 1, 1, 8, 8)
    projection = resize_spatial_5d(basis, 4, 4, mode="bicubic").reshape(64, 16).T
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


def test_exact_visual_detail_influences_suffix_through_real_vdn(native, monkeypatch):
    import inspect

    import test_partitioned_native_attention as attention_fixture

    from h3_flow_regenerate import partitioned_transformer as transform
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStageRuntime

    fixture = attention_fixture._without_sol(monkeypatch)
    cases = []
    original_case = fixture._Case

    def record_case(*args, **kwargs):
        case = original_case(*args, **kwargs)
        cases.append(case)
        return case

    monkeypatch.setattr(fixture, "_Case", record_case)
    # Reuse the native Core + real learned VDN branch fixture, including its
    # ordinary uniform/domain parity checks. No VDN arithmetic is substituted.
    fixture.test_real_vdn_readout_keeps_uniform_stream_outputs_and_isolation(
        monkeypatch, retain=True, anchors="rows", same_grid=False
    )
    case = cases[0]
    state = inspect.getclosurevars(case.wrappers[0]).nonlocals["state"]
    state.cfg["linear_enabled"] = True
    h, w = case.geometry.target_hw
    sh, sw = case.geometry.source_h, case.geometry.source_w
    basis = torch.eye(h * w).reshape(h * w, 1, 1, h, w)
    projection = resize_spatial_5d(basis, sh, sw, mode="bicubic").reshape(h * w, sh * sw).T
    _, _, vh = torch.linalg.svd(projection, full_matrices=True)
    detail = vh[-1].reshape(1, 1, 1, h, w).repeat(1, 24, case.owner.prefix_t, 1, 1)
    changed = PartitionedStagePlan(
        prefix=case.owner.prefix + detail,
        prefix_noise=case.owner.prefix_noise,
        temporal=case.owner.temporal,
        source_h=sh,
        source_w=sw,
    )
    source_prefix = resize_spatial_5d(case.owner.prefix, sh, sw, mode="bicubic")
    assert torch.allclose(resize_spatial_5d(changed.prefix, sh, sw, mode="bicubic"), source_prefix, atol=1e-6)
    source_video = torch.cat(
        (source_prefix, case.carrier(case.video), fixture.target_band_tail(case.video, case.geometry)), 2
    )
    plan = PartitionedStagePlan(source_prefix, case.owner.temporal, sh, sw, torch.zeros_like(source_prefix))
    mask = torch.ones(1, 1, case.owner.temporal, sh, sw)
    mask[:, :, : case.owner.prefix_t] = 0

    def execute(exact):
        metrics = H3FlowMetrics()
        runtime = PartitionedStageRuntime(plan=plan, metrics=metrics, exact_prefix_visual_context=exact)
        out = fixture._forward(
            case.dm,
            # The production node puts Flow first, before VDN captures the
            # native layout. Exercise that actual wrapper order here.
            [transform.partitioned_diffusion_wrapper, *case.wrappers],
            source_video,
            case.audio,
            case.context,
            {PARTITIONED_STAGE_KEY: runtime, "h3_flow_stage": "low"},
            mask,
            case.audio_mask,
        )
        assert metrics.counters["partitioned_vdn_uniform_linear_calls"] == len(case.dm.blocks)
        assert torch.isfinite(out[0]).all() and torch.isfinite(out[1]).all()
        return out[0][:, :, case.owner.prefix_t :]

    # The generated input, projected prefix and audio are identical. Only exact
    # detail changes, and it reaches the suffix through the shared deep context.
    baseline = execute(case.owner)
    assert not torch.equal(execute(changed), baseline)
    assert torch.equal(execute(case.owner), baseline)
