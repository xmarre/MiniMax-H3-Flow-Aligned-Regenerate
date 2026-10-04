from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.audio_guided_overlap import (
    AUDIO_GUIDED_OVERLAP_ENV,
    apply_audio_guided_overlap_mask,
    configured_audio_guided_overlap_ticks,
)
from h3_flow_regenerate.comfy_compat import _canonicalize_exact_masked_output
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY,
    PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY,
    apply_partitioned_diagnostic_controls,
    resolve_partitioned_audio_guided_overlap_ticks,
    resolve_partitioned_video_guided_overlap_tokens,
)
from h3_flow_regenerate.partitioned_node import H3PartitionedExactPrefixDiagnosticHandoff
from h3_flow_regenerate.partitioned_scheduler import _partitioned_high_video_overlap_mask


def _case(dtype=torch.float32):
    video = torch.randn(2, 24, 17, 8, 12, dtype=dtype)
    audio = torch.randn(2, 32, 2, 70, dtype=dtype)
    latent, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    audio_mask = torch.ones_like(audio)
    video_mask[:, :, :12] = 0
    audio_mask[..., :64] = 0
    return latent, list(shapes), pack_streams((video_mask, audio_mask))[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize("requested", [8, 12, 10**30])
def test_wider_video_overlap_is_high_only_and_restores_the_entire_original_prefix(dtype, requested):
    latent, shapes, exact_mask = _case(dtype)
    original = exact_mask.clone()
    metrics = H3FlowMetrics()
    options = {PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY: requested}

    high_mask = _partitioned_high_video_overlap_mask(exact_mask, exact_mask, shapes, options, metrics)

    video, audio = unpack_streams(high_mask, shapes)
    old_video, old_audio = unpack_streams(original, shapes)
    used = min(requested, 12)
    start = 12 - used
    numerators = (
        [29, 57, 86, 114, 143, 171, 200, 228] if used == 8 else [20, 40, 60, 79, 99, 119, 138, 158, 178, 197, 217, 237]
    )
    expected = torch.tensor(numerators, dtype=dtype) / 256
    torch.testing.assert_close(video[0, 0, start:12, 0, 0], expected)
    torch.testing.assert_close(video[-1, -1, start:12, -1, -1], expected)
    assert torch.equal(video[:, :, :start], old_video[:, :, :start])
    assert torch.equal(video[:, :, 12:], old_video[:, :, 12:])
    assert torch.equal(audio, old_audio)
    assert torch.equal(exact_mask, original)
    fields = metrics.events[-1].fields
    assert fields["requested_tokens"] == requested
    assert fields["applied_tokens"] == used
    assert fields["hard_prefix_tokens"] == start
    assert fields["width_limited_by_prefix"] is (requested > 12)
    assert fields["stage"] == "high"
    assert fields["low_probe_sampler_mask_unchanged"] is True
    assert fields["final_exact_prefix_restore"] is True

    sampled = latent + high_mask
    generated_before_restore = unpack_streams(sampled, shapes)[0][:, :, 12:].clone()
    restored, stats = _canonicalize_exact_masked_output(sampled, latent, original)
    restored_video, restored_audio = unpack_streams(restored, shapes)
    source_video, source_audio = unpack_streams(latent, shapes)
    assert torch.equal(restored_video[:, :, :12], source_video[:, :, :12])
    assert torch.equal(restored_audio[..., :64], source_audio[..., :64])
    assert torch.equal(restored_video[:, :, 12:], generated_before_restore)
    assert stats["final_exact"] is True


@pytest.mark.parametrize("requested", [32, 64, 10**30])
def test_wider_audio_overlap_uses_available_prefix_and_preserves_video(requested):
    latent, shapes, exact_mask = _case()
    original = exact_mask.clone()

    runtime_mask, report = apply_audio_guided_overlap_mask(exact_mask, shapes, ticks=requested)

    video, audio = unpack_streams(runtime_mask, shapes)
    old_video, old_audio = unpack_streams(original, shapes)
    used = min(requested, 64)
    start = 64 - used
    assert report["applied"] is True
    assert report["requested_ticks"] == requested
    assert report["applied_ticks"] == used
    assert report["hard_prefix_ticks"] == start
    assert report["width_limited_by_prefix"] is (requested > 64)
    assert len(report["ramp_values"]) == used
    assert torch.equal(video, old_video)
    assert torch.equal(audio[..., :start], old_audio[..., :start])
    assert torch.equal(audio[..., 64:], old_audio[..., 64:])
    assert bool((audio[..., start:64] > 0).all())
    assert bool((audio[..., start:64] < 1).all())
    assert torch.equal(exact_mask, original)
    restored, stats = _canonicalize_exact_masked_output(latent + runtime_mask, latent, original)
    restored_audio = unpack_streams(restored, shapes)[1]
    source_audio = unpack_streams(latent, shapes)[1]
    assert torch.equal(restored_audio[..., :64], source_audio[..., :64])
    assert stats["final_exact"] is True


def test_overlap_node_and_configuration_accept_widths_above_former_ceilings(monkeypatch):
    required = H3PartitionedExactPrefixDiagnosticHandoff.INPUT_TYPES()["required"]
    assert "max" not in required["video_guided_overlap_tokens"][1]
    assert "max" not in required["audio_guided_overlap_ticks"][1]
    model = SimpleNamespace(model_options={"transformer_options": {}})
    apply_partitioned_diagnostic_controls(
        model,
        H3FlowMetrics(),
        vdn_linear_diagnostic="normal",
        audio_guided_overlap_ticks=32,
        video_guided_overlap_tokens=8,
    )
    assert model.model_options[PARTITIONED_VIDEO_GUIDED_OVERLAP_TOKENS_KEY] == 8
    assert model.model_options[PARTITIONED_AUDIO_GUIDED_OVERLAP_TICKS_KEY] == 32
    assert resolve_partitioned_video_guided_overlap_tokens(model.model_options) == (8, "diagnostic_node")
    assert resolve_partitioned_audio_guided_overlap_ticks(model.model_options) == (32, "diagnostic_node")
    monkeypatch.setenv(AUDIO_GUIDED_OVERLAP_ENV, "32")
    assert configured_audio_guided_overlap_ticks() == 32
