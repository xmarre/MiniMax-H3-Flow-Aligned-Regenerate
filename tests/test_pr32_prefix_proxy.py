from __future__ import annotations

import torch

from h3_flow_regenerate import runtime
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.pr32_prefix_proxy import (
    _capture_candidate,
    _proxy_packed_model_input,
    flow_predict_wrapper_with_prefix_proxy,
)


def _eligible_metrics():
    return {
        "suffix_representation_bridge_tone_bias_accepted": True,
        "suffix_representation_bridge_geometry_accepted": False,
        "suffix_representation_bridge_geometry_reason": "native_boundary_replacement_geometry_consistent",
    }


def test_prefix_proxy_replaces_only_protected_model_representation():
    torch.manual_seed(1234)
    prefix_t = 2
    video_shape = (1, 24, 5, 8, 10)
    audio_shape = (1, 32, 2, 7)
    learned = torch.randn(video_shape, dtype=torch.float32)
    exact = learned[:, :, :prefix_t] + 0.25
    tone_bias = torch.linspace(-0.1, 0.1, 24).view(1, 24, 1, 1, 1)
    corrected = learned.clone()
    corrected[:, :, prefix_t:] += tone_bias

    payload, report = _capture_candidate(learned, exact, corrected, _eligible_metrics())
    assert payload is not None
    assert report["suffix_representation_bridge_high_prefix_proxy_ready"] is True
    assert report["suffix_representation_bridge_high_prefix_proxy_reason"] == "tone_aligned_learned_prefix_ready"

    noise = torch.randn_like(exact)
    aug = 0.999
    model_video = torch.randn(video_shape, dtype=torch.float32)
    model_video[:, :, :prefix_t] = aug * exact + (1.0 - aug) * noise
    audio = torch.randn(audio_shape, dtype=torch.float32)
    packed, shapes = pack_streams((model_video, audio))
    before = packed.clone()
    contract = {"prefix_t": prefix_t, "shapes": tuple(shapes)}

    proxied, telemetry = _proxy_packed_model_input(
        packed,
        contract,
        payload,
        visual_cond_timestep=aug,
    )
    out_video, out_audio = unpack_streams(proxied, shapes)
    expected_proxy_clean = learned[:, :, :prefix_t] + tone_bias
    expected_prefix = aug * expected_proxy_clean + (1.0 - aug) * noise

    # Re-associating the float32 expression can differ by one ULP. The proxy
    # algebra must still reproduce native-inpaint semantics to float32 precision.
    torch.testing.assert_close(out_video[:, :, :prefix_t], expected_prefix, rtol=0.0, atol=1e-6)
    assert torch.equal(out_video[:, :, prefix_t:], model_video[:, :, prefix_t:])
    assert torch.equal(out_audio, audio)
    assert torch.equal(packed, before)
    assert telemetry["prefix_proxy_suffix_delta_rms"] == 0.0
    assert telemetry["prefix_proxy_audio_delta_rms"] == 0.0
    assert telemetry["prefix_proxy_visual_cond_timestep"] == aug


def test_prefix_proxy_refuses_geometry_rebase_confound():
    learned = torch.zeros((1, 24, 4, 8, 8), dtype=torch.float32)
    exact = torch.zeros((1, 24, 2, 8, 8), dtype=torch.float32)
    corrected = learned.clone()
    corrected[:, :, 2:] += 0.1
    metrics = _eligible_metrics()
    metrics["suffix_representation_bridge_geometry_accepted"] = True

    payload, report = _capture_candidate(learned, exact, corrected, metrics)

    assert payload is None
    assert report["suffix_representation_bridge_high_prefix_proxy_ready"] is False
    assert report["suffix_representation_bridge_high_prefix_proxy_reason"] == "geometry_rebase_would_confound_proxy"


def test_pr32_overlay_is_the_registered_runtime_predict_wrapper():
    assert runtime.flow_predict_wrapper is flow_predict_wrapper_with_prefix_proxy
    assert getattr(runtime.flow_predict_wrapper, "_h3_pr32_prefix_proxy", False) is True
