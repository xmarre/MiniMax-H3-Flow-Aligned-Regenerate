"""Native-prefix work budget and model-local opt-in; no trained quality claim."""

import sys
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import source_prefix_projection as projection
from h3_flow_regenerate.decode_context import video_latent_fingerprint


@pytest.fixture
def vae(monkeypatch):
    from test_prefix_projection_audit import FakeVAE

    class Native:
        vae_ratio, vae_ratio_t, clip_length, token_drop, token_overlap = 16, 4, 17, 3, 2

    value = FakeVAE()
    value.first_stage_model = Native()
    monkeypatch.setitem(sys.modules, "comfy.ldm.minimax.vae", SimpleNamespace(MiniMaxH3VideoVAE=Native))

    class Format:
        def process_out(self, x):
            return x

    monkeypatch.setitem(sys.modules, "comfy.latent_formats", SimpleNamespace(MiniMaxH3Video=Format))
    return value


def test_roundtrip_uses_one_decode_and_one_encode_and_owned_prefix(vae):
    prefix = torch.rand(1, 24, 12, 8, 10) * 0.2 + 0.2
    original = prefix.clone()
    result, receipt = projection.project_source_prefix(vae, prefix, 6, 8)
    assert result.shape == (1, 24, 12, 6, 8)
    assert torch.equal(prefix.view(torch.int32), original.view(torch.int32))
    assert len(vae.decode_calls) == len(vae.encode_calls) == 1
    assert vae.encode_calls[0].shape == (39, 96, 128, 3)
    assert receipt["projected_prefix"] == video_latent_fingerprint(result)
    assert receipt["authoritative_prefix"] == video_latent_fingerprint(prefix)
    assert receipt["extra_h3_nfe"] == 0
    assert receipt["extra_vae_decode_calls"] == receipt["extra_vae_encode_calls"] == 1
    assert set(receipt["timings_ms"]) == {"target_prefix_decode", "source_rgb_encode"}
    assert all(v >= 0 for v in receipt["timings_ms"].values())


def test_each_invocation_encodes_fresh_prefix_and_does_not_cache(vae):
    p = torch.full((1, 24, 12, 8, 10), 0.3)
    a, _ = projection.project_source_prefix(vae, p, 6, 8)
    b, _ = projection.project_source_prefix(vae, p + 0.1, 6, 8)
    assert not torch.equal(a, b)
    assert len(vae.decode_calls) == len(vae.encode_calls) == 2


def test_wrong_temporal_phase_or_encoder_output_stops_candidate(vae):
    p = torch.full((1, 24, 12, 8, 10), 0.3)
    with pytest.raises(ValueError, match="5k\\+2"):
        projection.project_source_prefix(vae, p[:, :, :11], 6, 8)
    assert not vae.decode_calls and not vae.encode_calls
    vae.encoded_t_delta = 1
    with pytest.raises(ValueError, match="temporal ownership"):
        projection.project_source_prefix(vae, p, 6, 8)


def test_native_decode_failure_propagates_without_latent_bicubic_retry(vae, monkeypatch):
    def fail(*_args, **_kwargs):
        raise RuntimeError("decode failed")

    monkeypatch.setattr(vae, "decode", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        projection.project_source_prefix(vae, torch.zeros(1, 24, 12, 8, 10), 6, 8)
    assert not vae.encode_calls


def test_configuration_is_model_local_and_default_releases_stale_vae(vae):
    old = {"keep": object()}
    model = SimpleNamespace(model_options=old)
    projection.configure_source_prefix_projection(model, "vae_rgb_roundtrip", vae, "progressive_uniform_source")
    assert projection.SOURCE_PREFIX_PROJECTION_KEY not in old
    configured = model.model_options
    assert configured[projection.SOURCE_PREFIX_PROJECTION_KEY].vae is vae
    projection.configure_source_prefix_projection(model, "latent_bicubic", None, "same_grid_target_control")
    assert projection.SOURCE_PREFIX_PROJECTION_KEY not in model.model_options
    assert configured[projection.SOURCE_PREFIX_PROJECTION_KEY].vae is vae
    assert model.model_options["keep"] is old["keep"]


def test_configuration_rejects_invalid_mode_or_missing_native_vae(vae):
    model = SimpleNamespace(model_options={})
    with pytest.raises(ValueError, match="unsupported source_prefix_projection"):
        projection.configure_source_prefix_projection(model, "other", vae, "progressive_uniform_source")
    with pytest.raises(ValueError, match="Connect"):
        projection.configure_source_prefix_projection(model, "vae_rgb_roundtrip", None, "progressive_uniform_source")
    with pytest.raises(ValueError, match="spatial stage"):
        projection.configure_source_prefix_projection(model, "vae_rgb_roundtrip", vae, "same_grid_target_control")
    with pytest.raises(ValueError, match="native MiniMax"):
        projection.configure_source_prefix_projection(
            model, "vae_rgb_roundtrip", object(), "progressive_uniform_source"
        )
    assert model.model_options == {}
