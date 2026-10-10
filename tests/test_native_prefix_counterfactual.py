"""Structural tests for saved-source, decoder-only H3 prefix counterfactual."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate.decode_context import video_latent_fingerprint
from h3_flow_regenerate.geometry import resize_spatial_5d
from h3_flow_regenerate.local_boundary_audit import STAGES
from h3_flow_regenerate.native_prefix_counterfactual import (
    audit_native_prefix_counterfactual,
    make_prefix_counterfactual,
)

ROIS = '{"left":[0.0,0.0,0.48,0.8],"right":[0.52,0.0,1.0,0.8]}'


def _save(directory, manifest, name, video):
    raw = video.contiguous().numpy().tobytes()
    (directory / f"{name}.bin").write_bytes(raw)
    manifest["tensor_bytes"][name] = {
        "file": f"{name}.bin",
        "shape": list(video.shape),
        "dtype": "torch.float32",
        "byte_order": "native_torch_contiguous",
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@pytest.fixture
def native_capture(tmp_path):
    g = torch.Generator().manual_seed(17)
    target_video = torch.rand((1, 24, 22, 6, 6), generator=g) / 4
    prefix = target_video[:, :, :12].clone()
    source = torch.rand((1, 24, 22, 4, 4), generator=g) / 4
    source[:, :, :12] = resize_spatial_5d(prefix, 4, 4, mode="bicubic") + 0.1
    manifest = {
        "schema": 1,
        "kind": "h3_flow_native_boundary_decoder_window_evidence",
        "metadata": {
            "policy": "native_boundary_decoder_window_evidence_v1",
            "domain": "model_internal_clean_except_sampler_input_and_mask",
            "first_high_actual": True,
            "provider_clean_provenance": "actual_learned_provider_uniform_source",
            "decoder_comparison_prefix": "replace_with_authoritative_prefix_bytes",
            "process_latent_out_required_before_vae": True,
            "full_video_snapshots": True,
            "full_video_temporal_start_t": 0,
            "low_probe_native_carrier_decodable": True,
            "spatial_stage_control": "progressive_uniform_source",
            "target_band_tokens": 0,
            "target_band_transfer_start_t": 12,
            "source_probe_clean_grid": [4, 4],
            "source_prefix_projection_policy": "native_source_carry_v1",
            "source_prefix_projection": {
                "policy": "native_source_carry_v1",
                "prefix_t": 12,
                "projected_prefix": video_latent_fingerprint(source[:, :, :12]),
                "authoritative_prefix": video_latent_fingerprint(prefix),
            },
            "window": {"prefix_t": 12, "temporal": 22, "decoded_trim_frames": 39},
        },
        "tensor_bytes": {},
    }
    for name in STAGES.values():
        _save(tmp_path, manifest, name, target_video.clone())
    mask = torch.ones_like(target_video)
    mask[:, :, :12] = 0
    _save(tmp_path, manifest, "initial_high_video_mask_full", mask)
    _save(tmp_path, manifest, "authoritative_prefix_full", prefix)
    _save(tmp_path, manifest, "source_probe_clean_full", source)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest, source, prefix


class FakeNativeVAE:
    def __init__(self, *, cross_boundary):
        self.first_stage_model = SimpleNamespace(
            tokens_chunk_size=5, token_overlap=2, frame_pre_padding=3, clip_length=17
        )
        self.cross_boundary = cross_boundary
        self.inputs = []

    def decode(self, latent):
        self.inputs.append(latent.clone())
        frames = (latent.shape[2] - 2) // 5 * 17 + 5
        out = []
        prefix_mean = latent[:, :3, :12].mean(dim=(2, 3, 4), keepdim=True)
        for frame in range(frames):
            t = min(5 * (frame // 17) + 2, latent.shape[2] - 1)
            img = F.interpolate(latent[:, :3, t], scale_factor=16, mode="nearest")
            if self.cross_boundary and frame >= 39 and latent.shape[2] >= 22:
                img = img + prefix_mean[:, :, 0]
            out.append(img.movedim(1, -1))
        return torch.stack(out, dim=1).clamp(0, 1)


def _run(capture, vae, *, expected_hash=None, feature_tracking=False):
    return audit_native_prefix_counterfactual(
        vae,
        capture[0],
        175,
        lambda x: x,
        static_roi_profile="custom",
        static_roi_json=ROIS,
        feature_tracking_enabled=feature_tracking,
        expected_manifest_sha256=expected_hash,
    )


def test_prefix_intervention_is_only_protected_tokens_and_does_not_mutate_operands():
    src = torch.randn((1, 24, 22, 4, 4))
    target = torch.randn((1, 24, 12, 6, 6))
    before_src, before_target = src.clone(), target.clone()
    a, b = make_prefix_counterfactual(src, target, 12)
    assert torch.equal(a, before_src)
    assert torch.equal(b[:, :, 12:], a[:, :, 12:])
    assert torch.equal(b[:, :, :12], resize_spatial_5d(target, 4, 4, mode="bicubic"))
    assert torch.equal(src, before_src) and torch.equal(target, before_target)
    assert a.data_ptr() != src.data_ptr() and b.data_ptr() != src.data_ptr()


@pytest.mark.parametrize("wrong", ["time", "spatial", "dtype", "domain"])
def test_invalid_tensor_domain_or_phase_rejected(wrong):
    src = torch.zeros((1, 24, 22, 4, 4))
    target = torch.zeros((1, 24, 12, 6, 6))
    if wrong == "time":
        src = src[:, :, :21]
    elif wrong == "spatial":
        target = target[..., :5, :]
    elif wrong == "dtype":
        src = src.half()
    else:
        target = target[0]
    with pytest.raises(ValueError, match="native H3"):
        make_prefix_counterfactual(src, target, 12)


def test_full_native_replay_preserves_saved_bytes_and_makes_three_vae_calls(native_capture):
    directory, _manifest, source, _prefix = native_capture
    saved = {p.name: p.read_bytes() for p in directory.iterdir()}
    vae = FakeNativeVAE(cross_boundary=True)
    report = _run(native_capture, vae)
    assert report["generated_suffix_bitwise_identical"] is True
    assert report["generated_suffix_tokens"] == [12, 22]
    assert report["replacement_tokens"] == [0, 12]
    assert report["changed_prefix_float32_elements"] > 0
    assert report["baseline_matches_saved_source_window_bytes"] is True
    assert report["vae"]["extra_vae_calls"] == len(vae.inputs) == 3
    assert [v.shape[2] for v in vae.inputs] == [22, 22, 17]
    assert torch.equal(vae.inputs[0], source)
    assert torch.equal(vae.inputs[0][:, :, 12:], vae.inputs[1][:, :, 12:])
    assert report["production_sampling_rerun"] is False
    assert report["vae"]["extra_h3_nfe"] == 0
    assert report["frame_labels"][0] == 170
    assert 190 in report["frame_labels"]
    assert report["full_vs_cropped_baseline"]["same_frame"]["policy"] == "h3_same_frame_prefix_context_v1"
    assert report["temporal_increment_delta"]["175"]["temporal_increment_change_rms"] is not None
    assert saved == {p.name: p.read_bytes() for p in directory.iterdir()}


def test_background_tracker_receives_only_named_static_rois_for_both_anchors_and_arms(native_capture, monkeypatch):
    from h3_flow_regenerate import native_prefix_counterfactual as counterfactual

    seen = []

    def capture_rois(pixels, labels, *, join_frame, rois):
        # Snapshot each call: do not accept a mutable dict later modified by
        # an appended foreground/upper-region diagnostic.
        seen.append((join_frame, dict(rois)))
        return {"status": "measured", "join_frame": join_frame, "regions_xyxy": dict(rois)}

    monkeypatch.setattr(counterfactual, "track_background_features", capture_rois)
    report = _run(native_capture, FakeNativeVAE(cross_boundary=True), feature_tracking=True)
    expected = {
        "left": (0.0, 0.0, 0.48, 0.8),
        "right": (0.52, 0.0, 1.0, 0.8),
    }

    assert seen == [(175, expected), (179, expected), (175, expected), (179, expected)]
    assert report["background_tracking_support"] == "independently_detected_per_variant_and_anchor; static_rois_only"
    for arm in ("baseline", "counterfactual"):
        variant = report[arm]
        assert set(variant["static_background_rois"]["regions"]) == set(expected)
        assert variant["feature_tracking_from_f174"]["regions_xyxy"] == expected
        assert variant["feature_tracking_from_f178"]["regions_xyxy"] == expected
    for key in ("same_frame_A_to_B",):
        assert "upper45_full" in report[key]["regions"]
    assert "upper45_full" in report["full_vs_cropped_baseline"]["same_frame"]["regions"]
    assert report["same_frame_geometry_upper45"]["status"] in ("estimated", "insufficient_texture")


def test_equal_prefix_decodes_identically(native_capture):
    directory, manifest, source, prefix = native_capture
    source[:, :, :12] = resize_spatial_5d(prefix, 4, 4, mode="bicubic")
    _save(directory, manifest, "source_probe_clean_full", source)
    manifest["metadata"]["source_prefix_projection"]["projected_prefix"] = video_latent_fingerprint(source[:, :, :12])
    (directory / "manifest.json").write_text(json.dumps(manifest))
    result = _run(native_capture, FakeNativeVAE(cross_boundary=True))
    assert result["changed_prefix_float32_elements"] == 0
    assert result["source_prefix_equal_to_projection"] is True
    assert max(r["rgb_rmse"] for r in result["temporal_increment_delta"].values()) == 0


@pytest.mark.parametrize("cross_boundary", [False, True])
def test_decoder_cross_boundary_dependence_is_distinguishable(native_capture, cross_boundary):
    result = _run(native_capture, FakeNativeVAE(cross_boundary=cross_boundary))
    later = result["temporal_increment_delta"]["190"]["rgb_rmse"]
    if cross_boundary:
        assert later > 0
    else:
        assert later == 0


@pytest.mark.parametrize("fault", ["source_hash", "receipt", "manifest", "domain"])
def test_bad_provenance_fails_before_vae_calls(native_capture, fault):
    directory, manifest, _source, _prefix = native_capture
    if fault == "source_hash":
        path = directory / "source_probe_clean_full.bin"
        raw = bytearray(path.read_bytes())
        raw[0] ^= 0x10
        path.write_bytes(raw)
    elif fault == "receipt":
        manifest["metadata"]["source_prefix_projection"]["projected_prefix"]["sha256_float32"] = "0" * 64
    elif fault == "manifest":
        pass
    else:
        manifest["metadata"]["domain"] = "vae_encoded"
    if fault not in ("source_hash", "manifest"):
        (directory / "manifest.json").write_text(json.dumps(manifest))
    vae = FakeNativeVAE(cross_boundary=False)
    with pytest.raises(ValueError):
        _run(native_capture, vae, expected_hash=("0" * 64 if fault == "manifest" else None))
    assert vae.inputs == []


def test_missing_native_vae_rejected_before_any_decode(native_capture):
    bad = FakeNativeVAE(cross_boundary=False)
    bad.first_stage_model.token_overlap = 0
    with pytest.raises(ValueError, match="native MiniMax H3"):
        _run(native_capture, bad)
    assert bad.inputs == []
