import ast
import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate import local_boundary_audit as audit


@pytest.mark.parametrize("host", ["wsl.localhost", "wsl$"])
@pytest.mark.parametrize("manifest_input", [False, True])
@pytest.mark.parametrize("quoted", [False, True])
def test_wsl_explorer_paths_load_real_bundle(bundle, monkeypatch, host, manifest_input, quoted):
    directory, _manifest, _video = bundle
    monkeypatch.setenv("WSL_DISTRO_NAME", "Ubuntu-22.04")
    path = directory / "manifest.json" if manifest_input else directory
    unc = "\\\\" + host + "\\Ubuntu-22.04" + str(path).replace("/", "\\")
    if quoted:
        unc = f'  "{unc}"  '
    expected = audit.load_replay_operands(path, 175)
    actual = audit.load_replay_operands(unc, 175)
    assert actual[0] == expected[0]
    assert actual[2] == expected[2]
    for stage in actual[1]:
        assert torch.equal(actual[1][stage], expected[1][stage])


@pytest.mark.parametrize(
    "path",
    ["/tmp/bundle", "relative/bundle", r"/tmp/literal\name", "//tmp/bundle", "~/bundle"],
)
def test_linux_path_spelling_is_preserved(path):
    assert audit.normalize_bundle_path(path, platform="posix", wsl_distro=None) == path


@pytest.mark.parametrize("path", [r"\\wsl.localhost\Ubuntu-22.04\home\toor", r"C:\bundle"])
def test_windows_runtime_keeps_native_windows_paths(path):
    assert audit.normalize_bundle_path(path, platform="nt", wsl_distro=None) == path


def test_forward_slash_wsl_share_and_distribution_case():
    assert (
        audit.normalize_bundle_path(
            "//WSL.LOCALHOST/ubuntu-22.04/home/toor/bundle", platform="posix", wsl_distro="Ubuntu-22.04"
        )
        == "/home/toor/bundle"
    )


@pytest.mark.parametrize(
    ("path", "distro", "message"),
    [
        (r"\\wsl.localhost\Debian\home\toor", "Ubuntu-22.04", "distribution 'Debian'"),
        (r"\\wsl$\Ubuntu-22.04\home\toor", None, "WSL_DISTRO_NAME"),
        (r"\\wsl.localhost\Ubuntu-22.04", "Ubuntu-22.04", "incomplete WSL path"),
        (r"\\server\share\bundle", "Ubuntu-22.04", "network paths"),
        (r"C:\bundle", "Ubuntu-22.04", "drive paths"),
        ("C:/bundle", "Ubuntu-22.04", "drive paths"),
        ("C:bundle", "Ubuntu-22.04", "drive paths"),
        ("   ", None, "select the boundary bundle"),
        ('""', None, "select the boundary bundle"),
    ],
)
def test_inaccessible_path_formats_fail_with_actionable_error(path, distro, message):
    with pytest.raises(ValueError, match=message):
        audit.normalize_bundle_path(path, platform="posix", wsl_distro=distro)


@pytest.mark.parametrize("manifest_input", [False, True])
def test_missing_manifest_reports_expected_file_without_doubled_filename(tmp_path, manifest_input):
    path = tmp_path / "manifest.json" if manifest_input else tmp_path
    with pytest.raises(FileNotFoundError, match="Select an existing exported bundle") as error:
        audit.load_replay_operands(path, 175)
    assert str(tmp_path / "manifest.json") in str(error.value)
    assert "manifest.json/manifest.json" not in str(error.value)


def write_operand(directory, manifest, name, value):
    raw = value.contiguous().numpy().tobytes()
    path = directory / f"{name}.bin"
    path.write_bytes(raw)
    manifest["tensor_bytes"][name] = {
        "file": path.name,
        "shape": list(value.shape),
        "dtype": "torch.float32",
        "byte_order": "native_torch_contiguous",
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@pytest.fixture
def bundle(tmp_path):
    generator = torch.Generator().manual_seed(13)
    video = torch.rand((1, 24, 22, 2, 3), generator=generator) * 0.5
    manifest = {
        "schema": 1,
        "kind": "h3_flow_native_boundary_decoder_window_evidence",
        "metadata": {
            "policy": "native_boundary_decoder_window_evidence_v1",
            "first_high_actual": True,
            "provider_clean_provenance": "actual_learned_provider_before_target_band_splice",
            "decoder_comparison_prefix": "replace_with_authoritative_prefix_bytes",
            "process_latent_out_required_before_vae": True,
            "full_video_snapshots": True,
            "full_video_temporal_start_t": 0,
            "low_probe_native_carrier_decodable": False,
            "target_band_tokens": 4,
            "target_band_transfer_start_t": 16,
            "window": {"prefix_t": 12, "temporal": 22, "decoded_trim_frames": 39},
        },
        "tensor_bytes": {},
    }
    for name in audit.STAGES.values():
        value = video.clone()
        if name == "provider_native_clean_full":
            value[:, :, :16] += 0.1
        elif name.startswith("first_high"):
            value += 0.2
        write_operand(tmp_path, manifest, name, value)
    write_operand(tmp_path, manifest, "authoritative_prefix_full", video[:, :, :12])
    write_operand(tmp_path, manifest, "low_probe_native_carrier_clean_full", video)
    mask = torch.ones_like(video)
    mask[:, :, :12] = 0
    write_operand(tmp_path, manifest, "initial_high_video_mask_full", mask)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest, video


def test_replay_covers_late_band_edge_and_restores_only_prefix_without_writing_files(bundle):
    directory, _manifest, video = bundle
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    plan, stages, identity = audit.load_replay_operands(directory, 175)
    assert (plan["token_start"], plan["token_stop"]) == (5, 22)
    assert plan["decoded_frames"] == 56
    assert plan["decoded_origin_frame"] == 153
    assert plan["shared_tokens"] == [[10, 12], [15, 17]]
    assert plan["temporal_blend_local_frames"] == [[17, 22], [34, 39]]
    assert plan["measured_local_frames"] == [18, 43]
    assert {171, 175, 188, 195} <= set(range(plan["decoded_origin_frame"] + 18, plan["decoded_origin_frame"] + 43))
    for value in stages.values():
        assert torch.equal(value[:, :, :7], video[:, :, 5:12])
        assert value.untyped_storage().nbytes() == value.numel() * 4
    assert torch.equal(stages["provider"][:, :, 7:11], video[:, :, 12:16] + 0.1)
    assert torch.equal(stages["pre_high"][:, :, 7:11], video[:, :, 12:16])
    assert len(identity["operand_sha256"]) == 8
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("operand", ["provider_native_clean_full", "first_high_before_flow_full"])
def test_corrupt_saved_bytes_are_rejected_before_vae_decode(bundle, operand):
    directory, _manifest, _video = bundle
    path = directory / f"{operand}.bin"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="hash differs"):
        audit.load_replay_operands(directory, 175)


@pytest.mark.parametrize("failure", ["prefix", "mask", "band", "path", "nonfinite"])
def test_invalid_ownership_and_operand_provenance_fail_closed(bundle, failure):
    directory, manifest, video = bundle
    if failure == "prefix":
        bad = video.clone()
        bad[:, :, 0] += 1
        write_operand(directory, manifest, "final_post_high_internal_clean_full", bad)
    elif failure == "mask":
        bad = torch.ones_like(video)
        bad[:, :, :13] = 0
        write_operand(directory, manifest, "initial_high_video_mask_full", bad)
    elif failure == "band":
        bad = video.clone()
        bad[:, :, 15] += 1
        write_operand(directory, manifest, "low_probe_native_carrier_clean_full", bad)
    elif failure == "path":
        manifest["tensor_bytes"]["provider_native_clean_full"]["file"] = "../outside.bin"
    else:
        bad = video.clone()
        bad[:, :, 16] = float("nan")
        write_operand(directory, manifest, "provider_native_clean_full", bad)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit.load_replay_operands(directory, 175)


def test_partial_tail_cannot_be_replayed_as_a_complete_decoder_overlap(bundle):
    _directory, manifest, _video = bundle
    manifest["metadata"]["target_band_tokens"] = 9
    manifest["metadata"]["target_band_transfer_start_t"] = 21
    with pytest.raises(ValueError, match="complete following decoder window"):
        audit.replay_plan(manifest["metadata"], 0)


@pytest.mark.parametrize("band", [1, 4, 9])
@pytest.mark.parametrize("prefix", [2, 12, 22])
def test_join_frame_changes_only_labels_and_never_selects_different_samples(bundle, band, prefix):
    _directory, manifest, _video = bundle
    metadata = dict(manifest["metadata"])
    metadata["window"] = {"prefix_t": prefix, "temporal": 62, "decoded_trim_frames": 17 * ((prefix - 2) // 5) + 5}
    metadata.update(target_band_tokens=band, target_band_transfer_start_t=prefix + band)
    absolute = audit.replay_plan(metadata, 175)
    relative = audit.replay_plan(metadata, 0)
    assert absolute["measured_local_frames"] == relative["measured_local_frames"]
    assert absolute["decoded_origin_frame"] - relative["decoded_origin_frame"] == 175
    assert {k: v for k, v in absolute.items() if k not in ("decoded_origin_frame", "join_frame")} == {
        k: v for k, v in relative.items() if k not in ("decoded_origin_frame", "join_frame")
    }
    begin, end = absolute["measured_local_frames"]
    assert 0 < begin < end <= absolute["decoded_frames"]


class FakeVAE:
    def __init__(self):
        self.first_stage_model = SimpleNamespace(
            tokens_chunk_size=5, token_overlap=2, frame_pre_padding=3, clip_length=17
        )
        self.inputs = []

    def decode(self, value):
        self.inputs.append(value.clone())
        frames = (value.shape[2] - 2) // 5 * 17 + 5
        return torch.zeros(1, frames, value.shape[-2] * 16, value.shape[-1] * 16, 3)


@pytest.fixture
def high_call_bundle(bundle):
    directory, manifest, video = bundle
    window = manifest["metadata"]["window"]
    window.update(window_start_t=10, window_stop_t=17, window_tokens=7)
    calls = []
    for index, actual in enumerate((True, False, True)):
        call = {"call_index": index, "sigma": 0.8 - index * 0.1, "actual": actual}
        for point in ("before_flow", "after_flow"):
            name = f"first_high_{point}" if index == 0 else f"high_prediction_{index:02d}_{point}"
            value = torch.full_like(video[:, :, 10:17], 0.2 + index * 0.05)
            value[:, :, :2] = 0.9  # Must be replaced by the actual prefix only for replay.
            if point == "after_flow":
                value[:, :, 3] += 0.01
            write_operand(directory, manifest, name, value)
            call[point] = name
        calls.append(call)
    window["high_prediction_trace"] = {
        "policy": "bounded_high_prediction_windows_v1",
        "max_calls": 16,
        "calls": calls,
        "omitted_call_indices": [],
    }
    manifest["metadata"]["first_high_call_index"] = 0
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest, video


def test_high_tone_replay_pairs_actual_and_forecast_calls_and_preserves_saved_files(high_call_bundle):
    directory, _manifest, video = high_call_bundle
    originals = {path.name: path.read_bytes() for path in directory.iterdir()}

    class ToneVAE(FakeVAE):
        def decode(self, value):
            self.inputs.append(value.clone())
            pixels = super().decode(value)
            pixels[:, :] = value[:, :, 2:].mean()
            # A local temporal drop independent of the global stage bias.
            pixels[:, 10:, :10, :10] -= value[:, :, 3].mean() * 0.1
            return pixels

    vae = ToneVAE()
    result = audit.audit_local_boundary(
        vae, directory, 175, lambda x: x, scope="high_prediction_tone", detail_region="upper_left"
    )
    assert result["extra_vae_calls"] == 6
    assert result["frame_labels"] == list(range(176, 187))
    assert [call["actual"] for call in result["calls"]] == [True, False, True]
    assert "between_calls_change" not in result["calls"][0]
    assert result["calls"][1]["previous_captured_call_index"] == 0
    assert result["calls"][1]["between_calls_change"]["full"]["same_frame_luma_change"][0] > 0.04
    assert result["calls"][1]["immediate_flow_change"]["upper_left"]["adjacent_luma_increment_change"][4] < 0
    for value in vae.inputs:
        assert torch.equal(value[:, :, :2], video[:, :, 10:12])
    assert originals == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert result["rendered_acceptance"] is False


@pytest.mark.parametrize("failure", ["index", "sigma", "actual", "name", "hash", "timing", "omitted"])
def test_high_tone_replay_rejects_corrupt_call_provenance_before_decoding(high_call_bundle, failure):
    directory, manifest, _video = high_call_bundle
    trace = manifest["metadata"]["window"]["high_prediction_trace"]
    call = trace["calls"][1]
    if failure == "index":
        call["call_index"] = 0
    elif failure == "sigma":
        call["sigma"] = float("nan")
    elif failure == "actual":
        call["actual"] = 1
    elif failure == "name":
        call["before_flow"] = "first_high_before_flow"
    elif failure == "hash":
        manifest["tensor_bytes"][call["before_flow"]]["sha256"] = "0" * 64
    elif failure == "timing":
        manifest["metadata"]["window"]["window_start_t"] = 9
    else:
        trace["omitted_call_indices"] = [1]
    (directory / "manifest.json").write_text(json.dumps(manifest))
    vae = FakeVAE()
    with pytest.raises(ValueError):
        audit.audit_local_boundary(vae, directory, 175, lambda x: x, scope="high_prediction_tone")
    assert vae.inputs == []


def test_high_tone_scope_reports_older_bundle_without_attempting_a_decode(bundle):
    directory, _manifest, _video = bundle
    vae = FakeVAE()
    with pytest.raises(ValueError, match="predates high-call capture"):
        audit.audit_local_boundary(vae, directory, 175, lambda x: x, scope="high_prediction_tone")
    assert vae.inputs == []


@pytest.fixture
def source_bundle(bundle):
    directory, manifest, _video = bundle
    # Real H3 projection requires even spatial axes. Enlarge the compact fixture
    # before adding a uniform reduced view of its target-grid head and stored tail.
    values = {}
    for name, entry in list(manifest["tensor_bytes"].items()):
        value = torch.frombuffer(bytearray((directory / entry["file"]).read_bytes()), dtype=torch.float32)
        value = value.reshape(entry["shape"]).repeat_interleave(2, -2).repeat_interleave(2, -1)
        values[name] = value
        write_operand(directory, manifest, name, value)
    native = values["low_probe_native_carrier_clean_full"]
    head = native[:, :, :16].clone()
    head[:, :, :12] = values["authoritative_prefix_full"]
    projected = audit.resize_spatial_5d_h3_patch_lattice(head, 2, 4)
    source = torch.cat((projected, native[:, :, 16:, :2, :4]), dim=2)
    manifest["metadata"]["source_probe_clean_grid"] = [2, 4]
    write_operand(directory, manifest, "source_probe_clean_full", source)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    return directory, manifest, source


def test_extended_replay_validates_saved_source_view_and_preserves_native_tail_bytes(source_bundle):
    directory, _manifest, source = source_bundle
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    _plan, stages, identity = audit.load_replay_operands(directory, 175, include_source=True)
    assert tuple(stages["source_grid"].shape) == (1, 24, 17, 2, 4)
    assert torch.equal(stages["source_grid"], source[:, :, 5:22])
    assert len(identity["operand_sha256"]) == 9
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize(
    "projection_policy",
    [None, "h3_dense_patch_center_lattice_v2", "half_pixel_latent_v1", "native_vae_rgb_roundtrip_v1"],
)
def test_uniform_source_bundle_has_no_band_or_padded_carrier_requirement(source_bundle, projection_policy):
    directory, manifest, source = source_bundle
    metadata = manifest["metadata"]
    metadata.update(
        spatial_stage_control="progressive_uniform_source",
        provider_clean_provenance="actual_learned_provider_uniform_source",
        target_band_tokens=0,
        target_band_transfer_start_t=12,
        low_probe_native_carrier_layout="uniform_source",
        low_probe_native_carrier_decodable=True,
    )
    manifest["tensor_bytes"].pop("low_probe_native_carrier_clean_full")
    if projection_policy is not None:
        metadata["source_prefix_projection_policy"] = projection_policy
    if projection_policy in ("half_pixel_latent_v1", "native_vae_rgb_roundtrip_v1"):
        entry = manifest["tensor_bytes"]["authoritative_prefix_full"]
        prefix = torch.frombuffer(bytearray((directory / entry["file"]).read_bytes()), dtype=torch.float32)
        prefix = prefix.reshape(entry["shape"])
        source = source.clone()
        source[:, :, :12] = audit.resize_spatial_5d(prefix, 2, 4, mode="bicubic")
        if projection_policy == "native_vae_rgb_roundtrip_v1":
            from h3_flow_regenerate.decode_context import video_latent_fingerprint

            # Native reconstruction cannot be reproduced with latent resize.
            source[:, :, :12] += 0.04
            metadata["source_prefix_projection"] = {
                "policy": projection_policy,
                "prefix_t": 12,
                "authoritative_prefix": video_latent_fingerprint(prefix),
                "projected_prefix": video_latent_fingerprint(source[:, :, :12]),
            }
        write_operand(directory, manifest, "source_probe_clean_full", source)
    (directory / "manifest.json").write_text(json.dumps(manifest))
    plan, stages, _identity = audit.load_replay_operands(directory, 175, include_source=True)
    assert plan["head_t"] == plan["prefix_t"] == 12
    assert torch.equal(stages["source_grid"], source[:, :, 5:22])
    report = audit.audit_local_boundary(FakeVAE(), directory, 175, lambda x: x)
    assert report["extra_vae_calls"] == 5
    assert report["stages"]["provider"]["frame_labels"] == list(range(171, 196))


@pytest.mark.parametrize("policy", ["unknown", "half_pixel_latent_v1"])
def test_extended_replay_rejects_unsupported_projection_without_decoding(source_bundle, policy):
    directory, manifest, _source = source_bundle
    # Half-pixel is valid only for a native uniform clip, not a mixed band view.
    manifest["metadata"]["source_prefix_projection_policy"] = policy
    (directory / "manifest.json").write_text(json.dumps(manifest))
    vae = FakeVAE()
    with pytest.raises(ValueError, match="source prefix projection policy"):
        audit.audit_local_boundary(vae, directory, 175, lambda x: x, scope="transfer_and_decoder_context")
    assert vae.inputs == []


@pytest.mark.parametrize("failure", ["head", "tail", "grid", "shape", "hash"])
def test_extended_replay_rejects_wrong_or_corrupt_source_view(source_bundle, failure):
    directory, manifest, source = source_bundle
    if failure in ("head", "tail"):
        bad = source.clone()
        bad[:, :, 12 if failure == "head" else 16] += 0.01
        write_operand(directory, manifest, "source_probe_clean_full", bad)
    elif failure == "grid":
        manifest["metadata"]["source_probe_clean_grid"] = [2, 3]
    elif failure == "shape":
        write_operand(directory, manifest, "source_probe_clean_full", source[:, :, :, :, :2])
    else:
        manifest["tensor_bytes"]["source_probe_clean_full"]["sha256"] = "0" * 64
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit.load_replay_operands(directory, 175, include_source=True)


def test_extended_audit_uses_same_time_window_context_and_keeps_reduced_grid_units(source_bundle, monkeypatch):
    directory, _manifest, _source = source_bundle
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})
    vae = FakeVAE()
    rng = torch.random.get_rng_state().clone()
    report = audit.audit_local_boundary(vae, directory, 175, lambda v: v, scope="transfer_and_decoder_context")
    assert report["extra_vae_calls"] == len(vae.inputs) == 18
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert len(report["stages"]) == 6 and len(report["comparisons"]) == 4
    assert report["stages"]["source_grid"]["decoded_pixel_hw"] == [32, 64]
    assert report["stages"]["pre_high"]["decoded_pixel_hw"] == [64, 96]
    assert report["stages"]["source_grid"]["native_reduced_grid_generation_for_head"] is False
    for stage_index, stage in enumerate(report["stages"].values()):
        full, left, right = vae.inputs[stage_index * 3 : stage_index * 3 + 3]
        assert torch.equal(left, full[:, :, 5:12])
        assert torch.equal(right, full[:, :, 10:17])
        context = stage["window_context"]
        assert context["frame_labels"] == list(range(187, 192))
        assert context["rgb_difference_rms"] == [0.0] * 5
        assert context["production_blends_before_pixel_clamp"] is True
        assert context["standalone_pixels_used_to_reassemble_output"] is False
    encoded = json.dumps(report, allow_nan=False)
    assert str(directory) not in encoded and "tensor(" not in encoded


def test_window_context_pairs_identical_frame_times_instead_of_adjacent_motion(monkeypatch):
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})
    inputs = torch.arange(12).view(1, 1, 12, 1, 1).expand(1, 24, 12, 2, 4).float().clone()

    class ContextVAE(FakeVAE):
        def decode(self, value):
            pixels = super().decode(value)
            # Seven-token contexts start five tokens / seventeen pixel times apart.
            origin = float(value[0, 0, 0, 0, 0]) * 17 / 5
            times = torch.arange(22, dtype=pixels.dtype) + origin
            pixels += (0.1 + times * 0.001)[None, :, None, None, None]
            if origin:
                pixels += 0.02  # Same-time context disagreement, separate from motion.
            return pixels

    pixels_before = inputs.clone()
    report = audit.measure_window_context(
        ContextVAE(), inputs, lambda v: v, {"decoded_origin_frame": 170, "head_t": 16, "token_start": 10}
    )
    assert report["rgb_difference_rms"] == pytest.approx([0.02] * 5, abs=1e-7)
    assert report["luma_mean_change"] == pytest.approx([0.02] * 5, abs=1e-7)
    assert torch.equal(inputs, pixels_before)


def test_unknown_audit_scope_fails_before_decoding(bundle):
    directory, _manifest, _video = bundle
    vae = FakeVAE()
    with pytest.raises(ValueError, match="unsupported local boundary audit scope"):
        audit.audit_local_boundary(vae, directory, 175, lambda v: v, scope="unknown")
    assert vae.inputs == []


def test_audit_uses_connected_vae_sequentially_and_returns_only_numerical_data(bundle, monkeypatch):
    directory, _manifest, _video = bundle
    vae = FakeVAE()
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})
    converted = []

    def process_out(value):
        converted.append(value.clone())
        return value * 0.5

    rng = torch.random.get_rng_state().clone()
    report = audit.audit_local_boundary(vae, directory, 175, process_out)
    assert report["policy"] == "local_target_band_native_window_audit_v2"
    assert report["static_roi_profile"] == "off"
    assert report["static_roi_measurement_enabled"] is False
    assert report["static_roi_bounds_xyxy"] == {}
    assert report["extra_vae_calls"] == 5
    assert report["temporal_blend_reproduced_for_measured_frames"] is True
    assert report["decoded_pixels_saved"] is False
    assert report["rendered_acceptance"] is False
    assert len(vae.inputs) == len(converted) == 5
    for actual, internal in zip(vae.inputs, converted, strict=True):
        assert torch.equal(actual, internal * 0.5)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert report["stages"]["final"]["frame_labels"] == list(range(171, 196))
    assert len(report["comparisons"]) == 4
    for stage in report["stages"].values():
        assert stage["adjacent_rgb_difference_rms"] == [0.0] * 25
        assert stage["adjacent_luma_mean_change"] == [0.0] * 25
        assert stage["adjacent_frame_geometry"]["frames"] == list(range(171, 196))
        assert stage["adjacent_frame_geometry_upper45"]["frames"] == list(range(171, 196))
    assert report["final_adjacent_frame_geometry"] is report["stages"]["final"]["adjacent_frame_geometry"]
    encoded = json.dumps(report, allow_nan=False)
    assert str(directory) not in encoded
    assert "tensor(" not in encoded


def test_temporal_increment_locates_new_stage_discontinuity_without_confusing_stable_bias(bundle, monkeypatch):
    directory, _manifest, _video = bundle
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})

    class TemporalVAE(FakeVAE):
        def decode(self, value):
            pixels = super().decode(value)
            stage = len(self.inputs) - 1
            times = torch.arange(pixels.shape[1], dtype=pixels.dtype)[None, :, None, None, None]
            pixels += 0.1 + times * 0.001  # Same smooth motion in every stage.
            if stage >= 1:
                pixels += 0.02  # Stable provider-to-pre-high bias, no temporal edge.
            if stage >= 2:
                pixels[:, 35:] += 0.1  # First high creates a step at assembled frame 188.
            return pixels

    vae = TemporalVAE()
    rng = torch.random.get_rng_state().clone()
    report = audit.audit_local_boundary(vae, directory, 175, lambda value: value)
    assert report["extra_vae_calls"] == len(vae.inputs) == 5
    assert torch.equal(torch.random.get_rng_state(), rng)
    index = report["stages"]["pre_high"]["frame_labels"].index(188)
    for stage in ("provider", "pre_high"):
        assert report["stages"][stage]["adjacent_rgb_difference_rms"] == pytest.approx([0.001] * 25, abs=1e-7)
    for stage in ("first_high_before_flow", "first_high_after_flow", "final"):
        expected = [0.001] * 25
        expected[index] = 0.101
        assert report["stages"][stage]["adjacent_rgb_difference_rms"] == pytest.approx(expected, abs=1e-7)
    for pair, measurement in report["comparisons"].items():
        expected = [0.0] * 25
        if pair == "pre_high_to_first_high_before_flow":
            expected[index] = 0.1
        assert measurement["temporal_increment_change_rms"] == pytest.approx(expected, abs=1e-7)


@pytest.mark.parametrize("detail_region", ["off", "upper_left"])
def test_node_saves_only_json_and_is_registered(bundle, monkeypatch, tmp_path, detail_region):
    import sys

    directory, _manifest, _video = bundle
    monkeypatch.setattr(audit, "_AUDIT_MODEL_OWNERS", SimpleNamespace(retain=lambda: None))
    output = tmp_path / "output"
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(output)))
    monkeypatch.setitem(sys.modules, "comfy", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "comfy.latent_formats",
        SimpleNamespace(MiniMaxH3Video=lambda: SimpleNamespace(process_out=lambda value: value)),
    )
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})
    result = audit.H3FlowLocalBoundaryAudit().audit(FakeVAE(), str(directory), 175, detail_region=detail_region)
    files = list(output.rglob("*.*"))
    assert len(files) == 1 and files[0].suffix == ".json"
    assert json.loads(files[0].read_text()) == json.loads(result["result"][0])
    assert json.loads(result["result"][0])["detail_region"] == detail_region
    assert audit.NODE_CLASS_MAPPINGS["H3FlowLocalBoundaryAudit"] is audit.H3FlowLocalBoundaryAudit


@pytest.mark.parametrize("scope,call_count", [("stage_continuity", 5), ("transfer_and_decoder_context", 18)])
def test_detail_region_is_additive_without_new_decodes_or_changed_native_measurements(
    source_bundle, monkeypatch, scope, call_count
):
    directory, _manifest, _source = source_bundle
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})
    baseline = audit.audit_local_boundary(FakeVAE(), directory, 175, lambda v: v, scope=scope)
    vae = FakeVAE()
    rng = torch.random.get_rng_state().clone()
    detailed = audit.audit_local_boundary(vae, directory, 175, lambda v: v, scope=scope, detail_region="upper_left")
    assert len(vae.inputs) == detailed["extra_vae_calls"] == call_count
    assert torch.equal(torch.random.get_rng_state(), rng)
    for stage in detailed["stages"].values():
        region = stage["detail_region"]
        h, w = stage["decoded_pixel_hw"]
        assert region["bounds_xyxy"] == [0, 0, round(w / 3), round(h * 0.45)]
        assert region["frame_labels"] == stage["frame_labels"]
        if scope == "transfer_and_decoder_context":
            context = stage["window_context"]["detail_region"]
            assert context["frame_labels"] == list(range(187, 192))
            assert context["luma_mean_change"] == [0.0] * 5
    for pair in detailed["comparisons"].values():
        assert pair["detail_region"]["temporal_increment_change_rms"] == [0.0] * 25

    def remove_detail(value):
        if isinstance(value, dict):
            return {k: remove_detail(v) for k, v in value.items() if k != "detail_region"}
        if isinstance(value, list):
            return [remove_detail(v) for v in value]
        return value

    assert remove_detail(detailed) == remove_detail(baseline)


def test_detail_region_separates_local_tone_onset_from_motion_elsewhere(bundle, monkeypatch):
    directory, _manifest, _video = bundle
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})

    class LocalToneVAE(FakeVAE):
        def decode(self, value):
            pixels = super().decode(value)
            times = torch.arange(pixels.shape[1])[None, :, None, None, None]
            pixels += 0.2 + times * 0.001
            h, w = pixels.shape[2:4]
            stop_y, stop_x = round(h * 0.45), round(w / 3)
            # Motion outside the chosen region contributes to broad frame statistics.
            pixels[:, :, :, stop_x:] += (times % 2) * 0.1
            stage = len(self.inputs) - 1
            if stage >= 1:
                pixels += 0.02
            if stage >= 2:
                pixels[:, 27:, :stop_y, :stop_x] -= 0.02
            return pixels

    vae = LocalToneVAE()
    report = audit.audit_local_boundary(vae, directory, 175, lambda v: v, detail_region="upper_left")
    assert len(vae.inputs) == report["extra_vae_calls"] == 5
    labels = report["stages"]["final"]["frame_labels"]
    index = labels.index(180)
    for stage in ("provider", "pre_high"):
        region = report["stages"][stage]["detail_region"]
        assert region["adjacent_luma_mean_change"] == pytest.approx([0.001] * 25, abs=1e-7)
    region = report["stages"]["final"]["detail_region"]
    expected = [0.001] * 25
    expected[index] -= 0.02
    assert region["adjacent_luma_mean_change"] == pytest.approx(expected, abs=1e-7)
    stable = report["comparisons"]["provider_to_pre_high"]["detail_region"]
    assert stable["luma_mean_change"] == pytest.approx([0.02] * 25, abs=1e-7)
    assert stable["temporal_increment_change_rms"] == pytest.approx([0.0] * 25, abs=1e-7)
    onset = report["comparisons"]["pre_high_to_first_high_before_flow"]["detail_region"]
    expected = [0.0] * 25
    expected[index] = 0.02
    assert onset["temporal_increment_change_rms"] == pytest.approx(expected, abs=1e-7)


def test_detail_geometry_measures_local_translation_without_foreground_changes_or_pixel_mutation():
    generator = torch.Generator().manual_seed(71)
    image = F.avg_pool2d(torch.rand(1, 3, 128, 192, generator=generator), 3, stride=1, padding=1)
    shifted = torch.roll(image, 2, dims=-1)
    shifted[:, :, :, 64:] = 0.1
    a, b = image.permute(0, 2, 3, 1), shifted.permute(0, 2, 3, 1)
    original_a, original_b = a.clone(), b.clone()
    rng = torch.random.get_rng_state().clone()
    measured = audit._detail_pair(a, b, [188], "upper_left")
    row = measured["geometry"]["frames"][0]
    assert row["center_dx_dy_pixels"][0] == pytest.approx(-2.0, abs=0.15)
    assert row["affine_huber"] < row["zero_huber"] * 0.1
    assert torch.equal(a, original_a) and torch.equal(b, original_b)
    assert torch.equal(torch.random.get_rng_state(), rng)


@pytest.mark.parametrize("region", ["unknown", None, True])
def test_unsupported_detail_region_fails_before_decoding(bundle, region):
    directory, _manifest, _video = bundle
    vae = FakeVAE()
    with pytest.raises(ValueError, match="unsupported local boundary detail region"):
        audit.audit_local_boundary(vae, directory, 175, lambda v: v, detail_region=region)
    assert vae.inputs == []


def test_affine_measurement_recovers_translation_without_pixel_or_rng_mutation():
    generator = torch.Generator().manual_seed(71)
    image = torch.rand(1, 3, 64, 80, generator=generator)
    image = F.avg_pool2d(image, 3, stride=1, padding=1)
    shifted = torch.roll(image, 2, dims=-1)
    a, b = image.permute(0, 2, 3, 1), shifted.permute(0, 2, 3, 1)
    original_a, original_b = a.clone(), b.clone()
    rng = torch.random.get_rng_state().clone()
    measured = audit.geometry_comparison(a, b, [188])
    row = measured["frames"][0]
    assert row["frame"] == 188
    assert row["center_dx_dy_pixels"][0] == pytest.approx(-2.0, abs=0.15)
    assert row["affine_huber"] < row["zero_huber"] * 0.1
    assert torch.equal(a, original_a) and torch.equal(b, original_b)
    assert torch.equal(torch.random.get_rng_state(), rng)


def test_affine_measurement_detects_scale_and_shear_as_distinct_from_translation():
    generator = torch.Generator().manual_seed(19)
    image = F.avg_pool2d(torch.rand(1, 3, 80, 96, generator=generator), 3, stride=1, padding=1)
    yy, xx = torch.meshgrid(torch.arange(80), torch.arange(96), indexing="ij")
    x, y = xx.float() - 47.5, yy.float() - 39.5
    warped_x = 1.04 * x + 0.025 * y + 47.5
    warped_y = 0.015 * x + 0.97 * y + 39.5
    grid = torch.stack((2 * warped_x / 95 - 1, 2 * warped_y / 79 - 1), -1)[None]
    candidate = F.grid_sample(image, grid, align_corners=True, padding_mode="border")
    result = audit.geometry_comparison(image.permute(0, 2, 3, 1), candidate.permute(0, 2, 3, 1), [188])
    row = result["frames"][0]
    gradients = row["displacement_gradients"]
    assert gradients[0][0] == pytest.approx(0.04, abs=0.015)
    assert gradients[1][1] == pytest.approx(-0.03, abs=0.015)
    assert gradients[0][1] == pytest.approx(0.025, abs=0.015)
    assert gradients[1][0] == pytest.approx(0.015, abs=0.015)
    assert row["affine_huber"] < row["zero_huber"] * 0.5


def test_untextured_regions_are_not_reported_as_zero_geometric_shift():
    pixels = torch.zeros(2, 64, 80, 3)
    report = audit.geometry_comparison(pixels, pixels, [188, 189])
    assert report["status"] == "insufficient_texture"
    assert report["frames"] == [
        {"frame": 188, "status": "insufficient_texture"},
        {"frame": 189, "status": "insufficient_texture"},
    ]


def test_two_window_crop_matches_production_native_temporal_decode_at_band_edge(bundle, monkeypatch):
    root = os.environ.get("COMFYUI_ROOT")
    if root is None:
        pytest.skip("native temporal source oracle requires COMFYUI_ROOT")
    source = Path(root) / "comfy/ldm/minimax/vae.py"
    tree = ast.parse(source.read_text())
    native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3VideoVAE")
    wanted = {
        "blend",
        "_decode_temporal_pad_frames",
        "_decode_temporal_frame_plan",
        "_decode_temporal_chunks",
        "decode_temporal",
    }
    body = [node for node in native.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
    assert {node.name for node in body} == wanted
    namespace = {
        "torch": torch,
        "math": math,
        "comfy": SimpleNamespace(model_management=SimpleNamespace(intermediate_device=lambda: "cpu")),
    }
    code = ast.Module(
        body=[ast.ClassDef(name="NativeTiming", bases=[], keywords=[], body=body, decorator_list=[])], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(code), str(source), "exec"), namespace)
    vae = namespace["NativeTiming"]()
    vae.tokens_chunk_size = 5
    vae.token_overlap = 2
    vae.vae_ratio_t = 4
    vae.token_drop = 3
    vae.frame_pre_padding = 3
    vae.clip_length = 17
    vae.frame_overlap = 5
    vae._finalize_pixels = lambda value: value
    vae.decode_output_shape = lambda shape: (1, 3, (shape[2] - 2) // 5 * 17 + 5, shape[3], shape[4])

    def decode_window(value):
        pixels = value[:, :3].repeat_interleave(4, dim=2)
        return pixels + value.mean() / 10  # Different window contexts have different outputs.

    vae._adaptive_decode = decode_window
    directory, _manifest, video = bundle
    plan, stages, _identity = audit.load_replay_operands(directory, 175)
    production = vae.decode_temporal(video)
    replay = vae.decode_temporal(stages["final"])
    start, stop = plan["measured_local_frames"]
    offset = 17 * (plan["token_start"] // 5)
    assert torch.equal(replay[:, :, start:stop], production[:, :, offset + start : offset + stop])
    assert not torch.equal(replay[:, :, :5], production[:, :, offset : offset + 5])
    crop = stages["final"][:, :, 5:17]
    band_replay = vae.decode_temporal(crop)
    left = vae.decode_temporal(crop[:, :, :7])
    right = vae.decode_temporal(crop[:, :, 5:12])
    assert torch.equal(band_replay[:, :, 7:17], left[:, :, 7:17])
    assert torch.equal(band_replay[:, :, 22:39], right[:, :, 5:22])
    context_bias = (crop[:, :, 5:12].mean() - crop[:, :, :7].mean()) / 10
    assert torch.allclose(right[:, :, :5] - left[:, :, 17:22], context_bias.expand_as(right[:, :, :5]), atol=1e-7)

    class NativeAdapter:
        def decode(self, value):
            pixels = vae.decode_temporal(value)
            pixels = pixels.repeat_interleave(16, -2).repeat_interleave(16, -1)
            return pixels.permute(0, 2, 3, 4, 1)

    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})
    context = audit.measure_window_context(NativeAdapter(), stages["final"], lambda v: v, plan)
    assert context["frame_labels"] == list(range(187, 192))
    assert context["rgb_difference_rms"] == pytest.approx([abs(float(context_bias))] * 5, abs=1e-7)

    # Native blending is applied to raw decoder values before pixel clamping.
    # Mixing finalized standalone windows would change the production result.
    def saturated_window(value):
        level = 1.5 if float(value[0, 0, 0, 0, 0]) else -0.5
        return torch.full((1, 3, 28, *value.shape[-2:]), level)

    vae._adaptive_decode = saturated_window
    vae._finalize_pixels = lambda value: value.clamp(0, 1)
    inputs = torch.arange(12).view(1, 1, 12, 1, 1).expand(1, 24, 12, 2, 3).float()
    native_blend = vae.decode_temporal(inputs)
    standalone_left = vae.decode_temporal(inputs[:, :, :7])
    standalone_right = vae.decode_temporal(inputs[:, :, 5:12])
    finalized_blend = vae.blend(standalone_left[:, :, 17:22], standalone_right[:, :, :5], 5, dim=2)
    assert bool((native_blend[:, :, 18] == 0).all())
    assert bool((finalized_blend[:, :, 1] == 0.2).all())


class ManagedVAE:
    """ComfyUI VAE surface: managed ``decode`` versus the resident first-stage model."""

    def __init__(self, oom=False):
        self.patcher = SimpleNamespace(
            current_loaded_device=lambda: torch.device("cpu"),
            loaded_size=lambda: 1024,
            model_size=lambda: 1024,
        )
        self.device = torch.device("cpu")
        self.output_device = torch.device("cpu")
        self.vae_dtype = torch.float32
        self.disable_offload = False
        self.managed_calls = 0
        self.oom = oom
        outer = self

        class FirstStage:
            def decode(self, z):
                if outer.oom:
                    raise torch.cuda.OutOfMemoryError("resident decode")
                frames = (z.shape[2] - 2) // 5 * 17 + 5
                return torch.full((1, 3, frames, z.shape[-2] * 16, z.shape[-1] * 16), 0.25)

        self.first_stage_model = FirstStage()

    def vae_output_dtype(self):
        return torch.float32

    def process_output(self, image):
        return image

    def decode(self, value):
        self.managed_calls += 1
        frames = (value.shape[2] - 2) // 5 * 17 + 5
        return torch.zeros(1, frames, value.shape[-2] * 16, value.shape[-1] * 16, 3)


@pytest.mark.parametrize("oom", [False, True])
def test_audit_decode_keeps_other_models_resident(monkeypatch, oom):
    pytest.importorskip("comfy.cli_args").args.cpu = True
    model_management = pytest.importorskip("comfy.model_management")
    requests = []
    monkeypatch.setattr(
        model_management,
        "load_models_gpu",
        lambda models, memory_required=0, **kwargs: requests.append((tuple(models), memory_required)),
    )
    monkeypatch.setattr(model_management, "soft_empty_cache", lambda *args, **kwargs: None)
    vae = ManagedVAE(oom=oom)
    others = [object(), object()]
    monkeypatch.setattr(model_management, "loaded_models", lambda: [*others, vae.patcher])
    latent = torch.zeros(1, 24, 7, 2, 3)
    if oom:
        with pytest.raises(RuntimeError, match="Resident models were preserved"):
            audit._decode_owned_pixels(vae, latent, lambda x: x, 22)
    else:
        decoded = audit._decode_owned_pixels(vae, latent, lambda x: x, 22)
        assert tuple(decoded.shape) == (1, 22, 32, 48, 3)
        assert float(decoded.max()) == 0.25
    assert requests == []
    assert vae.managed_calls == 0
    assert model_management.loaded_models() == [*others, vae.patcher]


def test_audit_decodes_a_partially_loaded_vae_in_place(monkeypatch):
    pytest.importorskip("comfy.cli_args").args.cpu = True
    management = pytest.importorskip("comfy.model_management")
    vae = ManagedVAE()
    vae.patcher.loaded_size = lambda: 512
    monkeypatch.setattr(management, "loaded_models", lambda: [vae.patcher])

    def forbidden(*args, **kwargs):
        pytest.fail("an audit entered model memory admission")

    monkeypatch.setattr(management, "load_models_gpu", forbidden)
    monkeypatch.setattr(management, "free_memory", forbidden)
    decoded = audit._decode_owned_pixels(vae, torch.zeros(1, 24, 7, 2, 3), lambda x: x, 22)
    assert float(decoded.max()) == 0.25
    assert vae.managed_calls == 0


@pytest.mark.parametrize("state", ["absent", "other_device"])
@pytest.mark.parametrize("fits", [True, False])
def test_audit_loads_a_nonresident_vae_only_into_free_memory(monkeypatch, state, fits):
    pytest.importorskip("comfy.cli_args").args.cpu = True
    management = pytest.importorskip("comfy.model_management")
    vae = ManagedVAE()
    vae.patcher.loaded_size = lambda: 0
    vae.memory_used_decode = lambda shape, dtype: 4096
    monkeypatch.setattr(management, "loaded_models", lambda: [] if state == "absent" else [vae.patcher])
    if state == "other_device":
        vae.patcher.current_loaded_device = lambda: torch.device("cuda:0")
    monkeypatch.setattr(management, "minimum_inference_memory", lambda: 2048)
    monkeypatch.setattr(management, "extra_reserved_memory", lambda: 0)
    monkeypatch.setattr(management, "get_free_memory", lambda device: (math.ceil(1024 * 1.1) + 4096) if fits else 4096)
    requests = []
    monkeypatch.setattr(
        management,
        "load_models_gpu",
        lambda models, memory_required=0, **kwargs: requests.append((tuple(models), memory_required)),
    )
    monkeypatch.setattr(management, "free_memory", lambda *args, **kwargs: pytest.fail("audit freed memory"))
    latent = torch.zeros(1, 24, 7, 2, 3)
    if fits:
        decoded = audit._decode_owned_pixels(vae, latent, lambda x: x, 22)
        assert float(decoded.max()) == 0.25
        assert requests == [((vae.patcher,), 4096)]
    else:
        with pytest.raises(RuntimeError, match="without unloading other models"):
            audit._decode_owned_pixels(vae, latent, lambda x: x, 22)
        assert requests == []
    assert vae.managed_calls == 0


@pytest.mark.parametrize("other_device", [False, True])
def test_audit_admission_accounts_for_core_weight_reserve_and_device(monkeypatch, other_device):
    pytest.importorskip("comfy.cli_args").args.cpu = True
    management = pytest.importorskip("comfy.model_management")
    vae = ManagedVAE()
    vae.patcher.loaded_size = lambda: 1024 if other_device else 0
    if other_device:
        vae.patcher.current_loaded_device = lambda: torch.device("cuda:0")
    vae.memory_used_decode = lambda shape, dtype: 4096
    monkeypatch.setattr(management, "loaded_models", lambda: [])
    monkeypatch.setattr(management, "minimum_inference_memory", lambda: 2048)
    monkeypatch.setattr(management, "extra_reserved_memory", lambda: 0)
    monkeypatch.setattr(management, "get_free_memory", lambda device: 1024 + 4096)
    monkeypatch.setattr(management, "load_models_gpu", lambda *a, **kw: pytest.fail("insufficient admission budget"))
    with pytest.raises(RuntimeError, match="without unloading other models"):
        audit._resident_decode(vae, torch.zeros(1, 24, 7, 2, 3))


def test_full_video_decoder_context_validates_same_saved_pixels_without_production_mutation(bundle):
    directory, _manifest, _video = bundle
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    vae = FakeVAE()
    report = audit.audit_local_boundary(
        vae,
        directory,
        175,
        lambda latent: latent,
        validate_full_video_decoder=True,
    )
    comparison = report["full_video_decoder_context_validation"]
    assert report["full_video_decoder_comparison_requested"] is True
    assert report["extra_vae_calls"] == 6
    assert len(vae.inputs) == 6
    assert tuple(vae.inputs[-1].shape) == tuple(_video.shape)
    assert torch.equal(vae.inputs[-1], _video)
    assert comparison["policy"] == "h3_native_full_vs_crop_decoder_window_v1"
    assert comparison["full_decoded_global_origin"] == 136
    assert comparison["cropped_replay_global_origin"] == 153
    assert comparison["frame_labels"] == list(range(170, 196))
    assert comparison["full_decoded_frames"] == 73
    assert comparison["same_saved_final_clean_state"] is True
    assert comparison["same_connected_native_video_vae"] is True
    assert comparison["extra_h3_nfe"] == 0
    assert comparison["extra_vae_calls"] == 1
    assert comparison["per_frame_rgb_difference_rms"] == [0.0] * 26
    assert comparison["per_frame_luma_mean_change"] == [0.0] * 26
    assert not comparison["production_output_modified"]
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    json.dumps(report, allow_nan=False)


def test_full_video_decoder_context_defaults_off_with_no_extra_decode(bundle):
    directory, _manifest, _video = bundle
    vae = FakeVAE()
    report = audit.audit_local_boundary(vae, directory, 175, lambda latent: latent)
    assert report["full_video_decoder_comparison_requested"] is False
    assert "full_video_decoder_context_validation" not in report
    assert report["extra_vae_calls"] == 5
    assert len(vae.inputs) == 5


def test_full_video_decoder_validation_rejects_non_boolean_and_high_tone_scope(bundle):
    directory, _manifest, _video = bundle
    vae = FakeVAE()
    with pytest.raises(TypeError, match="boolean"):
        audit.audit_local_boundary(
            vae,
            directory,
            175,
            lambda v: v,
            validate_full_video_decoder=1,
        )
    with pytest.raises(ValueError, match="requires stage_continuity"):
        audit.audit_local_boundary(
            vae,
            directory,
            175,
            lambda v: v,
            scope="high_prediction_tone",
            validate_full_video_decoder=True,
        )
    assert vae.inputs == []
