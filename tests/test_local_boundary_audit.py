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
    assert (plan["token_start"], plan["token_stop"]) == (10, 22)
    assert plan["decoded_origin_frame"] == 170
    assert plan["shared_tokens"] == [15, 17]
    assert plan["temporal_blend_local_frames"] == [17, 22]
    assert 188 in range(plan["decoded_origin_frame"] + 14, plan["decoded_origin_frame"] + 26)
    for value in stages.values():
        assert torch.equal(value[:, :, :2], video[:, :, 10:12])
        assert value.untyped_storage().nbytes() == value.numel() * 4
    assert torch.equal(stages["provider"][:, :, 2:6], video[:, :, 12:16] + 0.1)
    assert torch.equal(stages["pre_high"][:, :, 2:6], video[:, :, 12:16])
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


class FakeVAE:
    def __init__(self):
        self.first_stage_model = SimpleNamespace(
            tokens_chunk_size=5, token_overlap=2, frame_pre_padding=3, clip_length=17
        )
        self.inputs = []

    def decode(self, value):
        self.inputs.append(value.clone())
        return torch.zeros(1, 39, value.shape[-2] * 16, value.shape[-1] * 16, 3)


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
    assert report["extra_vae_calls"] == 5
    assert report["temporal_blend_reproduced_for_measured_frames"] is True
    assert report["decoded_pixels_saved"] is False
    assert report["rendered_acceptance"] is False
    assert len(vae.inputs) == len(converted) == 5
    for actual, internal in zip(vae.inputs, converted, strict=True):
        assert torch.equal(actual, internal * 0.5)
    assert torch.equal(torch.random.get_rng_state(), rng)
    assert report["stages"]["final"]["frame_labels"] == list(range(184, 196))
    assert len(report["comparisons"]) == 4
    for stage in report["stages"].values():
        assert stage["adjacent_rgb_difference_rms"] == [0.0] * 12
        assert stage["adjacent_luma_mean_change"] == [0.0] * 12
        assert stage["adjacent_frame_geometry"]["frames"] == list(range(184, 196))
        assert stage["adjacent_frame_geometry_upper45"]["frames"] == list(range(184, 196))
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
            times = torch.arange(39, dtype=pixels.dtype)[None, :, None, None, None]
            pixels += 0.1 + times * 0.001  # Same smooth motion in every stage.
            if stage >= 1:
                pixels += 0.02  # Stable provider-to-pre-high bias, no temporal edge.
            if stage >= 2:
                pixels[:, 18:] += 0.1  # First high creates a step at assembled frame 188.
            return pixels

    vae = TemporalVAE()
    rng = torch.random.get_rng_state().clone()
    report = audit.audit_local_boundary(vae, directory, 175, lambda value: value)
    assert report["extra_vae_calls"] == len(vae.inputs) == 5
    assert torch.equal(torch.random.get_rng_state(), rng)
    index = report["stages"]["pre_high"]["frame_labels"].index(188)
    for stage in ("provider", "pre_high"):
        assert report["stages"][stage]["adjacent_rgb_difference_rms"] == pytest.approx([0.001] * 12, abs=1e-7)
    for stage in ("first_high_before_flow", "first_high_after_flow", "final"):
        expected = [0.001] * 12
        expected[index] = 0.101
        assert report["stages"][stage]["adjacent_rgb_difference_rms"] == pytest.approx(expected, abs=1e-7)
    for pair, measurement in report["comparisons"].items():
        expected = [0.0] * 12
        if pair == "pre_high_to_first_high_before_flow":
            expected[index] = 0.1
        assert measurement["temporal_increment_change_rms"] == pytest.approx(expected, abs=1e-7)


def test_node_saves_only_json_and_is_registered(bundle, monkeypatch, tmp_path):
    import sys

    directory, _manifest, _video = bundle
    output = tmp_path / "output"
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(output)))
    monkeypatch.setitem(sys.modules, "comfy", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "comfy.latent_formats",
        SimpleNamespace(MiniMaxH3Video=lambda: SimpleNamespace(process_out=lambda value: value)),
    )
    monkeypatch.setattr(audit, "geometry_comparison", lambda a, b, labels: {"frames": labels})
    result = audit.H3FlowLocalBoundaryAudit().audit(FakeVAE(), str(directory), 175)
    files = list(output.rglob("*.*"))
    assert len(files) == 1 and files[0].suffix == ".json"
    assert json.loads(files[0].read_text()) == json.loads(result["result"][0])
    assert audit.NODE_CLASS_MAPPINGS["H3FlowLocalBoundaryAudit"] is audit.H3FlowLocalBoundaryAudit


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


def test_two_window_crop_matches_production_native_temporal_decode_at_band_edge(bundle):
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
    assert torch.equal(replay[:, :, start:stop], production[:, :, 34 + start : 34 + stop])
    assert not torch.equal(replay[:, :, :5], production[:, :, 34:39])
