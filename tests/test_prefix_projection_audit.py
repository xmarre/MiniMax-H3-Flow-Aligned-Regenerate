import ast
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate import prefix_projection_audit as audit
from h3_flow_regenerate.geometry import resize_spatial_5d


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
    video = torch.rand((1, 24, 22, 8, 10), generator=torch.Generator().manual_seed(5)) * 0.2 + 0.2
    prefix = video[:, :, :12].clone()
    source = resize_spatial_5d(video, 6, 8)
    manifest = {
        "schema": 1,
        "kind": "h3_flow_native_boundary_decoder_window_evidence",
        "metadata": {
            "policy": "native_boundary_decoder_window_evidence_v1",
            "domain": "model_internal_clean_except_sampler_input_and_mask",
            "spatial_stage_control": "progressive_uniform_source",
            "source_prefix_projection_policy": "half_pixel_latent_v1",
            "first_high_actual": True,
            "provider_clean_provenance": "actual_learned_provider_uniform_source",
            "process_latent_out_required_before_vae": True,
            "full_video_snapshots": True,
            "full_video_temporal_start_t": 0,
            "low_probe_native_carrier_decodable": True,
            "target_band_tokens": 0,
            "target_band_transfer_start_t": 12,
            "source_probe_clean_grid": [6, 8],
            "window": {"prefix_t": 12, "temporal": 22, "decoded_trim_frames": 39},
        },
        "tensor_bytes": {},
    }
    write_operand(tmp_path, manifest, "authoritative_prefix_full", prefix)
    write_operand(tmp_path, manifest, "source_probe_clean_full", source)
    write_operand(tmp_path, manifest, "final_post_high_internal_clean_full", video)
    mask = torch.ones_like(video)
    mask[:, :, :12] = 0
    write_operand(tmp_path, manifest, "initial_high_video_mask_full", mask)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest, prefix


class FakeVAE:
    """Spatial RGB/latent mismatch with an invertible test RGB encoder."""

    def __init__(self, *, bias=0.03, context_bias=0, encoded_t_delta=0):
        self.bias = bias
        self.context_bias = context_bias
        self.encoded_t_delta = encoded_t_delta
        self.decode_calls = []
        self.encode_calls = []

    def decode(self, latent):
        self.decode_calls.append(latent.clone())
        frames = (latent.shape[2] - 2) // 5 * 17 + 5
        x = latent[:, :3, :12].mean(2)
        x = F.interpolate(x, scale_factor=16, mode="bilinear", align_corners=False)
        x = x.movedim(1, -1)[:, None].expand(-1, frames, -1, -1, -1).clone()
        if latent.shape[-2] == 6:
            x += self.bias
        if latent.shape[2] > 12:
            x += self.context_bias
        return x

    def encode(self, frames):
        self.encode_calls.append(frames.clone())
        value = F.interpolate(frames.mean(0).movedim(-1, 0)[None], size=(6, 8), mode="area") - self.bias
        return value.repeat(1, 8, 1, 1)[:, :, None].expand(-1, -1, 12 + self.encoded_t_delta, -1, -1).clone()


def test_projection_ab_verifies_saved_source_and_preserves_all_bundle_bytes(bundle):
    directory, _, prefix = bundle
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    vae = FakeVAE()
    conversions = []

    def process_out(value):
        conversions.append(value.shape)
        value.add_(0.01)  # Exercise nonidentity/mutating boundary conversions on owned copies.
        return value

    report, reference, legacy, candidate = audit.audit_prefix_projection(
        vae,
        directory,
        175,
        process_out,
        rois={},
        feature_tracking=False,
    )
    assert report["frame_labels"] == list(range(136, 175))
    assert reference.shape == legacy.shape == candidate.shape == (39, 96, 128, 3)
    assert len(vae.decode_calls) == 5 and len(vae.encode_calls) == 1
    assert len(conversions) == 4  # Never process_out on native encode's normalized output.
    assert torch.equal(vae.encode_calls[0], reference)
    assert torch.equal(prefix, audit.load_prefix_operands(directory, 175)[0])
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    assert report["authoritative_prefix_preserved_bitwise"]
    assert report["extra_h3_nfe"] == 0
    a = report["comparisons"]["latent_bicubic"]["regions"]["full_frame"]["summary"]
    b = report["comparisons"]["vae_rgb_roundtrip"]["regions"]["full_frame"]["summary"]
    assert b["luma_abs_bias_mean"] < a["luma_abs_bias_mean"]
    json.dumps(report, allow_nan=False)


def test_context_control_reveals_prefix_terminal_padding_confound(bundle):
    directory, _, _ = bundle
    report, *_ = audit.audit_prefix_projection(
        FakeVAE(context_bias=0.04),
        directory,
        175,
        lambda x: x,
        rois={},
        feature_tracking=False,
    )
    controls = report["decoder_context_controls"]
    assert controls["following_saved_tokens"] == 5
    for name in ("target_prefix_isolated_vs_saved_future", "source_prefix_isolated_vs_saved_future"):
        assert controls[name]["full_frame"]["summary"]["luma_bias_mean"] == pytest.approx(0.04, abs=1e-6)
    assert set(report["comparisons_against_target_with_saved_future"]) == {"latent_bicubic", "vae_rgb_roundtrip"}


@pytest.mark.parametrize("failure", ["hash", "prefix", "mask", "source", "phase", "policy", "domain", "path"])
def test_invalid_bundle_stops_before_vae_work(bundle, failure):
    directory, manifest, _prefix = bundle
    if failure == "hash":
        path = directory / "authoritative_prefix_full.bin"
        raw = bytearray(path.read_bytes())
        raw[-1] ^= 1
        path.write_bytes(raw)
    elif failure in ("prefix", "mask", "source"):
        name = {
            "prefix": "final_post_high_internal_clean_full",
            "mask": "initial_high_video_mask_full",
            "source": "source_probe_clean_full",
        }[failure]
        entry = manifest["tensor_bytes"][name]
        x = torch.frombuffer(bytearray((directory / entry["file"]).read_bytes()), dtype=torch.float32).reshape(
            entry["shape"]
        )
        x[:, :, 0] += 0.1
        write_operand(directory, manifest, name, x)
    elif failure == "phase":
        manifest["metadata"]["window"]["decoded_trim_frames"] = 38
    elif failure == "policy":
        manifest["metadata"]["source_prefix_projection_policy"] = "h3_patch_lattice_v1"
    elif failure == "domain":
        manifest["metadata"]["domain"] = "sampler_external"
    else:
        manifest["tensor_bytes"]["authoritative_prefix_full"]["file"] = "../outside.bin"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    vae = FakeVAE()
    with pytest.raises(ValueError):
        audit.audit_prefix_projection(vae, directory, 175, lambda x: x, rois={}, feature_tracking=False)
    assert not vae.decode_calls and not vae.encode_calls


@pytest.mark.parametrize("delta", [-1, 1])
def test_native_encoder_cannot_silently_change_protected_token_count(bundle, delta):
    directory, _, _ = bundle
    vae = FakeVAE(encoded_t_delta=delta)
    with pytest.raises(ValueError, match="temporal ownership"):
        audit.audit_prefix_projection(vae, directory, 175, lambda x: x, rois={}, feature_tracking=False)
    assert len(vae.encode_calls) == 1 and len(vae.decode_calls) == 4


def test_common_grid_errors_reject_smoother_but_inaccurate_reconstruction():
    x = torch.rand(3, 40, 48, 3, generator=torch.Generator().manual_seed(42)) * 0.7 + 0.15
    smooth = F.avg_pool2d(x.movedim(-1, 1), 5, stride=1, padding=2).movedim(1, -1)
    exact = audit.paired_rgb_metrics(x, x, [0, 1, 2], {})["full_frame"]["summary"]
    blurred = audit.paired_rgb_metrics(x, smooth, [0, 1, 2], {})["full_frame"]["summary"]
    assert exact["rgb_rmse"] == 0 and exact["sobel_error_rms"] == 0
    assert exact["luma_ssim_mean"] == pytest.approx(1)
    assert blurred["sobel_error_rms"] > 0.01
    assert blurred["highpass_3_error_rms"] > 0.01
    assert blurred["luma_ssim_mean"] < 0.5
    assert blurred["temporal_rgb_error_increment_rmse_mean"] > 0


def test_rgb_resize_is_spatial_only_and_declares_clipping():
    rgb = torch.stack([torch.full((96, 128, 3), n / 4) for n in range(4)])
    resized, clipping = audit.resize_rgb(rgb, 64, 96)
    assert resized.shape == (4, 64, 96, 3)
    assert torch.allclose(resized[:, 0, 0, 0], torch.arange(4) / 4, atol=1e-6)
    assert clipping == 0


def test_same_time_geometry_reports_translation_and_indeterminate_texture():
    pytest.importorskip("cv2")
    torch.manual_seed(1)
    x = torch.rand(2, 128, 160, 3)
    candidate = torch.roll(x, (1, 2), dims=(1, 2))
    rois = {"left": (0, 0, 0.35, 1), "right": (0.65, 0, 1, 1)}
    fit = audit._same_time_geometry(x, candidate, [170, 171], rois)
    assert all(row["status"] == "measured" for row in fit["frames"])
    assert all(row["left_median_dx_px"] == pytest.approx(2, abs=0.1) for row in fit["frames"])
    flat = audit._same_time_geometry(torch.ones_like(x), torch.ones_like(x), [170, 171], rois)
    assert all(row["status"] == "insufficient_initial_landmarks" for row in flat["frames"])


def native_timing():
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        pytest.skip("native VAE source oracle requires COMFYUI_ROOT")
    source = Path(root) / "comfy/ldm/minimax/vae.py"
    tree = ast.parse(source.read_text())
    native = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3VideoVAE")
    wanted = {
        "encode_temporal",
        "encode",
        "decode",
        "decode_temporal",
        "decode_output_shape",
        "blend",
        "_decode_temporal_pad_frames",
        "_decode_temporal_frame_plan",
        "_decode_temporal_chunks",
        "_normalize_pixels",
        "_finalize_pixels",
    }
    body = [n for n in native.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    assert {n.name for n in body} == wanted
    namespace = {
        "torch": torch,
        "math": math,
        "comfy": SimpleNamespace(model_management=SimpleNamespace(intermediate_device=lambda: "cpu")),
    }
    module = ast.Module(
        body=[ast.ClassDef(name="NativeTiming", bases=[], keywords=[], body=body, decorator_list=[])], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), namespace)
    model = namespace["NativeTiming"]()
    for key, value in dict(
        tokens_chunk_size=5,
        token_overlap=2,
        vae_ratio_t=4,
        vae_ratio=16,
        token_drop=3,
        frame_pre_padding=3,
        clip_length=17,
        frame_overlap=5,
        still_frame=None,
    ).items():
        setattr(model, key, value)
    model.pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1, 1)
    model.pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1, 1)
    model.decoder = SimpleNamespace(out_channels=3)
    model.latents_mean = torch.linspace(-1, 1, 24)
    model.latents_std = torch.linspace(0.5, 2, 24)
    return model


@pytest.mark.parametrize("frames,tokens", [(5, 2), (22, 7), (39, 12), (56, 17)])
def test_actual_native_encoder_decoder_timing_normalization_and_axes(frames, tokens):
    model = native_timing()
    clips = []

    def encode_moments(x):
        clips.append(x.clone())
        # Synthetic encoder activations; real Core temporal IO and normalization.
        level = x.mean().expand(1, 24, 5, x.shape[-2] // 16, x.shape[-1] // 16)
        return torch.cat((level, torch.full_like(level, 100)), 1)

    model._adaptive_encode = encode_moments
    x = torch.arange(frames * 3, dtype=torch.float32).reshape(1, 3, frames, 1, 1) / (frames * 3)
    x = x.expand(1, 3, frames, 32, 64).clone()
    rgb = x.clone()
    encoded = model.encode(x * 2 - 1, device="cpu")
    assert encoded.shape == (1, 24, tokens, 2, 4)
    assert model.decode_output_shape(encoded.shape) == (1, 3, frames, 32, 64)
    assert len(clips) == math.ceil(frames / 17)
    assert all(clip.shape == (1, 3, 17, 32, 64) for clip in clips)
    expected = (rgb[:, :, :17] - model.pixel_mean) / model.pixel_std
    if frames < 17:
        expected = torch.cat((expected, expected[:, :, -1:].expand(-1, -1, 17 - frames, -1, -1)), 2)
    assert torch.allclose(clips[0], expected, atol=1e-6)
    first_mean = encoded[:, :, :1] * model.latents_std.view(1, 24, 1, 1, 1) + model.latents_mean.view(1, 24, 1, 1, 1)
    assert torch.allclose(first_mean, clips[0].mean().expand_as(first_mean), atol=1e-6)
    model._adaptive_decode = lambda z: (
        z[:, :3].repeat_interleave(4, 2).repeat_interleave(16, 3).repeat_interleave(16, 4)
    )
    decoded = model.decode(encoded)
    assert decoded.shape == (1, 3, frames, 32, 64)
    assert bool(((decoded >= 0) & (decoded <= 1)).all())


def test_resident_encoder_uses_native_input_scaling_and_never_managed_admission(monkeypatch):
    requests = []
    patcher = SimpleNamespace(current_loaded_device=lambda: torch.device("cpu"))
    management = SimpleNamespace(
        loaded_models=lambda: [patcher],
        cuda_device_context=lambda device: torch.inference_mode(),
        load_models_gpu=lambda *a, **k: requests.append((a, k)),
        OOM_EXCEPTION=torch.cuda.OutOfMemoryError,
    )
    monkeypatch.setitem(sys.modules, "comfy.model_management", management)
    # Python's import also needs the parent package when Core isn't installed.
    parent = sys.modules.get("comfy")
    if parent is None:
        monkeypatch.setitem(sys.modules, "comfy", SimpleNamespace(model_management=management))
    else:
        monkeypatch.setattr(parent, "model_management", management, raising=False)
    calls = []

    def native_encode(x, device):
        calls.append((x.clone(), device))
        return torch.zeros(1, 24, 12, 6, 8)

    vae = SimpleNamespace(
        patcher=patcher,
        device=torch.device("cpu"),
        output_device=torch.device("cpu"),
        vae_dtype=torch.float32,
        vae_output_dtype=lambda: torch.float32,
        process_input=lambda x: x * 2 - 1,
        first_stage_model=SimpleNamespace(encode=native_encode),
        encode=lambda x: pytest.fail("entered managed VAE.encode"),
    )
    images = torch.full((39, 96, 128, 3), 0.75)
    encoded = audit._encode_owned_pixels(vae, images, (1, 24, 12, 6, 8))
    assert encoded.shape == (1, 24, 12, 6, 8)
    assert calls[0][0].shape == (1, 3, 39, 96, 128)
    assert torch.equal(calls[0][0], torch.full_like(calls[0][0], 0.5))
    assert torch.equal(images, torch.full_like(images, 0.75))
    assert requests == []

    def oom(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("encoder")

    vae.first_stage_model.encode = oom
    with pytest.raises(RuntimeError, match="Resident models were preserved"):
        audit._encode_owned_pixels(vae, images, (1, 24, 12, 6, 8))
    assert requests == []
