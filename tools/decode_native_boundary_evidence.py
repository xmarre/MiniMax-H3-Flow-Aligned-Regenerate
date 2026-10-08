#!/usr/bin/env python3
"""Decode saved H3 stage operands with Core's native seven-token VAE window.

This is an offline counterfactual replay, independent of production sampling.
The preceding decoder window is absent from this bundle: eight retained frames
are replayed, but the preceding temporal blend is not.
Target-band auxiliary full-video and low-grid snapshots are not replayed here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch

TENSOR_NAMES = frozenset(
    {
        "authoritative_prefix",
        "provider_native_clean",
        "pre_high_exact_restored",
        "first_high_sampler_input",
        "initial_high_video_mask",
        "first_high_before_flow",
        "first_high_after_flow",
        "final_post_high_internal_clean",
    }
)


def load_bundle(directory: Path) -> tuple[dict, dict[str, torch.Tensor]]:
    directory = directory.resolve()
    manifest = json.loads((directory / "manifest.json").read_text())
    metadata = manifest["metadata"]
    band = metadata.get("provider_clean_provenance") == "actual_learned_provider_before_target_band_splice"
    if (
        manifest.get("schema") != 1
        or manifest.get("kind") != "h3_flow_native_boundary_decoder_window_evidence"
        or metadata.get("policy") != "native_boundary_decoder_window_evidence_v1"
        or metadata.get("first_high_actual") is not True
        or metadata.get("decoder_comparison_prefix") != "replace_with_authoritative_prefix_bytes"
        or metadata.get("process_latent_out_required_before_vae") is not True
        or (not band and metadata.get("provider_clean_provenance") != "actual_clean_postprocess")
    ):
        raise ValueError("unsupported native boundary evidence contract")
    entries = manifest["tensor_bytes"]
    expected_names = TENSOR_NAMES
    if band:
        tokens = metadata.get("target_band_tokens")
        transfer_start = metadata.get("target_band_transfer_start_t")
        window = metadata["window"]
        if (
            type(tokens) is not int
            or tokens <= 0
            or type(transfer_start) is not int
            or transfer_start != window["prefix_t"] + tokens
            or transfer_start >= window["temporal"]
            or metadata.get("full_video_snapshots") is not True
            or metadata.get("full_video_temporal_start_t") != 0
            or metadata.get("low_probe_native_carrier_decodable") is not False
        ):
            raise ValueError("unsupported target-band evidence contract")
        auxiliary = {"source_probe_clean", "low_probe_native_carrier_clean"}
        expected_names = TENSOR_NAMES | auxiliary | {name + "_full" for name in TENSOR_NAMES | auxiliary}
    # Optional per-call high prediction windows are declared by the bundle's
    # bounded trace. They are not replayed here, but must be exactly the
    # declared later-call names; the first call reuses the first_high operands.
    trace = metadata.get("window", {}).get("high_prediction_trace")
    traced = set()
    if trace is not None:
        if not isinstance(trace, dict) or trace.get("policy") != "bounded_high_prediction_windows_v1":
            raise ValueError("unsupported high prediction trace")
        for call in trace.get("calls", ()):
            for point in ("before_flow", "after_flow"):
                name = call.get(point)
                if name not in {f"first_high_{point}", f"high_prediction_{call.get('call_index', -1):02d}_{point}"}:
                    raise ValueError("high prediction trace names an unexpected operand")
                traced.add(name)
        traced -= TENSOR_NAMES
    if set(entries) != expected_names | traced:
        raise ValueError("native boundary evidence must contain all eight stage operands")
    values = {}
    # Only the eight validated target-grid window operands enter this replay.
    # Auxiliary full timelines and padded carriers have separate geometries.
    for name in TENSOR_NAMES:
        entry = entries[name]
        path = (directory / entry["file"]).resolve()
        if not path.is_relative_to(directory):
            raise ValueError(f"tensor path leaves the evidence bundle: {name}")
        shape = entry["shape"]
        if (
            entry.get("dtype") != "torch.float32"
            or entry.get("byte_order") != "native_torch_contiguous"
            or len(shape) != 5
            or any(type(n) is not int or n <= 0 for n in shape)
            or shape[:2] != [1, 24]
            or entry.get("nbytes") != math.prod(shape) * 4
        ):
            raise ValueError(f"unsupported native tensor geometry or dtype: {name}")
        raw = path.read_bytes()
        if len(raw) != entry["nbytes"] or hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError(f"tensor bytes do not match the manifest: {name}")
        value = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(shape)
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"native tensor contains non-finite values: {name}")
        values[name] = value
    window = metadata["window"]
    shape = values["provider_native_clean"].shape
    prefix_shape = values["authoritative_prefix"].shape
    if (
        window["window_tokens"] != 7
        or window["token_overlap"] != 2
        or window["chunk_stride_tokens"] != 5
        or window["window_stop_t"] - window["window_start_t"] != 7
        or window["prefix_t"] - window["window_start_t"] != 2
        or window["window_start_t"] % 5
        or window["window_start_t"] < 0
        or window["window_stop_t"] > window["temporal"]
        or window["decoder_chunk_output_start_frame"] != 17 * (window["window_start_t"] // 5)
        or window["decoded_trim_frames"] != window["decoder_chunk_output_start_frame"] + 5
        or window["first_retained_local_frame"] != 5
        or shape[2] != 7
        or prefix_shape != (1, 24, 2, *shape[-2:])
        or any(v.shape != shape for name, v in values.items() if name != "authoritative_prefix")
    ):
        raise ValueError("native boundary window geometry drifted")
    for name in ("pre_high_exact_restored", "final_post_high_internal_clean"):
        if not torch.equal(
            values[name][:, :, :2].contiguous().view(torch.int32),
            values["authoritative_prefix"].view(torch.int32),
        ):
            raise ValueError(f"protected clean prefix differs from authoritative bytes: {name}")
    expected_mask = torch.ones_like(values["initial_high_video_mask"])
    expected_mask[:, :, :2] = 0
    if not torch.equal(values["initial_high_video_mask"], expected_mask):
        raise ValueError("initial high video mask does not protect exactly the two carried tokens")
    return manifest, values


def clean_stages(values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    stages = {"provider_native": values["provider_native_clean"].clone()}
    for name, source in (
        ("provider_exact_prefix", "provider_native_clean"),
        ("pre_high_dc", "pre_high_exact_restored"),
        ("first_high_before_flow", "first_high_before_flow"),
        ("first_high_after_flow", "first_high_after_flow"),
        ("final", "final_post_high_internal_clean"),
    ):
        stage = values[source].clone()
        stage[:, :, :2] = values["authoritative_prefix"]
        stages[name] = stage
    return stages


def import_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def native_tile_selection(vae, latent_hw: tuple[int, int], tile_yx: tuple[int, int]) -> dict:
    """Locate an unchanged native tile and its spatially unblended interior."""
    rows, heights, overlaps_y = vae.split_tiles(latent_hw[0] * vae.vae_ratio)
    cols, widths, overlaps_x = vae.split_tiles(latent_hw[1] * vae.vae_ratio)
    row, col = tile_yx
    if not 0 <= row < len(rows) or not 0 <= col < len(cols):
        raise ValueError("native tile index is outside the full decoder lattice")
    top = overlaps_y[row - 1] if row else 0
    bottom = heights[row] - (overlaps_y[row] if row < len(rows) - 1 else 0)
    left = overlaps_x[col - 1] if col else 0
    right = widths[col] - (overlaps_x[col] if col < len(cols) - 1 else 0)
    if top >= bottom or left >= right:
        raise ValueError("native tile has no interior free of neighboring spatial blend")
    return {
        "tile_index_yx": [row, col],
        "tile_latent_origin_yx": [rows[row] // vae.vae_ratio, cols[col] // vae.vae_ratio],
        "tile_latent_hw": [heights[row] // vae.vae_ratio, widths[col] // vae.vae_ratio],
        "tile_pixel_origin_yx": [rows[row], cols[col]],
        "interior_local_yx_bounds": [top, bottom, left, right],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--core-checkout", type=Path, required=True)
    parser.add_argument("--continuum-checkout", type=Path, required=True)
    parser.add_argument("--vae", type=Path, required=True, help="unquantized native video VAE safetensors")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--stage-batch-size", type=int, default=1, help="maximum simultaneous stage canvases")
    parser.add_argument("--stages", default="", help="optional comma-separated clean stage names")
    parser.add_argument(
        "--native-tile",
        type=int,
        nargs=2,
        metavar=("ROW", "COL"),
        help="decode one unchanged native tile; report its unblended interior separately",
    )
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("threads must be positive")
    if args.stage_batch_size < 1:
        parser.error("stage batch size must be positive")
    manifest, values = load_bundle(args.bundle)
    stages = clean_stages(values)
    if args.stages:
        names = args.stages.split(",")
        if len(set(names)) != len(names) or not set(names) <= set(stages):
            parser.error("stages must be distinct available clean stage names")
        stages = {name: stages[name] for name in names}
    sys.path.insert(0, str(args.core_checkout.resolve()))
    import comfy.cli_args

    comfy.cli_args.args.cpu = args.device == "cpu"
    comfy.cli_args.args.disable_dynamic_vram = True
    comfy.cli_args.args.use_pytorch_cross_attention = True
    import comfy.ldm.minimax.vae as native
    import comfy.model_management
    import comfy.ops
    from comfy.latent_formats import MiniMaxH3Video
    from safetensors import safe_open

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA replay was requested but CUDA is unavailable")
    dtype = getattr(torch, args.dtype)
    torch.set_num_threads(args.threads)
    latent_format = MiniMaxH3Video()
    if latent_format.scale_factor != 1.0:
        raise RuntimeError("native video latent output conversion changed")
    with safe_open(args.vae, framework="pt", device="cpu") as sf:
        keys = sf.keys()
        if any("quant" in k and k.startswith("decoder.") for k in keys):
            raise ValueError("this replay requires an unquantized native VAE checkpoint")
        selected = {
            k: sf.get_tensor(k)
            for k in keys
            if k.startswith(("decoder.", "post_quant_conv.")) or k in ("latents_mean", "latents_std")
        }
        layers = sum(k.endswith(".scale1") for k in selected)
        if layers < 1 or any(v.dtype not in (torch.float16, torch.float32) for v in selected.values()):
            raise ValueError("unsupported native decoder checkpoint")
        with torch.device("meta"):
            vae = native.MiniMaxH3VideoVAE(operations=comfy.ops.manual_cast, num_layers=layers)
        geometry = (
            vae.clip_length,
            vae.tokens_chunk_size,
            vae.token_overlap,
            vae.frame_pre_padding,
            vae.vae_ratio_t,
            vae.vae_ratio,
            vae.tile_size,
        )
        if geometry != (17, 5, 2, 3, 4, 16, 256):
            raise ValueError("Core's native video VAE geometry changed")
        loaded = vae.load_state_dict(selected, strict=False, assign=True)
        if loaded.unexpected_keys or any(not k.startswith(("encoder.", "quant_conv.")) for k in loaded.missing_keys):
            raise ValueError("native decoder checkpoint is incomplete or incompatible")
        for module in vae.modules():
            if hasattr(module, "comfy_cast_weights"):
                module.comfy_cast_weights = True
        vae.decoder.pos_embed = native.RotaryEmbeddingND(48, rotary_base=100.0, n_dim=3)
        for block in vae.decoder.transformer_blocks:
            block.attn.qk_norm_scale = torch.ones(64)
        vae.pixel_mean = torch.tensor(native.IMAGENET_MEAN).view(1, 3, 1, 1, 1)
        vae.pixel_std = torch.tensor(native.IMAGENET_STD).view(1, 3, 1, 1, 1)
        vae.eval().requires_grad_(False)
        if args.device == "cpu":
            # Bound transport to one native tile per stage. Each batch element
            # still executes the complete native spatial/temporal tile.
            comfy.model_management.get_free_memory = lambda device: 64 * 2**20
        else:
            vae.decoder.to(args.device)
            vae.post_quant_conv.to(args.device)
        first = vae.frame_pre_padding + manifest["metadata"]["window"]["first_retained_local_frame"]
        started = time.monotonic()
        calls = [0]

        def progress(module, operands, output):
            calls[0] += 1
            print(f"native tile batch {calls[0]}: {time.monotonic() - started:.1f}s", flush=True)

        vae.decoder.register_forward_hook(progress)
        args.output.mkdir(parents=True, exist_ok=True)
        trajectory_path = args.continuum_checkout / "v3/trajectory_diagnostics.py"
        tone_path = args.continuum_checkout / "v3/video_tone.py"
        trajectory = import_source(trajectory_path, "native_continuum_trajectory")
        tone = import_source(tone_path, "native_continuum_tone")
        selection = None
        if args.native_tile is not None:
            selection = native_tile_selection(vae, tuple(values["provider_native_clean"].shape[-2:]), args.native_tile)
        report = {
            "policy": "offline_native_boundary_stage_decode_v1",
            "manifest_sha256": hashlib.sha256((args.bundle / "manifest.json").read_bytes()).hexdigest(),
            "vae_sha256": sha256_file(args.vae),
            "core_vae_source_sha256": hashlib.sha256(Path(native.__file__).read_bytes()).hexdigest(),
            "trajectory_source_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
            "tone_source_sha256": hashlib.sha256(tone_path.read_bytes()).hexdigest(),
            "decoder_dtype": args.dtype,
            "decoder_device": args.device,
            "torch_version": str(torch.__version__),
            "stage_batch_size": args.stage_batch_size,
            "attention": "native_pytorch_sdpa",
            "full_frame_spatial_blend_reproduced": selection is None,
            "preceding_temporal_blend_reproduced": False,
            "first_retained_raw_decoder_frame": first,
            "first_boundary_pair_uses_raw_unblended_prefix": True,
            "production_sampling_rerun": False,
            "rendered_acceptance": False,
            "stages": {},
        }
        if selection is not None:
            report.update(
                spatial_scope="one native tile; interior has no contribution from neighboring spatial tiles",
                spatial_selection=selection,
            )
        names = list(stages)
        for start in range(0, len(names), args.stage_batch_size):
            batch_names = names[start : start + args.stage_batch_size]
            z = latent_format.process_out(torch.cat([stages[name] for name in batch_names])).to(
                device=args.device, dtype=dtype
            )
            z = z * vae.latents_std.view(1, -1, 1, 1, 1).to(z) + vae.latents_mean.view(1, -1, 1, 1, 1).to(z)
            with torch.inference_mode():
                if selection is None:
                    raw = vae._adaptive_decode(z)
                else:
                    y, x = selection["tile_latent_origin_yx"]
                    h, w = selection["tile_latent_hw"]
                    raw = vae._decode_pixels(z[..., y : y + h, x : x + w])
                pixels = vae._finalize_pixels(raw[:, :, first - 2 : first + 8]).cpu()
                del raw, z
            for index, name in enumerate(batch_names):
                frames = pixels[index].permute(1, 2, 3, 0)
                motion = trajectory.measure_decoded_boundary_trajectory(
                    frames[:2], frames, trim_frames=2, boundary_global_frame=-1, forward_frames=8
                )
                report["stages"][name] = {
                    "trajectory": motion,
                    "tone": tone._profile(tone._sample_rgb(frames[2:])),
                }
                if selection is not None:
                    top, bottom, left, right = selection["interior_local_yx_bounds"]
                    interior = frames[:, top:bottom, left:right]
                    report["stages"][name]["native_spatial_interior"] = {
                        "trajectory": trajectory.measure_decoded_boundary_trajectory(
                            interior[:2], interior, trim_frames=2, boundary_global_frame=-1, forward_frames=8
                        ),
                        "tone": tone._profile(tone._sample_rgb(interior[2:])),
                    }
                # A view serializes its complete backing batch, including the
                # other stages. Own only this stage's pixels in its artifact.
                torch.save(pixels[index].clone(), args.output / f"{name}.pt")
            del frames, pixels
        report["wall_s"] = time.monotonic() - started
        (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(args.output / "report.json")


if __name__ == "__main__":
    main()
