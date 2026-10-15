"""Opt-in counterfactual evaluations of the first actual target-high H3 call.

The capture repeats only the first actual high-stage prediction with one input
or model-stack change per arm. Each arm calls the guider's unwrapped CFG
prediction, so predict-noise wrappers (Spectrum step accounting, Flow guidance
and boundary traces) never observe it. Spectrum's per-step diffusion-model keys
are removed, the partitioned stage runtime is replaced by a copy with scratch
metrics, and caller RNG state is restored afterwards. Arm outputs are kept as
the native seven-token boundary decoder window and exported with the boundary
evidence bundle. The production prediction is returned unchanged.

Each arm costs one extra H3 evaluation (two with CFG) at target resolution.
Arms are causal tests of the first prediction, not a repair.
"""

from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any

import torch

from .geometry import pack_streams, unpack_streams
from .metrics import H3FlowMetrics

FIRST_HIGH_COUNTERFACTUAL_KEY = "h3_flow_first_high_counterfactual_v1"
FIRST_HIGH_COUNTERFACTUAL_POLICY = "first_high_counterfactual_v1"
FIRST_HIGH_COUNTERFACTUAL_ARMS = (
    "replay",
    "provider_prefix",
    "suffix_noise_reseed",
    "native_high_attention",
)
SPECTRUM_DIFFUSION_STEP_KEYS = ("spectrum_h3_runtime", "spectrum_h3_run_id", "spectrum_h3_step_id")
VISUAL_COND_TIMESTEP = 0.999
WINDOW_TOKENS = 7
WINDOW_CONTEXT_TOKENS = 2


def arm_tensor_name(arm: str) -> str:
    return f"first_high_counterfactual_{arm}"


@dataclasses.dataclass(frozen=True)
class FirstHighCounterfactualRequest:
    arms: tuple[str, ...]
    noise_seed: int

    def __post_init__(self):
        if not self.arms or len(set(self.arms)) != len(self.arms):
            raise ValueError("first-high counterfactual requires at least one distinct arm")
        unknown = [arm for arm in self.arms if arm not in FIRST_HIGH_COUNTERFACTUAL_ARMS]
        if unknown:
            raise ValueError(f"unsupported first-high counterfactual arms: {unknown}")
        if type(self.noise_seed) is not int or not 0 <= self.noise_seed < 2**63:
            raise ValueError("first-high counterfactual noise seed must be a non-negative 64-bit integer")


def _token_rms(value: torch.Tensor) -> list[float]:
    return value.float().square().mean(dim=(0, 1, 3, 4)).sqrt().tolist()


def _arm_options(model_options, *, stage_key, scratch_metrics, prefix=None, native_attention=False):
    local = dict(model_options or {})
    transformer = dict(local.get("transformer_options") or {})
    stripped = [key for key in SPECTRUM_DIFFUSION_STEP_KEYS if transformer.pop(key, None) is not None]
    runtime = transformer.get(stage_key)
    receipt = {"spectrum_step_keys_removed": stripped, "partitioned_runtime_present": runtime is not None}
    if native_attention:
        # Without the stage runtime the partitioned diffusion wrapper forwards
        # to the installed Core/VDN/Sol chain, including Core BSA's block
        # producer. Core inpaint has already injected the exact prefix.
        transformer.pop(stage_key, None)
        receipt["partitioned_runtime_removed"] = runtime is not None
    elif runtime is not None:
        plan = runtime.plan if prefix is None else dataclasses.replace(runtime.plan, prefix=prefix)
        transformer[stage_key] = dataclasses.replace(runtime, plan=plan, metrics=scratch_metrics)
    local["transformer_options"] = transformer
    return local, receipt


def _pdd_head_receipt(guider) -> dict[str, Any]:
    head = getattr(
        getattr(getattr(getattr(guider, "inner_model", None), "diffusion_model", None), "final_layer", None),
        "video_out",
        None,
    )
    weight = getattr(head, "weight", None)
    out_features = getattr(head, "out_features", None)
    if weight is None or not out_features:
        return {"status": "unavailable"}
    rows = int(weight.shape[0])
    return {
        "status": "measured",
        "video_out_weight_rows": rows,
        "video_out_out_features": int(out_features),
        "video_head_bank_count": rows // int(out_features),
    }


def _noise_ledger(x_suffix, pre_suffix, sigma, noise_scale):
    signal = (1.0 - sigma) * pre_suffix
    implied = (x_suffix - signal) / (sigma * noise_scale)
    centered = implied - implied.mean()
    pre_centered = pre_suffix - pre_suffix.mean()
    correlation = float((centered * pre_centered).sum() / (centered.norm() * pre_centered.norm()).clamp_min(1e-30))
    return implied, {
        "sigma": sigma,
        "clean_signal_weight": 1.0 - sigma,
        "noise_weight": sigma * noise_scale,
        "signal_to_noise_power": ((1.0 - sigma) / (sigma * noise_scale)) ** 2,
        "suffix_state_rms_per_token": _token_rms(x_suffix[:, :, :5]),
        "suffix_clean_signal_rms_per_token": _token_rms(signal[:, :, :5]),
        "implied_noise_mean": float(implied.mean()),
        "implied_noise_std": float(implied.std()),
        "implied_noise_std_per_token": implied[:, :, :5].float().std(dim=(0, 1, 3, 4)).tolist(),
        "implied_noise_correlation_with_pre_high_clean": correlation,
    }


def maybe_run_first_high_counterfactual(
    guider,
    x,
    timestep,
    model_options,
    seed,
    *,
    production,
    trace,
    call_index,
    sigma,
    actual,
    metrics,
    stage_key,
):
    """Evaluate the requested arms once, at the first actual target-high call."""

    request = (getattr(guider, "model_options", None) or {}).get(FIRST_HIGH_COUNTERFACTUAL_KEY)
    if not isinstance(request, FirstHighCounterfactualRequest):
        return
    evidence = getattr(trace, "window_evidence", None)
    if evidence is None or not getattr(evidence, "capture_full_video", False):
        if not getattr(trace, "first_high_counterfactual_reported", False):
            trace.first_high_counterfactual_reported = True
            metrics.event(
                "partitioned_first_high_counterfactual",
                policy=FIRST_HIGH_COUNTERFACTUAL_POLICY,
                status="unavailable",
                reason="requires capture_boundary_witness full-stage evidence on this continuation",
            )
        return
    plan = evidence.plan
    if not actual or evidence.first_high_call_index != call_index or "first_high_counterfactual" in plan:
        return
    started = time.perf_counter()
    shapes = getattr(getattr(guider, "inner_model", None), "latent_shapes", None)
    if not isinstance(shapes, list) or len(shapes) != 2:
        raise RuntimeError("first-high counterfactual could not resolve H3 AV latent shapes")
    prefix_t = int(plan["prefix_t"])
    start, stop = int(plan["window_start_t"]), int(plan["window_stop_t"])
    exact = evidence.tensors["authoritative_prefix_full"]
    provider = evidence.tensors["provider_native_clean_full"]
    pre_high = evidence.tensors["pre_high_exact_restored_full"]
    production_video = unpack_streams(production, shapes)[0]
    production_reference = production_video.detach().clone()
    x_video, x_audio = unpack_streams(x, shapes)
    device, dtype = x_video.device, x_video.dtype
    sigma = float(sigma)
    noise_scale = float(getattr(getattr(guider.inner_model, "model_sampling", None), "noise_scale", 1.0))
    scratch = H3FlowMetrics()
    transformer = (model_options or {}).get("transformer_options") or {}
    receipt: dict[str, Any] = {
        "policy": FIRST_HIGH_COUNTERFACTUAL_POLICY,
        "call_index": int(call_index),
        "sigma": sigma,
        "sample_sigmas": [float(v) for v in torch.as_tensor(transformer.get("sample_sigmas", [])).reshape(-1)],
        "prefix_t": prefix_t,
        "window_start_t": start,
        "window_stop_t": stop,
        "arm_prediction": "guider_predict_noise_without_predict_noise_wrappers",
        "production_prediction_returned_unchanged": True,
        "pdd_heads": _pdd_head_receipt(guider),
        "arms": [],
    }
    pre_suffix = pre_high[:, :, prefix_t:].to(device=device, dtype=dtype)
    _implied, receipt["high_entry_ledger"] = _noise_ledger(x_video[:, :, prefix_t:], pre_suffix, sigma, noise_scale)
    del _implied
    with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []), torch.no_grad():
        for arm in request.arms:
            arm_started = time.perf_counter()
            video = x_video.clone()
            prefix = None
            change: dict[str, Any] = {}
            if arm == "provider_prefix":
                exact_d = exact.to(device=device, dtype=dtype)
                prefix = provider[:, :, :prefix_t].to(device=device, dtype=dtype)
                video[:, :, :prefix_t] += VISUAL_COND_TIMESTEP * (prefix - exact_d)
                change["prefix_source"] = "provider_native_clean"
                change["provider_minus_exact_prefix_rms_per_token"] = _token_rms((prefix - exact_d)[:, :, -3:])
            elif arm == "suffix_noise_reseed":
                generator = torch.Generator(device="cpu").manual_seed(request.noise_seed)
                fresh = torch.randn(pre_suffix.shape, generator=generator, dtype=torch.float32)
                video[:, :, prefix_t:] = (1.0 - sigma) * pre_suffix + sigma * noise_scale * fresh.to(device, dtype)
                change["noise_seed"] = request.noise_seed
                changed = (video - x_video)[:, :, prefix_t : prefix_t + 5]
                change["suffix_state_change_rms_per_token"] = _token_rms(changed)
            arm_x = pack_streams((video, x_audio.clone()))[0]
            options, isolation = _arm_options(
                model_options,
                stage_key=stage_key,
                scratch_metrics=scratch,
                prefix=prefix,
                native_attention=arm == "native_high_attention",
            )
            row: dict[str, Any] = {"arm": arm, "input_change": change, "isolation": isolation}
            try:
                output = guider.predict_noise(arm_x, timestep, options, seed)
            except Exception as exc:
                row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                receipt["arms"].append(row)
                continue
            arm_video = unpack_streams(output, shapes)[0]
            name = arm_tensor_name(arm)
            window = arm_video[:, :, start:stop].detach().to(device="cpu", dtype=torch.float32, copy=True)
            evidence.tensors[name] = window.contiguous()
            evidence.cpu_tensor_bytes += window.numel() * window.element_size()
            difference = arm_video[:, :, start:stop].float() - production_reference[:, :, start:stop].float()
            row.update(
                status="ok",
                tensor=name,
                bitwise_equal_to_production=bool(torch.equal(arm_video, production_reference)),
                window_difference_from_production_rms_per_token=_token_rms(difference),
                elapsed_ms=(time.perf_counter() - arm_started) * 1000.0,
            )
            receipt["arms"].append(row)
            del output, arm_video, arm_x, video
    if not torch.equal(production_video, production_reference):
        raise RuntimeError("first-high counterfactual changed the production prediction")
    receipt["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    receipt["extra_h3_predictions"] = sum(row["status"] == "ok" for row in receipt["arms"])
    plan["first_high_counterfactual"] = receipt
    metrics.event("partitioned_first_high_counterfactual", status="captured", **receipt)


class H3FirstHighCounterfactual:
    @classmethod
    def INPUT_TYPES(cls):
        arm = {"default": True}
        return {
            "required": {
                "model": ("MODEL",),
                "replay": ("BOOLEAN", {**arm, "tooltip": "Repeat the first actual high prediction unchanged."}),
                "provider_prefix": (
                    "BOOLEAN",
                    {**arm, "tooltip": "Condition on the learned provider's own prefix instead of the exact prefix."},
                ),
                "suffix_noise_reseed": (
                    "BOOLEAN",
                    {**arm, "tooltip": "Re-noise the pre-high suffix with independent seeded noise."},
                ),
                "native_high_attention": (
                    "BOOLEAN",
                    {
                        **arm,
                        "tooltip": "Run the prediction through the installed native attention chain instead of "
                        "Flow's partitioned high-stage dispatch.",
                    },
                ),
                "noise_seed": ("INT", {"default": 1, "min": 0, "max": 2**63 - 1}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "MiniMax H3/diagnostics"
    DESCRIPTION = (
        "Diagnostic only. At the first actual target-high prediction of a continuation, repeat the prediction once "
        "per selected arm and save each result in the boundary evidence bundle. Requires capture_boundary_witness. "
        "Each arm adds one H3 evaluation per CFG branch; generated output is unchanged."
    )

    def patch(self, model, replay, provider_prefix, suffix_noise_reseed, native_high_attention, noise_seed):
        selected = dict(
            replay=replay,
            provider_prefix=provider_prefix,
            suffix_noise_reseed=suffix_noise_reseed,
            native_high_attention=native_high_attention,
        )
        request = FirstHighCounterfactualRequest(
            arms=tuple(arm for arm in FIRST_HIGH_COUNTERFACTUAL_ARMS if selected[arm]), noise_seed=int(noise_seed)
        )
        patched = model.clone()
        patched.model_options[FIRST_HIGH_COUNTERFACTUAL_KEY] = request
        return (patched,)


def audit_first_high_counterfactual(vae, bundle_path, join_frame, process_out):
    """Decode every saved arm window and measure adjacent-frame displacement at the join."""

    import os

    from .local_boundary_audit import (
        _decode_owned_pixels,
        geometry_comparison,
        normalize_bundle_path,
        read_verified_operand,
    )

    native = getattr(vae, "first_stage_model", None)
    expected = {"tokens_chunk_size": 5, "token_overlap": 2, "frame_pre_padding": 3, "clip_length": 17}
    if native is None or any(getattr(native, key, None) != value for key, value in expected.items()):
        raise ValueError("connect the native MiniMax H3 video VAE used for production decoding")
    if type(join_frame) is not int or join_frame < 0:
        raise ValueError("chunk_join_frame must be a non-negative integer")
    path = Path(normalize_bundle_path(bundle_path, platform=os.name, wsl_distro=os.environ.get("WSL_DISTRO_NAME")))
    directory = path.expanduser().resolve()
    if directory.name == "manifest.json":
        directory = directory.parent
    manifest = json.loads((directory / "manifest.json").read_bytes())
    window = manifest.get("metadata", {}).get("window", {})
    receipt = window.get("first_high_counterfactual")
    if not isinstance(receipt, dict) or receipt.get("policy") != FIRST_HIGH_COUNTERFACTUAL_POLICY:
        raise ValueError("this bundle has no first-high counterfactual capture")
    if (window.get("window_stop_t", 0) - window.get("window_start_t", 0)) != WINDOW_TOKENS:
        raise ValueError("first-high counterfactual windows differ from native boundary timing")
    entries = manifest["tensor_bytes"]
    hashes: dict[str, str] = {}

    def read(name):
        return read_verified_operand(directory, entries, name, hashes)

    exact_context = read("authoritative_prefix")
    provider_context = read("provider_native_clean")[:, :, :WINDOW_CONTEXT_TOKENS]
    latents = {"pre_high": read("pre_high_exact_restored"), "production_first_high": read("first_high_before_flow")}
    for row in receipt["arms"]:
        if row.get("status") == "ok":
            latents[row["arm"]] = read(row["tensor"])
            if row["arm"] == "provider_prefix":
                latents["provider_prefix_with_provider_context"] = latents["provider_prefix"]
    # Local frame 5 of a native seven-token window decode is the first frame of
    # the first generated token, i.e. the assembled join frame.
    labels = [join_frame - 5 + i for i in range(17)]
    measured = slice(2, 11)
    report = {
        "policy": "first_high_counterfactual_audit_v1",
        "join_frame": join_frame,
        "capture": receipt,
        "operand_sha256": hashes,
        "decoder": "standalone_native_seven_token_window",
        "context_tokens": "authoritative_prefix_except_provider_context_arm",
        "frame_labels": labels[measured][1:],
        "sign_convention": "sample candidate at (x-dx,y-dy) to compare with reference",
        "extra_vae_calls": 0,
        "stages": {},
    }
    pixels = {}
    for name, latent in latents.items():
        latent = latent.clone()
        latent[:, :, :WINDOW_CONTEXT_TOKENS] = (
            provider_context if name == "provider_prefix_with_provider_context" else exact_context
        )
        frames = _decode_owned_pixels(vae, latent, process_out, 22)[0, :17].detach().float().cpu()
        report["extra_vae_calls"] += 1
        frames = frames[measured]
        pixels[name] = frames
        upper = frames[:, : round(frames.shape[1] * 0.45)]
        report["stages"][name] = {
            "adjacent_frame_geometry": geometry_comparison(frames[:-1], frames[1:], labels[measured][1:]),
            "adjacent_frame_geometry_upper45": geometry_comparison(upper[:-1], upper[1:], labels[measured][1:]),
            "adjacent_rgb_difference_rms": (frames[1:] - frames[:-1]).square().mean((1, 2, 3)).sqrt().tolist(),
        }
    for name, frames in pixels.items():
        if name != "production_first_high":
            report["stages"][name]["same_frame_rgb_difference_from_production_rms"] = (
                (frames[1:] - pixels["production_first_high"][1:]).square().mean((1, 2, 3)).sqrt().tolist()
            )
    return report


class H3FirstHighCounterfactualAudit:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_vae": ("VAE",),
                "bundle_path": ("STRING", {"default": "", "tooltip": "Boundary bundle directory or manifest.json."}),
                "chunk_join_frame": ("INT", {"default": 0, "min": 0, "max": 10000000}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("numerical_report",)
    FUNCTION = "audit"
    CATEGORY = "MiniMax H3/diagnostics"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "Decode the first-high counterfactual arms saved in a boundary bundle with the production video VAE and "
        "report adjacent-frame displacement around the chunk join for each arm, pre-high and the production "
        "first prediction. Saves the numerical report as JSON in the output directory."
    )

    @classmethod
    def IS_CHANGED(cls, **_kwargs):
        return float("nan")

    def audit(self, video_vae, bundle_path, chunk_join_frame):
        import folder_paths
        from comfy.latent_formats import MiniMaxH3Video

        report = audit_first_high_counterfactual(video_vae, bundle_path, chunk_join_frame, MiniMaxH3Video().process_out)
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        output = Path(folder_paths.get_output_directory()) / f"first-high-counterfactual-{stamp}.json"
        output.write_text(text)
        return {"ui": {"text": [text]}, "result": (text,)}


NODE_CLASS_MAPPINGS = {
    "H3FirstHighCounterfactual": H3FirstHighCounterfactual,
    "H3FirstHighCounterfactualAudit": H3FirstHighCounterfactualAudit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FirstHighCounterfactual": "MiniMax H3 First-High Counterfactual [Diagnostic]",
    "H3FirstHighCounterfactualAudit": "MiniMax H3 First-High Counterfactual Audit",
}

__all__ = [
    "FIRST_HIGH_COUNTERFACTUAL_ARMS",
    "FIRST_HIGH_COUNTERFACTUAL_KEY",
    "FIRST_HIGH_COUNTERFACTUAL_POLICY",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "FirstHighCounterfactualRequest",
    "H3FirstHighCounterfactual",
    "H3FirstHighCounterfactualAudit",
    "arm_tensor_name",
    "audit_first_high_counterfactual",
    "maybe_run_first_high_counterfactual",
]
