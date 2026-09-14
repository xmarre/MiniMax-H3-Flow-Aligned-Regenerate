"""Controlled same-state cold high replay for the first-high progressive H3 artifact.

This diagnostic implements experiment R from FIRST_HIGH_STAGE_ARTIFACT_ARCHITECTURE.
A capture wrapper observes an otherwise unchanged controlled progressive run and
exports only tensors plus a versioned declarative manifest.  A fresh-process
replay wrapper then executes exactly the captured high suffix, with the original
Flow progressive wrapper suppressed for that one invocation.  It does not run
the low stage, the exact probe, or the learned upscaler.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import execution_contract_diagnostics as _diag
from . import execution_contract_provenance as _provenance
from . import runtime as _runtime
from .contracts import H3FlowTrajectory, TrajectorySample
from .geometry import H3Geometry, pack_streams

SCHEMA_VERSION = 1
CAPTURE_STATE_KEY = "h3_flow_same_state_replay_capture_v1"
REPLAY_STATE_KEY = "h3_flow_same_state_high_replay_v1"
_CAPTURE_WRAPPER_KEY = "h3_flow_regenerate.same_state_replay.capture.v1"
_REPLAY_WRAPPER_KEY = "h3_flow_regenerate.same_state_replay.execute.v1"
_MAX_BUNDLES = 2
_MAX_REPORTS = 4
_SAFE_PREFIX = re.compile(r"[^A-Za-z0-9._-]+")
_RUNTIME_ID_SUFFIX = re.compile(r"_(?:0x)?[0-9a-fA-F]{8,}$")
_MEMORY_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")

_REQUIRED_REPLAY_TENSORS = (
    "high_noise_argument",
    "high_latent_image",
    "first_high_sampler_input_video",
    "first_high_sampler_input_audio",
)


@dataclass(slots=True)
class _ReplayMaterial:
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor]


@dataclass(slots=True)
class _CaptureState:
    complete: deque[_ReplayMaterial] = field(default_factory=lambda: deque(maxlen=_MAX_BUNDLES))


@dataclass(slots=True)
class _ReplayState:
    manifest_path: str
    manifest: dict[str, Any]
    tensors: dict[str, torch.Tensor]
    complete: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_REPORTS))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tensor_sha256(value: torch.Tensor) -> str:
    work = value.detach().to(device="cpu").contiguous()
    return hashlib.sha256(work.view(torch.uint8).numpy().tobytes()).hexdigest()


def _cpu_clone(value: torch.Tensor) -> torch.Tensor:
    return value.detach().to(device="cpu", copy=True).contiguous()


def _snapshot_hashes(record: _diag._Record) -> dict[str, str]:
    return {name: _tensor_sha256(snapshot.tensor) for name, snapshot in sorted(record.snapshots.items())}


_DROP_PROVENANCE = object()


def _experiment_wrapper_entry(value: dict[str, Any]) -> bool:
    key = str(value.get("key", ""))
    return "same_state_replay" in key or key.startswith("h3_flow_regenerate.exec_contract.")


def _normalize_provenance(value: Any, *, path: tuple[str, ...] = ()) -> Any:
    """Build stable installed-runtime identity while excluding R/O-C instrumentation.

    Capture and replay intentionally use different outer wrappers, while the O/C
    recorder is observation-only. Their wrapper entries cannot be part of the
    source/backend identity tested by R. Runtime object addresses are likewise
    execution identity rather than source provenance.
    """
    if isinstance(value, dict):
        if _experiment_wrapper_entry(value):
            return _DROP_PROVENANCE
        out = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            lowered = key.lower()
            if key == "observer_wrapper_keys":
                continue
            if "same_state_replay" in lowered:
                continue
            if key.startswith("h3_flow_regenerate.exec_contract."):
                continue
            stable_key = _RUNTIME_ID_SUFFIX.sub("_<runtime-id>", key)
            normalized = _normalize_provenance(item, path=(*path, stable_key))
            if normalized is not _DROP_PROVENANCE:
                out[stable_key] = normalized
        return out
    if isinstance(value, (list, tuple)):
        normalized = []
        for item in value:
            child = _normalize_provenance(item, path=(*path, "[]"))
            if child is not _DROP_PROVENANCE:
                normalized.append(child)
        return normalized
    if isinstance(value, str):
        return _MEMORY_ADDRESS.sub("0x<addr>", value)
    return value


def _provenance_digest(manifest: dict[str, Any]) -> str:
    return _sha_json(_normalize_provenance(manifest))


def _guidance_config_dict(binding: _runtime.FlowBinding | None) -> dict[str, Any] | None:
    if binding is None or binding.guidance is None:
        return None
    if dataclasses.is_dataclass(binding.guidance):
        return dataclasses.asdict(binding.guidance)
    raise RuntimeError("same-state replay requires a declarative Flow guidance config")


def _guidance_run(binding: _runtime.FlowBinding | None):
    if binding is None or binding.trajectory is None:
        return None
    run_id = binding.captured_run_id
    if run_id:
        try:
            return binding.trajectory.select(run_id=run_id)
        except RuntimeError:
            pass
    try:
        return binding.trajectory.latest
    except RuntimeError:
        return None


def _geometry_dict(geometry: H3Geometry) -> dict[str, int]:
    return {field.name: int(getattr(geometry, field.name)) for field in dataclasses.fields(H3Geometry)}


def _stage_call(record: _diag._Record, stage: str) -> dict[str, Any] | None:
    for call in record.stage_calls:
        if call.get("stage") == stage:
            return call
    return None


def _runtime_policy_identity(runtime_snapshot: dict[str, Any] | None) -> dict[str, Any]:
    """Keep call policy/configuration while excluding cumulative backend counters."""
    source = runtime_snapshot or {}
    out: dict[str, Any] = {}
    refinement = source.get("h3_refinement")
    if refinement is not None:
        out["h3_refinement"] = refinement
    untwist = source.get("minimax_h3_untwist_rope")
    if isinstance(untwist, dict):
        keys = (
            "enabled",
            "beta",
            "start_percent",
            "end_percent",
            "high_scale_start",
            "high_scale_end",
            "low_scale_start",
            "low_scale_end",
            "scale_temporal_axis",
            "reference_scope",
            "progress",
            "progress_source",
        )
        out["minimax_h3_untwist_rope"] = {key: untwist.get(key) for key in keys if key in untwist}
    sol = source.get("sol_h3_runtime_v1")
    if isinstance(sol, dict):
        keys = (
            "api",
            "exact",
            "backend",
            "tau",
            "dense_evaluations",
            "dense_layers",
            "approximate",
            "attention_ownership",
            "sink_mode",
            "threshold",
            "kernel_contract",
            "sol_source",
            "sana_revision",
            "exact_kernel",
            "history_policy",
            "fingerprint",
        )
        out["sol_h3_runtime_v1"] = {key: sol.get(key) for key in keys if key in sol}
    for key in sorted(source):
        lowered = str(key).lower()
        if lowered.startswith("diffaid") or lowered.startswith("spectrum_h3_external_patch"):
            out[str(key)] = source[key]
    sample_sigmas = source.get("sample_sigmas")
    if isinstance(sample_sigmas, dict):
        out["sample_sigmas"] = {
            key: sample_sigmas.get(key)
            for key in ("shape", "dtype", "sample_digest")
            if key in sample_sigmas
        }
    return out


def _first_high_policy_snapshot(record: _diag._Record) -> dict[str, Any]:
    """Merge the persistent predict snapshot with the actual H3-call runtime snapshot."""
    merged: dict[str, Any] = {}
    if isinstance(record.first_high_runtime, dict):
        merged.update(record.first_high_runtime)
    h3_contract = record.first_high_h3_contract or {}
    h3_runtime = h3_contract.get("runtime") if isinstance(h3_contract, dict) else None
    if isinstance(h3_runtime, dict):
        merged.update(h3_runtime)
    return merged


def _material_from_record(record: _diag._Record, guider: Any) -> _ReplayMaterial:
    if record.error is not None:
        raise RuntimeError(f"cannot export a failed controlled run: {record.error}")
    high = _stage_call(record, "high")
    if high is None:
        raise RuntimeError("same-state replay capture did not observe a high-stage sampler lifetime")
    if str(high.get("sampler")) != "sample_euler":
        raise RuntimeError("same-state replay experiment R is bounded to the controlled Euler reproduction")
    if record.handoff_sigma is None or record.target_shapes is None:
        raise RuntimeError("same-state replay capture is missing target geometry or handoff sigma")
    if record.condition_compare is None or record.condition_compare.get("pre_core_equal") is not True:
        raise RuntimeError("same-state replay capture requires the validated pristine-target conditioning oracle")
    missing = [name for name in _REQUIRED_REPLAY_TENSORS if name not in record.snapshots]
    if missing:
        raise RuntimeError("same-state replay capture is missing required tensors: " + ", ".join(missing))

    binding = _runtime._resolve_binding(guider)
    run = _guidance_run(binding)
    if binding is not None and binding.guidance is not None and binding.guidance.mode != "off" and run is None:
        raise RuntimeError("same-state replay capture requires the committed low/probe guidance trajectory")

    tensors: dict[str, torch.Tensor] = {}
    for name in _REQUIRED_REPLAY_TENSORS:
        tensors[name] = _cpu_clone(record.snapshots[name].tensor)
    if "high_denoise_mask" in record.snapshots:
        tensors["high_denoise_mask"] = _cpu_clone(record.snapshots["high_denoise_mask"].tensor)
    high_sigmas = torch.tensor([float(value) for value in high["sigmas"]], dtype=torch.float32)
    tensors["high_sigmas"] = high_sigmas

    guidance_samples: list[dict[str, Any]] = []
    trajectory_meta = None
    if run is not None:
        trajectory_meta = {
            "schema_version": int(run.schema_version),
            "session_id": str(run.session_id),
            "chunk_id": str(run.chunk_id),
            "sampler": str(run.sampler),
            "scheduler": str(run.scheduler),
            "geometry": _geometry_dict(run.geometry),
            "audio_shape": [int(item) for item in run.audio_shape],
            "layout_signature": str(run.layout_signature),
            "conditioning_signature": str(run.conditioning_signature),
        }
        for index, sample in enumerate(run.samples):
            key = f"guidance_sample_{index:03d}_video_x0"
            tensors[key] = _cpu_clone(sample.video_x0)
            guidance_samples.append(
                {
                    "tensor_key": key,
                    "coordinate": float(sample.coordinate),
                    "video_sigma": float(sample.video_sigma),
                    "audio_sigma": float(sample.audio_sigma),
                    "outer_step": int(sample.outer_step),
                    "call_index": int(sample.call_index),
                    "phase": str(sample.phase),
                    "provenance": str(sample.provenance),
                }
            )

    metric_delta = {}
    if binding is not None:
        metric_delta = _diag._counter_delta(record.metric_start, binding.metrics.counters)
    topology = {
        "logical": int(metric_delta.get("sampler_logical_calls", 0)),
        "actual": int(metric_delta.get("transformer_actual_nfe", 0)),
        "forecast": int(metric_delta.get("spectrum_forecast_calls", 0)),
        "low": {
            "logical": int(metric_delta.get("sampler_logical_calls_low", 0)),
            "actual": int(metric_delta.get("transformer_actual_nfe_low", 0)),
            "forecast": int(metric_delta.get("spectrum_forecast_calls_low", 0)),
        },
        "probe": {
            "logical": int(metric_delta.get("sampler_logical_calls_probe", 0)),
            "actual": int(metric_delta.get("transformer_actual_nfe_probe", 0)),
            "forecast": int(metric_delta.get("spectrum_forecast_calls_probe", 0)),
        },
        "high": {
            "logical": int(metric_delta.get("sampler_logical_calls_high", 0)),
            "actual": int(metric_delta.get("transformer_actual_nfe_high", 0)),
            "forecast": int(metric_delta.get("spectrum_forecast_calls_high", 0)),
        },
    }
    if topology != {
        "logical": 9,
        "actual": 7,
        "forecast": 2,
        "low": {"logical": 5, "actual": 4, "forecast": 1},
        "probe": {"logical": 1, "actual": 1, "forecast": 0},
        "high": {"logical": 3, "actual": 2, "forecast": 1},
    }:
        raise RuntimeError(f"same-state replay capture topology diverged from controlled O: {topology}")

    tensor_hashes = {name: _tensor_sha256(value) for name, value in tensors.items()}
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": "minimax_h3_same_state_high_replay",
        "capture_id": record.capture_id,
        "created_ns": time.time_ns(),
        "seed": int(record.seed),
        "original_sigmas": [float(value) for value in record.original_sigmas],
        "original_schedule_digest": str(record.original_schedule_digest),
        "handoff_sigma": float(record.handoff_sigma),
        "handoff_index": int(record.original_sigmas.index(record.handoff_sigma))
        if record.handoff_sigma in record.original_sigmas
        else int(
            min(
                range(len(record.original_sigmas)),
                key=lambda i: abs(record.original_sigmas[i] - record.handoff_sigma),
            )
        ),
        "source_shapes": [[int(dim) for dim in shape] for shape in (record.source_shapes or [])],
        "target_shapes": [[int(dim) for dim in shape] for shape in record.target_shapes],
        "sampler": str(high["sampler"]),
        "high_sigmas": [float(value) for value in high["sigmas"]],
        "conditioning_digest": str(record.high_pre_core_digest),
        "pristine_conditioning_digest": str(record.pristine_cond_digest),
        "guidance_config": _guidance_config_dict(binding),
        "trajectory": trajectory_meta,
        "guidance_samples": guidance_samples,
        "expected_snapshot_hashes": _snapshot_hashes(record),
        "tensor_hashes": tensor_hashes,
        "first_high_runtime_policy": _runtime_policy_identity(_first_high_policy_snapshot(record)),
        "first_high_runtime_policy_digest": _sha_json(
            _runtime_policy_identity(_first_high_policy_snapshot(record))
        ),
        "provenance_digest": _provenance_digest(record.state.manifest),
        "provenance_identity": _normalize_provenance(record.state.manifest),
        "controlled_topology": topology,
        "contract": {
            "separate_diagnostic_job": True,
            "expected_replay_topology": {"logical": 3, "actual": 2, "forecast": 1},
            "expected_upscaler_calls": 0,
            "stochastic_state_transport": False,
            "weighted_mixed_grid_active": False,
            "sample_sigmas_owned_by_child_sampler": True,
        },
    }
    return _ReplayMaterial(manifest=manifest, tensors=tensors)


def _capture_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    guider = executor.class_obj
    state = (getattr(guider, "model_options", None) or {}).get(CAPTURE_STATE_KEY)
    record = _diag._ACTIVE.get()
    result = executor(
        noise,
        latent_image,
        sampler,
        sigmas,
        denoise_mask,
        callback,
        disable_pbar,
        seed,
        latent_shapes=latent_shapes,
    )
    if isinstance(state, _CaptureState) and record is not None:
        state.complete.append(_material_from_record(record, guider))
    return result


def _sanitize_prefix(value: str) -> str:
    text = _SAFE_PREFIX.sub("_", str(value).strip()).strip("._")
    return text[:80] or "h3_same_state_replay"


def _save_material(material: _ReplayMaterial, output_dir: Path, filename_prefix: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{_sanitize_prefix(filename_prefix)}_{material.manifest['capture_id']}"
    tensors_path = output_dir / f"{stem}.pt"
    manifest_path = output_dir / f"{stem}.json"
    tensors_tmp = output_dir / f".{stem}.pt.tmp"
    manifest_tmp = output_dir / f".{stem}.json.tmp"

    torch.save(material.tensors, tensors_tmp)
    manifest = dict(material.manifest)
    manifest["tensors_file"] = tensors_path.name
    manifest["tensors_file_sha256"] = hashlib.sha256(tensors_tmp.read_bytes()).hexdigest()
    manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tensors_tmp, tensors_path)
    os.replace(manifest_tmp, manifest_path)
    return manifest_path, tensors_path


def _resolve_manifest_path(value: str) -> Path:
    candidate = Path(str(value)).expanduser()
    if candidate.is_absolute():
        return candidate
    try:
        import folder_paths  # type: ignore

        return Path(folder_paths.get_output_directory()) / candidate
    except Exception:
        return candidate.resolve()


def _load_bundle(manifest_value: str) -> tuple[Path, dict[str, Any], dict[str, torch.Tensor]]:
    manifest_path = _resolve_manifest_path(manifest_value)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"same-state replay manifest not found: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION or manifest.get("kind") != "minimax_h3_same_state_high_replay":
        raise RuntimeError("unsupported same-state replay bundle schema")
    tensors_name = manifest.get("tensors_file")
    if not isinstance(tensors_name, str) or Path(tensors_name).name != tensors_name:
        raise RuntimeError("same-state replay manifest has an invalid tensor filename")
    tensors_path = manifest_path.with_name(tensors_name)
    if not tensors_path.is_file():
        raise FileNotFoundError(f"same-state replay tensor file not found: {tensors_path}")
    expected_file_hash = manifest.get("tensors_file_sha256")
    actual_file_hash = hashlib.sha256(tensors_path.read_bytes()).hexdigest()
    if not isinstance(expected_file_hash, str) or actual_file_hash != expected_file_hash:
        raise RuntimeError("same-state replay tensor file hash mismatch")
    try:
        tensors = torch.load(tensors_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "same-state replay requires a PyTorch build with safe weights_only tensor loading"
        ) from exc
    if not isinstance(tensors, dict) or not all(
        isinstance(key, str) and torch.is_tensor(value) for key, value in tensors.items()
    ):
        raise RuntimeError("same-state replay tensor payload is not a pure tensor dictionary")
    expected_hashes = manifest.get("tensor_hashes") or {}
    for key, expected in expected_hashes.items():
        if key not in tensors or _tensor_sha256(tensors[key]) != expected:
            raise RuntimeError(f"same-state replay tensor digest mismatch for {key}")
    return manifest_path, manifest, tensors


def _rebuild_trajectory(
    manifest: dict[str, Any],
    tensors: dict[str, torch.Tensor],
) -> tuple[H3FlowTrajectory | None, str | None]:
    trajectory_meta = manifest.get("trajectory")
    samples = manifest.get("guidance_samples") or []
    if trajectory_meta is None:
        if samples:
            raise RuntimeError("same-state replay bundle has guidance samples without trajectory metadata")
        return None, None
    geometry = H3Geometry(**{key: int(value) for key, value in trajectory_meta["geometry"].items()})
    trajectory = H3FlowTrajectory(storage="system_ram", max_runs=1)
    run_id = trajectory.begin(
        session_id=str(trajectory_meta["session_id"]),
        chunk_id=str(trajectory_meta["chunk_id"]),
        sampler=str(trajectory_meta["sampler"]),
        scheduler=str(trajectory_meta["scheduler"]),
        geometry=geometry,
        audio_shape=tuple(int(value) for value in trajectory_meta["audio_shape"]),
        layout_signature=str(trajectory_meta["layout_signature"]),
        conditioning_signature=str(trajectory_meta["conditioning_signature"]),
    )
    for item in samples:
        tensor_key = str(item["tensor_key"])
        if tensor_key not in tensors:
            raise RuntimeError(f"same-state replay bundle is missing trajectory tensor {tensor_key}")
        trajectory.append(
            run_id,
            TrajectorySample(
                coordinate=float(item["coordinate"]),
                video_sigma=float(item["video_sigma"]),
                audio_sigma=float(item["audio_sigma"]),
                outer_step=int(item["outer_step"]),
                call_index=int(item["call_index"]),
                phase=str(item["phase"]),
                provenance=str(item["provenance"]),
                video_x0=tensors[tensor_key],
            ),
        )
    trajectory.commit(run_id)
    return trajectory, run_id


def _dict_equal(left: Any, right: Any) -> bool:
    return _canonical_json(left) == _canonical_json(right)


def _live_snapshot_hash(record: _diag._Record, name: str) -> str | None:
    snapshot = record.snapshots.get(name)
    return None if snapshot is None else _tensor_sha256(snapshot.tensor)


def _replay_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    guider = executor.class_obj
    options = getattr(guider, "model_options", None)
    state = (options or {}).get(REPLAY_STATE_KEY)
    if not isinstance(state, _ReplayState):
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    record = _diag._ACTIVE.get()
    if record is None:
        raise RuntimeError("same-state replay must run inside MiniMax H3 Execution Contract Diagnostics")
    if not isinstance(options, dict):
        raise RuntimeError("same-state replay requires mutable guider model options")
    config = options.get(_runtime.PROGRESSIVE_KEY)
    if not _diag._eligible(config):
        raise RuntimeError("same-state replay requires the original controlled progressive Target Input config")
    if str(state.manifest.get("sampler")) != _runtime.sampler_name(sampler):
        raise RuntimeError("same-state replay sampler differs from the captured controlled run")
    if int(seed or 0) != int(state.manifest["seed"]):
        raise RuntimeError("same-state replay seed differs from the captured controlled run")
    if not isinstance(latent_shapes, list):
        raise RuntimeError("same-state replay requires mutable target latent shape metadata")
    target_shapes = [tuple(int(dim) for dim in shape) for shape in state.manifest["target_shapes"]]
    current_shapes = [tuple(int(dim) for dim in shape) for shape in latent_shapes]
    if current_shapes != target_shapes:
        raise RuntimeError(
            f"same-state replay target geometry mismatch: current={current_shapes} captured={target_shapes}"
        )
    if _runtime._schedule_signature(sigmas) != str(state.manifest["original_schedule_digest"]):
        raise RuntimeError("same-state replay caller schedule differs from the captured original schedule")
    if record.pristine_cond_digest != str(state.manifest["pristine_conditioning_digest"]):
        raise RuntimeError("same-state replay pristine target conditioning differs from capture")
    if _provenance_digest(record.state.manifest) != str(state.manifest["provenance_digest"]):
        raise RuntimeError("same-state replay installed runtime provenance differs from capture")

    binding = _runtime._resolve_binding(guider)
    if binding is None:
        raise RuntimeError("same-state replay requires the active Flow binding")
    current_guidance = _guidance_config_dict(binding)
    if not _dict_equal(current_guidance, state.manifest.get("guidance_config")):
        raise RuntimeError("same-state replay Flow guidance config differs from capture")

    trajectory, run_id = _rebuild_trajectory(state.manifest, state.tensors)
    if current_guidance is not None and current_guidance.get("mode") != "off" and trajectory is None:
        raise RuntimeError("same-state replay guidance is active but bundle has no captured trajectory")

    high_sigmas = state.tensors["high_sigmas"].to(dtype=sigmas.dtype, device=sigmas.device)
    captured_high = torch.tensor(
        [float(value) for value in state.manifest["high_sigmas"]],
        dtype=high_sigmas.dtype,
        device=high_sigmas.device,
    )
    if high_sigmas.shape != captured_high.shape or not torch.equal(high_sigmas, captured_high):
        raise RuntimeError("same-state replay high suffix differs from captured high suffix")

    replay_latent = state.tensors["high_latent_image"].to(device=latent_image.device, dtype=latent_image.dtype)
    replay_mask = state.tensors.get("high_denoise_mask")
    if replay_mask is not None:
        replay_mask = replay_mask.to(device=denoise_mask.device if denoise_mask is not None else latent_image.device)

    # R owns the exact high-entry state X. Reconstruct the sampler noise through
    # Flow/Comfy's reviewed inverse rather than trusting a serialized noise
    # argument. This is the architecture's same-X, same-Linternal replay gate.
    state_video = state.tensors["first_high_sampler_input_video"].to(
        device=latent_image.device, dtype=latent_image.dtype
    )
    state_audio = state.tensors["first_high_sampler_input_audio"].to(
        device=latent_image.device, dtype=latent_image.dtype
    )
    replay_state, replay_shapes = pack_streams((state_video, state_audio))
    if [tuple(int(dim) for dim in shape) for shape in replay_shapes] != target_shapes:
        raise RuntimeError("same-state replay packed X geometry differs from captured target geometry")
    base_model = guider.model_patcher.model
    replay_latent_internal = _runtime._process_latent_in(base_model, replay_latent, target_shapes)
    replay_noise = _runtime._noise_argument(
        base_model,
        replay_state,
        float(state.manifest["handoff_sigma"]),
        replay_latent_internal,
    )
    captured_noise = state.tensors["high_noise_argument"].to(device=replay_noise.device, dtype=replay_noise.dtype)
    if not torch.equal(replay_noise, captured_noise):
        raise RuntimeError(
            "same-state replay exact initialization inverse did not reconstruct the captured high noise argument"
        )
    expected_mask = replay_mask is not None
    if bool(denoise_mask is not None) != expected_mask:
        raise RuntimeError("same-state replay caller mask presence differs from captured high stage")

    handoff_index = int(state.manifest["handoff_index"])
    original_steps = len(state.manifest["original_sigmas"]) - 1

    def replay_callback(step, x0, x, _total):
        if callback is not None:
            return callback(handoff_index + step, x0, x, original_steps)
        return None

    metric_start = binding.metrics.counters
    event_start = len(binding.metrics.events)
    previous_progressive = options.pop(_runtime.PROGRESSIVE_KEY)
    previous_trajectory = binding.trajectory
    previous_guidance_run_id = binding.guidance_run_id
    previous_capture_enabled = binding.capture_enabled
    binding.trajectory = trajectory
    binding.guidance_run_id = run_id
    binding.capture_enabled = False
    result = None
    replay_error: BaseException | None = None
    try:
        if record.pristine_conds is None:
            raise RuntimeError("same-state replay O/C record is missing pristine target conditions")
        _runtime._reset_guider_conds(guider, template=record.pristine_conds)
        with _runtime._flow_stage_contract(guider, "high"), _runtime._high_stage_contract(guider):
            result = executor(
                replay_noise,
                replay_latent,
                sampler,
                high_sigmas,
                replay_mask,
                replay_callback,
                disable_pbar,
                seed,
                latent_shapes=latent_shapes,
            )
        return result
    except BaseException as exc:
        replay_error = exc
        raise
    finally:
        binding.trajectory = previous_trajectory
        binding.guidance_run_id = previous_guidance_run_id
        binding.capture_enabled = previous_capture_enabled
        options[_runtime.PROGRESSIVE_KEY] = previous_progressive

        metric_delta = _diag._counter_delta(metric_start, binding.metrics.counters)
        events = binding.metrics.events[event_start:]
        upscaler_calls = sum(1 for event in events if event.kind == "handoff_learned_upscale_wall")
        expected_hashes = state.manifest.get("expected_snapshot_hashes") or {}
        equality = {}
        for name in (
            "first_high_sampler_input_video",
            "first_high_sampler_input_audio",
            "first_high_h3_input_video",
            "first_high_h3_input_audio",
            "first_high_h3_velocity_video",
            "first_high_h3_velocity_audio",
            "first_high_model_raw_video",
            "first_high_model_raw_audio",
            "first_high_pre_guidance_video",
            "last_high_model_raw_video",
            "last_high_pre_guidance_video",
        ):
            current_hash = _live_snapshot_hash(record, name)
            expected_hash = expected_hashes.get(name)
            equality[name] = {
                "captured_sha256": expected_hash,
                "replay_sha256": current_hash,
                "equal": current_hash is not None and expected_hash is not None and current_hash == expected_hash,
            }
        runtime_policy = _runtime_policy_identity(_first_high_policy_snapshot(record))
        topology = {
            "logical": int(metric_delta.get("sampler_logical_calls", 0)),
            "actual": int(metric_delta.get("transformer_actual_nfe", 0)),
            "forecast": int(metric_delta.get("spectrum_forecast_calls", 0)),
            "high_logical": int(metric_delta.get("sampler_logical_calls_high", 0)),
            "high_actual": int(metric_delta.get("transformer_actual_nfe_high", 0)),
            "high_forecast": int(metric_delta.get("spectrum_forecast_calls_high", 0)),
            "upscaler_calls": int(upscaler_calls),
        }
        expected_topology = {
            "logical": 3,
            "actual": 2,
            "forecast": 1,
            "high_logical": 3,
            "high_actual": 2,
            "high_forecast": 1,
            "upscaler_calls": 0,
        }
        state.complete.append(
            {
                "schema_version": SCHEMA_VERSION,
                "kind": "minimax_h3_same_state_high_replay_report",
                "manifest_path": state.manifest_path,
                "capture_id": state.manifest["capture_id"],
                "error": None if replay_error is None else f"{type(replay_error).__name__}: {replay_error}",
                "cold_process_required": True,
                "topology": topology,
                "topology_exact": topology == expected_topology,
                "condition_pre_core_equal": record.high_pre_core_digest
                == state.manifest.get("conditioning_digest"),
                "provenance_exact": _provenance_digest(record.state.manifest)
                == state.manifest.get("provenance_digest"),
                "guidance_config_exact": _dict_equal(current_guidance, state.manifest.get("guidance_config")),
                "runtime_policy": runtime_policy,
                "runtime_policy_digest": _sha_json(runtime_policy),
                "runtime_policy_exact": _sha_json(runtime_policy)
                == state.manifest.get("first_high_runtime_policy_digest"),
                "snapshot_equality": equality,
                "first_high_raw_equal": bool(
                    equality["first_high_model_raw_video"]["equal"]
                    and equality["first_high_model_raw_audio"]["equal"]
                ),
                "first_high_h3_velocity_equal": bool(
                    equality["first_high_h3_velocity_video"]["equal"]
                    and equality["first_high_h3_velocity_audio"]["equal"]
                ),
                "attribution_valid": bool(
                    replay_error is None
                    and topology == expected_topology
                    and record.high_pre_core_digest == state.manifest.get("conditioning_digest")
                    and _provenance_digest(record.state.manifest) == state.manifest.get("provenance_digest")
                    and _dict_equal(current_guidance, state.manifest.get("guidance_config"))
                    and _sha_json(runtime_policy) == state.manifest.get("first_high_runtime_policy_digest")
                    and equality["first_high_sampler_input_video"]["equal"]
                    and equality["first_high_sampler_input_audio"]["equal"]
                    and equality["first_high_h3_input_video"]["equal"]
                    and equality["first_high_h3_input_audio"]["equal"]
                ),
                "decision": (
                    "same-output: retained lifecycle cause weakened"
                    if equality["first_high_model_raw_video"]["equal"]
                    else "different-output: retained lifecycle state implicated if attribution_valid"
                ),
            }
        )


def patch_same_state_capture(model: Any) -> tuple[Any, _CaptureState]:
    state = (getattr(model, "model_options", None) or {}).get(_diag.DIAGNOSTIC_KEY)
    if not isinstance(state, _diag._State):
        raise RuntimeError("same-state replay capture must be applied after MiniMax H3 Execution Contract Diagnostics")
    patched = model.clone()
    _comfy_compat._copy_model_options(patched)
    if CAPTURE_STATE_KEY in patched.model_options or REPLAY_STATE_KEY in patched.model_options:
        raise RuntimeError("same-state replay diagnostic is already installed")
    capture = _CaptureState()
    patched.model_options[CAPTURE_STATE_KEY] = capture

    import comfy.patcher_extension

    _diag._insert_relative(
        patched,
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _runtime.OUTER_WRAPPER_KEY,
        _CAPTURE_WRAPPER_KEY,
        _capture_wrapper,
        before=True,
    )
    return patched, capture


def patch_same_state_replay(model: Any, manifest_path: str) -> tuple[Any, _ReplayState]:
    diagnostic = (getattr(model, "model_options", None) or {}).get(_diag.DIAGNOSTIC_KEY)
    if not isinstance(diagnostic, _diag._State):
        raise RuntimeError("same-state high replay must be applied after MiniMax H3 Execution Contract Diagnostics")
    resolved, manifest, tensors = _load_bundle(manifest_path)
    patched = model.clone()
    _comfy_compat._copy_model_options(patched)
    transformer = patched.model_options.setdefault("transformer_options", {})
    if "h3_flow_untwist_clock_trial_v1" in transformer or "h3_flow_sampling_context" in transformer:
        raise RuntimeError("same-state replay R must not carry the rejected Untwist clock trial U")
    if CAPTURE_STATE_KEY in patched.model_options or REPLAY_STATE_KEY in patched.model_options:
        raise RuntimeError("same-state replay diagnostic is already installed")
    state = _ReplayState(manifest_path=str(resolved), manifest=manifest, tensors=tensors)
    patched.model_options[REPLAY_STATE_KEY] = state

    import comfy.patcher_extension

    _diag._insert_relative(
        patched,
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _runtime.OUTER_WRAPPER_KEY,
        _REPLAY_WRAPPER_KEY,
        _replay_wrapper,
        before=True,
    )
    return patched, state


class H3SameStateReplayCapture:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Experiment R capture. Apply after MiniMax H3 Execution Contract Diagnostics for one unchanged controlled "
        "progressive run. It records exact high-entry tensors plus the low/probe guidance trajectory; it changes no "
        "sampler/model state and adds no H3 or upscaler call."
    )
    RETURN_TYPES = ("MODEL", "H3_FLOW_HIGH_REPLAY_CAPTURE")
    RETURN_NAMES = ("model", "capture")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL",)}}

    def apply(self, model):
        return patch_same_state_capture(model)


class H3SameStateReplayBundleSave:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Save the latest experiment-R capture as a JSON manifest plus pure-tensor .pt payload. Connect the sampled "
        "LATENT as trigger so the capture is complete before writing."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bundle_manifest",)
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "capture": ("H3_FLOW_HIGH_REPLAY_CAPTURE",),
                "trigger": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "h3_same_state_replay"}),
            }
        }

    def save(self, capture, trigger, filename_prefix):
        _ = trigger
        if not isinstance(capture, _CaptureState):
            raise TypeError("invalid H3 same-state replay capture handle")
        if not capture.complete:
            raise RuntimeError("no completed same-state replay capture is available")
        material = capture.complete.pop()
        try:
            import folder_paths  # type: ignore

            output_dir = Path(folder_paths.get_output_directory()) / "h3_flow_replay"
        except Exception:
            output_dir = Path.cwd() / "output" / "h3_flow_replay"
        manifest_path, _ = _save_material(material, output_dir, str(filename_prefix))
        return (str(manifest_path),)


class H3SameStateHighReplay:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Experiment R cold high-only replay. Apply after MiniMax H3 Execution Contract Diagnostics in a fresh Comfy "
        "process. It executes only the captured high suffix with the captured entry state and guidance anchors; low, "
        "probe and learned upscaler are not executed."
    )
    RETURN_TYPES = ("MODEL", "H3_FLOW_HIGH_REPLAY")
    RETURN_NAMES = ("model", "replay")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "bundle_manifest": ("STRING", {"multiline": False}),
            }
        }

    def apply(self, model, bundle_manifest):
        return patch_same_state_replay(model, str(bundle_manifest))


class H3SameStateHighReplayReport:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Emit experiment R attribution checks after the high-only replay. Connect the replayed LATENT as trigger."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"replay": ("H3_FLOW_HIGH_REPLAY",), "trigger": ("LATENT",)}}

    def extract(self, replay, trigger):
        _ = trigger
        if not isinstance(replay, _ReplayState):
            raise TypeError("invalid H3 same-state replay handle")
        if not replay.complete:
            raise RuntimeError("no completed same-state replay report is available")
        return (json.dumps(replay.complete.pop(), indent=2, sort_keys=True, default=str),)


NODE_CLASS_MAPPINGS = {
    "H3SameStateReplayCapture": H3SameStateReplayCapture,
    "H3SameStateReplayBundleSave": H3SameStateReplayBundleSave,
    "H3SameStateHighReplay": H3SameStateHighReplay,
    "H3SameStateHighReplayReport": H3SameStateHighReplayReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3SameStateReplayCapture": "MiniMax H3 Same-State Replay Capture",
    "H3SameStateReplayBundleSave": "Save MiniMax H3 Same-State Replay Bundle",
    "H3SameStateHighReplay": "MiniMax H3 Same-State Cold High Replay",
    "H3SameStateHighReplayReport": "MiniMax H3 Same-State Replay Report",
}
