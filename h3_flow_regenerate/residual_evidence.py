from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

from .vae_boundary_video import (
    H3_VAE_CHUNK_TOKENS,
    H3_VAE_CLIP_FRAMES,
    H3_VAE_FRAME_OVERLAP,
    H3_VAE_TOKEN_OVERLAP,
    H3_VAE_WINDOW_TOKENS,
)


class BoundaryWindowEvidence:
    """CPU stage snapshots, with optional full video for target-band evidence."""

    def __init__(
        self,
        exact_prefix: torch.Tensor,
        temporal: int,
        *,
        capture_full_video: bool = False,
        max_bytes=256 * 1024 * 1024,
    ):
        prefix_t = int(exact_prefix.shape[2])
        start = prefix_t - H3_VAE_TOKEN_OVERLAP
        stop = start + H3_VAE_WINDOW_TOKENS
        if start < 0 or start % H3_VAE_CHUNK_TOKENS:
            raise ValueError("boundary-window evidence requires the native 5k+2 prefix phase")
        if stop > temporal:
            raise ValueError("boundary-window evidence lacks five real generated decoder-context tokens")
        self.plan = {
            "policy": "native_h3_boundary_decoder_window_v1",
            "prefix_t": prefix_t,
            "temporal": temporal,
            "window_start_t": start,
            "window_stop_t": stop,
            "window_tokens": H3_VAE_WINDOW_TOKENS,
            "token_overlap": H3_VAE_TOKEN_OVERLAP,
            "chunk_stride_tokens": H3_VAE_CHUNK_TOKENS,
            "decoder_chunk_output_start_frame": start // H3_VAE_CHUNK_TOKENS * H3_VAE_CLIP_FRAMES,
            "decoded_trim_frames": start // H3_VAE_CHUNK_TOKENS * H3_VAE_CLIP_FRAMES + H3_VAE_FRAME_OVERLAP,
            "first_retained_local_frame": H3_VAE_FRAME_OVERLAP,
        }
        self.tensors: dict[str, torch.Tensor] = {}
        self.capture_full_video = capture_full_video
        self.max_bytes = max_bytes
        self.cpu_tensor_bytes = 0
        self.first_high_call_index: int | None = None
        self.first_high_sigma: float | None = None
        self.copy_wall_s = 0.0
        self.pending_high_prediction = None
        if capture_full_video:
            self.plan["high_prediction_trace"] = {
                "policy": "bounded_high_prediction_windows_v1",
                "max_calls": 16,
                "calls": [],
                "omitted_call_indices": [],
            }
        self.capture("authoritative_prefix", exact_prefix)

    def __deepcopy__(self, memo):
        memo[id(self)] = self
        return self

    def capture(self, name: str, video: torch.Tensor) -> None:
        start, stop = self.plan["window_start_t"], self.plan["window_stop_t"]
        if name == "authoritative_prefix":
            stop = self.plan["prefix_t"]
        window = video[:, :, start:stop].detach()
        if window.shape[2] != stop - start:
            raise RuntimeError("boundary-window snapshot is missing native decoder context")
        snapshots = {name: window}
        if self.capture_full_video:
            expected_t = self.plan["prefix_t"] if name == "authoritative_prefix" else self.plan["temporal"]
            if video.shape[2] != expected_t:
                raise RuntimeError("full-video stage snapshot does not match the declared timeline")
            snapshots[name + "_full"] = video.detach()
        if any(key in self.tensors for key in snapshots):
            raise RuntimeError(f"boundary-window snapshot {name!r} was reused")
        size = sum(value.numel() * value.element_size() for value in snapshots.values())
        if self.capture_full_video and self.cpu_tensor_bytes + size > self.max_bytes:
            raise RuntimeError("full-video stage evidence exceeded its explicit CPU byte budget")
        started = time.perf_counter()
        for key, value in snapshots.items():
            self.tensors[key] = value.to(device="cpu", copy=True).contiguous()
        self.cpu_tensor_bytes += size
        self.copy_wall_s += time.perf_counter() - started

    def observe_prediction(self, video, *, point, call_index, sigma, actual, sampler_input=None):
        if point == "before_flow" and actual and self.first_high_call_index is None:
            self.first_high_call_index = int(call_index)
            self.first_high_sigma = float(sigma)
            self.capture("first_high_before_flow", video)
            if sampler_input is not None:
                self.capture("first_high_sampler_input", sampler_input)
        elif (
            point == "after_flow"
            and actual
            and call_index == self.first_high_call_index
            and "first_high_after_flow" not in self.tensors
        ):
            self.capture("first_high_after_flow", video)
        if self.capture_full_video and point in {"before_flow", "after_flow"}:
            self._observe_high_prediction(video, point=point, call_index=call_index, sigma=sigma, actual=actual)

    def _observe_high_prediction(self, video, *, point, call_index, sigma, actual):
        trace = self.plan["high_prediction_trace"]
        if self.first_high_call_index is None or not 0 <= call_index < trace["max_calls"]:
            return
        if point == "before_flow":
            if self.pending_high_prediction is not None or any(
                call["call_index"] == call_index for call in trace["calls"]
            ):
                raise RuntimeError("high prediction window capture reused a call")
            first = call_index == self.first_high_call_index
            names = (
                ("first_high_before_flow", "first_high_after_flow")
                if first
                else (f"high_prediction_{call_index:02d}_before_flow", f"high_prediction_{call_index:02d}_after_flow")
            )
            start, stop = self.plan["window_start_t"], self.plan["window_stop_t"]
            window = video[:, :, start:stop].detach()
            window_bytes = window.numel() * window.element_size()
            # Keep room for the existing final full-video snapshot. Optional
            # later-call evidence must never make a previously valid capture fail.
            reserve = video.numel() * video.element_size() + window_bytes
            needed = 0 if first else 2 * window_bytes
            if self.cpu_tensor_bytes + needed + reserve > self.max_bytes:
                trace["omitted_call_indices"].append(int(call_index))
                return
            if not first:
                self._capture_prediction_window(names[0], window)
            self.pending_high_prediction = {
                "call_index": int(call_index),
                "sigma": float(sigma),
                "actual": bool(actual),
                "before_flow": names[0],
                "after_flow": names[1],
            }
        elif self.pending_high_prediction is not None:
            call = self.pending_high_prediction
            if (int(call_index), float(sigma), bool(actual)) != (call["call_index"], call["sigma"], call["actual"]):
                raise RuntimeError("high prediction window pair changed provenance")
            if call["after_flow"] not in self.tensors:
                start, stop = self.plan["window_start_t"], self.plan["window_stop_t"]
                self._capture_prediction_window(call["after_flow"], video[:, :, start:stop].detach())
            trace["calls"].append(call)
            self.pending_high_prediction = None

    def _capture_prediction_window(self, name, window):
        if name in self.tensors or window.shape[2] != self.plan["window_tokens"]:
            raise RuntimeError("high prediction window capture geometry or ownership changed")
        started = time.perf_counter()
        self.tensors[name] = window.to(device="cpu", copy=True).contiguous()
        self.cpu_tensor_bytes += window.numel() * window.element_size()
        self.copy_wall_s += time.perf_counter() - started

    def close(self):
        self.tensors.clear()
        self.cpu_tensor_bytes = 0
        self.pending_high_prediction = None


def _safe_component(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value or "unknown")).strip("-")
    return text[:80] or "unknown"


def _write_exact_tensor(path: Path, tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().to(device="cpu").contiguous()
    raw = value.view(torch.uint8).numpy().tobytes()
    digest = hashlib.sha256(raw).hexdigest()
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(raw)
            temporary = Path(handle.name)
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {
        "file": path.name,
        "sha256": digest,
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "nbytes": len(raw),
        "byte_order": "native_torch_contiguous",
    }


def export_residual_geometry_evidence(
    tensors: dict[str, torch.Tensor],
    *,
    session_id: str,
    chunk_id: str,
    seed: int,
    sigma: float,
    metadata: dict[str, Any],
    evidence_kind: str = "h3_flow_exact_prefix_residual_geometry_evidence",
) -> dict[str, Any]:
    """Write bounded raw tensor evidence into ComfyUI's ordinary output tree.

    The measurement mode is explicit, so failure to locate a ComfyUI output
    directory is reported rather than silently redirecting evidence elsewhere.
    No tensor in this function is used by the production sampler.
    """

    started = time.perf_counter()
    try:
        import folder_paths
    except ImportError:
        return {
            "status": "unavailable",
            "reason": "comfyui_output_directory_unavailable",
            "extra_vae_calls": 0,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }

    output_dir = Path(folder_paths.get_output_directory())
    root = output_dir / "h3_flow_regenerate" / "residual_geometry"
    root.mkdir(parents=True, exist_ok=True)
    identity = (
        f"session-{_safe_component(session_id)}_chunk-{_safe_component(chunk_id)}"
        f"_seed-{int(seed)}_sigma-{float(sigma):.8f}_{time.time_ns()}"
    )
    bundle = root / identity
    bundle.mkdir(parents=False, exist_ok=False)

    tensor_manifest: dict[str, Any] = {}
    total_bytes = 0
    used_files: set[str] = set()
    try:
        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"residual evidence {name!r} is not a tensor")
            file_name = f"{_safe_component(name)}.bin"
            if file_name in used_files:
                raise ValueError(f"residual evidence {name!r} collides with file {file_name!r}")
            used_files.add(file_name)
            receipt = _write_exact_tensor(bundle / file_name, tensor)
            tensor_manifest[name] = receipt
            total_bytes += int(receipt["nbytes"])
        manifest = {
            "schema": 1,
            "kind": evidence_kind,
            "session_id": str(session_id),
            "chunk_id": str(chunk_id),
            "seed": int(seed),
            "sigma": float(sigma),
            "tensor_bytes": tensor_manifest,
            "metadata": metadata,
            "decoded_media": {
                "status": "external_workflow_evidence_required",
                "reason": (
                    "Flow owns latent-stage tensors but does not own the already-produced "
                    "raw-decode/retained/assembled media required by the design gate"
                ),
                "extra_vae_calls": 0,
            },
        }
        manifest_text = json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n"
        manifest_path = bundle / "manifest.json"
        manifest_path.write_text(manifest_text, encoding="utf-8")
        manifest_sha256 = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()
        relative = bundle.relative_to(output_dir).as_posix()
        return {
            "status": "exported",
            "bundle": relative,
            "manifest_sha256": manifest_sha256,
            "tensor_count": len(tensor_manifest),
            "tensor_bytes": total_bytes,
            "tensor_sha256": {name: receipt["sha256"] for name, receipt in tensor_manifest.items()},
            "decoded_media_status": "external_workflow_evidence_required",
            "extra_vae_calls": 0,
            "elapsed_ms": (time.perf_counter() - started) * 1000.0,
        }
    except BaseException:
        for path in bundle.glob("*"):
            path.unlink(missing_ok=True)
        bundle.rmdir()
        raise


__all__ = ["BoundaryWindowEvidence", "export_residual_geometry_evidence"]
