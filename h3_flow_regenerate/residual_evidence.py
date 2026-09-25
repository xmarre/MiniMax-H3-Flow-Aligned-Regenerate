from __future__ import annotations

import hashlib
import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

import torch


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
    try:
        for name, tensor in tensors.items():
            if not torch.is_tensor(tensor):
                raise TypeError(f"residual evidence {name!r} is not a tensor")
            receipt = _write_exact_tensor(bundle / f"{_safe_component(name)}.bin", tensor)
            tensor_manifest[name] = receipt
            total_bytes += int(receipt["nbytes"])
        manifest = {
            "schema": 1,
            "kind": "h3_flow_exact_prefix_residual_geometry_evidence",
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


__all__ = ["export_residual_geometry_evidence"]
