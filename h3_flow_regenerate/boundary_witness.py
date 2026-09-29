"""Default-off, stage-owned CPU evidence from an existing VDN evaluation."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path

import torch

WITNESS_KEY = "h3_flow_partitioned_boundary_witness_v1"
WITNESS_API = 1
WITNESS_ENV = "H3_FLOW_BOUNDARY_WITNESS_DIR"


class BoundaryWitness:
    api = WITNESS_API

    def __init__(self, directory, metrics, *, max_bytes=32 * 1024 * 1024):
        self.directory = Path(directory).expanduser().resolve()
        self.metrics = metrics
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._claimed = False
        self._tensors = {}
        self._bytes = 0
        self._started = None
        self.completed = False
        self.copy_host_wall_s = 0.0

    def claim(self, context):
        # Preflight never supplies actual raw features. Probe and high are excluded.
        if context.get("stage") != "low":
            return False
        with self._lock:
            if self._claimed:
                return False
            self._claimed = True
            self._started = time.perf_counter()
            return True

    def add(self, name, tensor):
        started = time.perf_counter()
        if name in self._tensors:
            raise RuntimeError("boundary witness tensor name was reused")
        size = tensor.numel() * tensor.element_size()
        if self._bytes + size > self.max_bytes:
            raise RuntimeError("boundary witness exceeded its explicit CPU byte budget")
        # Copy before retained scratch is reused. Never retain a CUDA view.
        self._tensors[name] = tensor.detach().to(device="cpu", copy=True).contiguous()
        self._bytes += size
        self.copy_host_wall_s += time.perf_counter() - started

    def finish(self, context):
        if not self._tensors:
            raise RuntimeError("requested boundary witness did not observe a mixed short-conv window")
        write_started = time.perf_counter()
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = "boundary-witness-" + uuid.uuid4().hex
        target = self.directory / (stem + ".pt")
        temporary = target.with_suffix(".tmp")
        try:
            torch.save(self._tensors, temporary)
            temporary.replace(target)
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            modules = {}
            for name, module in tuple(sys.modules.items()):
                if not any(
                    part in name
                    for part in ("h3_flow", "vdn_h3", "sol_h3", "minimax", "continuum", "spectrum", "latent_upscal")
                ) and name not in ("comfy.samplers", "comfy.model_base", "comfy.model_sampling"):
                    continue
                source = getattr(module, "__file__", None)
                if source and Path(source).is_file():
                    modules[name] = {
                        "path": str(Path(source).resolve()),
                        "sha256": hashlib.sha256(Path(source).read_bytes()).hexdigest(),
                    }
            receipt = {
                "api": self.api,
                "context": context,
                "tensor_file": target.name,
                "tensor_sha256": digest,
                "cpu_tensor_bytes": self._bytes,
                "cpu_byte_budget": self.max_bytes,
                "host_wall_s_including_copies_and_io": time.perf_counter() - self._started,
                "tensor_copy_host_wall_s": self.copy_host_wall_s,
                "artifact_write_and_source_hash_host_wall_s": time.perf_counter() - write_started,
                "timing_is_diagnostic_not_production_overhead": True,
                "loaded_modules": modules,
                "torch": torch.__version__,
                "precision_at_actual_call": {
                    "matmul": torch.get_float32_matmul_precision(),
                    "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
                    "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
                },
                "extra_h3_nfe": 0,
                "extra_provider_calls": 0,
                "extra_vae_calls": 0,
                "output_mutated": False,
                "diagnostic_only": True,
                "checkpoint_provenance_complete": False,
            }
            target.with_suffix(".json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
            self.metrics.event("partitioned_boundary_witness", path=str(target), **receipt)
            self.completed = True
        finally:
            temporary.unlink(missing_ok=True)
            self._tensors.clear()


def configured_boundary_witness(metrics):
    directory = os.environ.get(WITNESS_ENV, "").strip()
    return BoundaryWitness(directory, metrics) if directory else None


def witness_requested():
    return bool(os.environ.get(WITNESS_ENV, "").strip())
