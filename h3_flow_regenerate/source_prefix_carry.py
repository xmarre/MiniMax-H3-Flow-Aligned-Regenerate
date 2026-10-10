"""Experimental, hash-paired native source context for sequential H3 chunks."""

from __future__ import annotations

import hashlib
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import wraps

import torch

from .decode_context import _frames, video_latent_fingerprint

SOURCE_PREFIX_CARRY_KEY = "h3_flow_source_prefix_carry_v1"
SOURCE_PREFIX_CARRY_POLICY = "native_source_carry_v1"


def _owned_video(video):
    if (
        not torch.is_tensor(video)
        or video.ndim != 5
        or tuple(video.shape[:2]) != (1, 24)
        or not video.is_floating_point()
        or any(n % 2 for n in video.shape[-2:])
        or not bool(torch.isfinite(video).all())
    ):
        raise ValueError("native source carry requires finite [1,24,T,H,W] video on an even grid")
    _frames(int(video.shape[2]))
    return video.detach().to(device="cpu", dtype=torch.float32, copy=True).contiguous()


@dataclass(frozen=True)
class _Snapshot:
    source: torch.Tensor
    target_hw: tuple[int, int]
    target_suffix_hashes: dict[int, str]
    generation: int
    sequence: tuple[str, int] | None


class SourcePrefixCarry:
    """One CPU source clip, paired with suffix hashes of its returned target.

    A complete outer sampling transaction publishes the next pair. Nested
    Flow wrappers of that same guider share the transaction; another guider
    cannot enter it. No generated state is retained on CUDA.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._guider = None
        self._snapshot = None
        self._source = None
        self._pending = None
        self._generation = 0

    @contextmanager
    def transaction(self, guider):
        thread = threading.get_ident()
        if self._thread == thread and self._guider is guider:
            yield
            return
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("native source carry is already sampling another chunk")
        self._thread, self._guider = thread, guider
        self._source = self._pending = None
        try:
            yield
            if self._pending is not None:
                self._snapshot = self._pending
                self._generation = self._pending.generation
        finally:
            self._source = self._pending = None
            self._thread = self._guider = None
            self._lock.release()

    def _require_transaction(self):
        if self._thread != threading.get_ident():
            raise RuntimeError("native source carry requires an active sampling transaction")

    def _sequence(self):
        options = getattr(self._guider, "model_options", None) or {}
        request = (options.get("transformer_options") or {}).get("h3_continuum")
        if not isinstance(request, dict) or request.get("active") is not True:
            return None
        try:
            chunk = int(request["chunk_index"])
        except (KeyError, ValueError, TypeError):
            raise ValueError("native source carry requires an integer Continuum chunk index") from None
        return str(request.get("session_id", "continuum")), chunk

    def begin_initial(self):
        self._require_transaction()
        self._snapshot = None

    def stage_source(self, video):
        self._require_transaction()
        if self._source is not None:
            raise RuntimeError("native source carry already staged a clean prediction for this chunk")
        self._source = _owned_video(video)

    def prepare_success(self, target):
        self._require_transaction()
        started = time.perf_counter()
        source = self._source
        if source is None:
            raise RuntimeError("native source carry lost this chunk's low/probe clean prediction")
        target = _owned_video(target)
        if source.shape[2] != target.shape[2] or any(
            a >= b for a, b in zip(source.shape[-2:], target.shape[-2:], strict=True)
        ):
            raise ValueError("native source carry requires paired temporal lengths and a larger target grid")
        hashes = {
            n: hashlib.sha256(target[:, :, -n:].contiguous().numpy().tobytes()).hexdigest()
            for n in range(2, target.shape[2] + 1, 5)
        }
        self._pending = _Snapshot(source, tuple(target.shape[-2:]), hashes, self._generation + 1, self._sequence())
        return {
            "policy": SOURCE_PREFIX_CARRY_POLICY,
            "generation": self._pending.generation,
            "source_shape": list(source.shape),
            "target_shape": list(target.shape),
            "retained_cpu_bytes": source.numel() * source.element_size(),
            "retained_cuda_bytes": 0,
            "source_role": "low_probe_clean_prediction_at_handoff_sigma",
            "sequence": self._pending.sequence,
            "publication": "after_successful_outer_sampling_transaction",
            "elapsed_ms": (time.perf_counter() - started) * 1000,
            "extra_h3_nfe": 0,
            "extra_vae_calls": 0,
        }

    def project(self, prefix, source_h, source_w):
        self._require_transaction()
        started = time.perf_counter()
        exact = _owned_video(prefix)
        snapshot = self._snapshot
        if snapshot is None:
            raise RuntimeError("native_source_carry requires a preceding successful chunk from this patched MODEL")
        n = int(exact.shape[2])
        fingerprint = video_latent_fingerprint(exact)
        sequence = self._sequence()
        if (sequence is not None or snapshot.sequence is not None) and (
            sequence is None
            or snapshot.sequence is None
            or sequence[0] != snapshot.sequence[0]
            or sequence[1] != snapshot.sequence[1] + 1
        ):
            raise RuntimeError("native_source_carry requires the immediately preceding chunk in the same sequence")
        if (
            tuple(exact.shape[-2:]) != snapshot.target_hw
            or tuple(snapshot.source.shape[-2:]) != (source_h, source_w)
            or snapshot.target_suffix_hashes.get(n) != fingerprint["sha256_float32"]
        ):
            raise RuntimeError(
                "native_source_carry prefix does not match the preceding returned target suffix, grid or temporal phase"
            )
        projected = snapshot.source[:, :, -n:].clone()
        return projected, {
            "policy": SOURCE_PREFIX_CARRY_POLICY,
            "mode": "native_source_carry",
            "generation": snapshot.generation,
            "previous_sequence": snapshot.sequence,
            "current_sequence": sequence,
            "prefix_t": n,
            "authoritative_prefix": fingerprint,
            "authoritative_prefix_preserved_bitwise": True,
            "projected_prefix": video_latent_fingerprint(projected),
            "previous_source_token_span": [int(snapshot.source.shape[2]) - n, int(snapshot.source.shape[2])],
            "native_temporal_phase_verified": True,
            "source_role": "previous_chunk_low_probe_clean_prediction_at_handoff_sigma",
            "scope": "protected_prefix_only_before_low_sampling",
            "extra_h3_nfe": 0,
            "extra_vae_decode_calls": 0,
            "extra_vae_encode_calls": 0,
            "elapsed_ms": (time.perf_counter() - started) * 1000,
        }


def source_carry_owner(guider):
    value = (getattr(guider, "model_options", None) or {}).get(SOURCE_PREFIX_CARRY_KEY)
    if value is not None and not isinstance(value, SourcePrefixCarry):
        raise TypeError("invalid native source carry owner")
    return value


def source_carry_scope(*, outer=False):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            guider = (
                (args[0] if args else kwargs["executor"]).class_obj
                if outer
                else (args[1] if len(args) > 1 else kwargs["guider"])
            )
            owner = source_carry_owner(guider)
            if owner is None:
                return function(*args, **kwargs)
            with owner.transaction(guider):
                return function(*args, **kwargs)

        return wrapped

    return decorate
