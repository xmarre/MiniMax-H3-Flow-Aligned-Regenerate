from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from typing import Literal

import torch

from .geometry import H3Geometry

SCHEMA_VERSION = 1
StoragePolicy = Literal["system_ram", "vram"]
SampleProvenance = Literal["actual", "forecast"]


@dataclass(frozen=True, slots=True)
class TrajectorySample:
    coordinate: float
    video_sigma: float
    audio_sigma: float
    outer_step: int
    call_index: int
    phase: str
    provenance: SampleProvenance
    video_x0: torch.Tensor
    timestamp_ns: int = field(default_factory=time.time_ns)

    @property
    def canonical(self) -> bool:
        return self.provenance == "actual" and self.phase in {"predicted", "corrected", "single"}


@dataclass(frozen=True, slots=True)
class TrajectoryRun:
    schema_version: int
    run_id: str
    session_id: str
    chunk_id: str
    sampler: str
    scheduler: str
    geometry: H3Geometry
    audio_shape: tuple[int, ...]
    layout_signature: str
    conditioning_signature: str
    storage: StoragePolicy
    samples: tuple[TrajectorySample, ...]
    started_ns: int
    completed_ns: int
    complete: bool
    abort_reason: str | None = None

    def exact_samples(self) -> tuple[TrajectorySample, ...]:
        return tuple(sample for sample in self.samples if sample.provenance == "actual")


@dataclass(slots=True)
class _PendingRun:
    run_id: str
    session_id: str
    chunk_id: str
    sampler: str
    scheduler: str
    geometry: H3Geometry
    audio_shape: tuple[int, ...]
    layout_signature: str
    conditioning_signature: str
    storage: StoragePolicy
    started_ns: int
    samples: list[TrajectorySample] = field(default_factory=list)


class H3FlowTrajectory:
    """Transactional trajectory handle passed through ComfyUI as H3_FLOW_TRAJECTORY."""

    api_version = SCHEMA_VERSION

    def __init__(self, *, storage: StoragePolicy = "system_ram", max_runs: int = 16):
        if storage not in {"system_ram", "vram"}:
            raise ValueError("trajectory storage must be system_ram or vram")
        if max_runs < 1:
            raise ValueError("max_runs must be positive")
        self.storage = storage
        self.max_runs = int(max_runs)
        self._lock = threading.RLock()
        self._pending: _PendingRun | None = None
        self._runs: tuple[TrajectoryRun, ...] = ()

    @property
    def runs(self) -> tuple[TrajectoryRun, ...]:
        with self._lock:
            return self._runs

    @property
    def latest(self) -> TrajectoryRun:
        with self._lock:
            if not self._runs:
                raise RuntimeError("trajectory has no runs")
            run = self._runs[-1]
            if not run.complete:
                raise RuntimeError("latest trajectory run is incomplete")
            return run

    @property
    def bytes(self) -> int:
        return sum(
            sample.video_x0.numel() * sample.video_x0.element_size() for run in self.runs for sample in run.samples
        )

    def begin(
        self,
        *,
        session_id: str,
        chunk_id: str,
        sampler: str,
        scheduler: str,
        geometry: H3Geometry,
        audio_shape: tuple[int, ...],
        layout_signature: str,
        conditioning_signature: str,
    ) -> str:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("trajectory already has an active transaction")
            run_id = uuid.uuid4().hex
            self._pending = _PendingRun(
                run_id=run_id,
                session_id=str(session_id),
                chunk_id=str(chunk_id),
                sampler=str(sampler),
                scheduler=str(scheduler),
                geometry=geometry,
                audio_shape=tuple(audio_shape),
                layout_signature=str(layout_signature),
                conditioning_signature=str(conditioning_signature),
                storage=self.storage,
                started_ns=time.time_ns(),
            )
            return run_id

    def append(self, run_id: str, sample: TrajectorySample) -> None:
        with self._lock:
            pending = self._require_pending(run_id)
            if sample.provenance not in {"actual", "forecast"}:
                raise ValueError("trajectory provenance must be actual or forecast")
            tensor = sample.video_x0.detach()
            if self.storage == "system_ram":
                tensor = tensor.to(device="cpu", dtype=torch.float32, copy=True)
                if torch.cuda.is_available():
                    tensor = tensor.pin_memory()
            else:
                tensor = tensor.clone()
            pending.samples.append(replace(sample, video_x0=tensor))

    def commit(self, run_id: str) -> TrajectoryRun:
        with self._lock:
            pending = self._require_pending(run_id)
            run = TrajectoryRun(
                schema_version=SCHEMA_VERSION,
                run_id=pending.run_id,
                session_id=pending.session_id,
                chunk_id=pending.chunk_id,
                sampler=pending.sampler,
                scheduler=pending.scheduler,
                geometry=pending.geometry,
                audio_shape=pending.audio_shape,
                layout_signature=pending.layout_signature,
                conditioning_signature=pending.conditioning_signature,
                storage=pending.storage,
                samples=tuple(pending.samples),
                started_ns=pending.started_ns,
                completed_ns=time.time_ns(),
                complete=True,
            )
            self._runs = (*self._runs, run)[-self.max_runs :]
            self._pending = None
            return run

    def abort(self, run_id: str, reason: str) -> TrajectoryRun:
        with self._lock:
            pending = self._require_pending(run_id)
            run = TrajectoryRun(
                schema_version=SCHEMA_VERSION,
                run_id=pending.run_id,
                session_id=pending.session_id,
                chunk_id=pending.chunk_id,
                sampler=pending.sampler,
                scheduler=pending.scheduler,
                geometry=pending.geometry,
                audio_shape=pending.audio_shape,
                layout_signature=pending.layout_signature,
                conditioning_signature=pending.conditioning_signature,
                storage=pending.storage,
                samples=tuple(pending.samples),
                started_ns=pending.started_ns,
                completed_ns=time.time_ns(),
                complete=False,
                abort_reason=str(reason),
            )
            self._runs = (*self._runs, run)[-self.max_runs :]
            self._pending = None
            return run

    def invalidate(self, run_id: str, reason: str) -> TrajectoryRun:
        """Make a previously committed run unavailable to future guidance selection."""
        with self._lock:
            for index in range(len(self._runs) - 1, -1, -1):
                run = self._runs[index]
                if run.run_id != str(run_id):
                    continue
                if not run.complete:
                    return run
                invalid = replace(
                    run,
                    complete=False,
                    abort_reason=str(reason),
                    completed_ns=time.time_ns(),
                )
                runs = list(self._runs)
                runs[index] = invalid
                self._runs = tuple(runs)
                return invalid
            raise RuntimeError("cannot invalidate an unknown trajectory run")

    def select(
        self,
        *,
        run_id: str | None = None,
        chunk_id: str | None = None,
        session_id: str | None = None,
        conditioning_signature: str | None = None,
    ) -> TrajectoryRun:
        with self._lock:
            candidates = [
                run
                for run in self._runs
                if (run_id is None or run.run_id == str(run_id))
                and (chunk_id is None or run.chunk_id == str(chunk_id))
                and (session_id is None or run.session_id == str(session_id))
                and (conditioning_signature is None or run.conditioning_signature == str(conditioning_signature))
            ]
            if not candidates:
                if run_id is not None:
                    raise RuntimeError(f"no trajectory matches requested run_id={run_id}")
                raise RuntimeError("no trajectory matches the requested session/chunk/conditioning")
            run = candidates[-1]
            if not run.complete:
                raise RuntimeError("latest matching trajectory is incomplete")
            if not any(sample.provenance == "actual" for sample in run.samples):
                raise RuntimeError("latest matching trajectory has no exact anchors")
            return run

    def clear(self) -> None:
        with self._lock:
            if self._pending is not None:
                raise RuntimeError("cannot clear a trajectory during an active transaction")
            self._runs = ()

    def _require_pending(self, run_id: str) -> _PendingRun:
        pending = self._pending
        if pending is None or pending.run_id != run_id:
            raise RuntimeError("stale or missing trajectory transaction")
        return pending
