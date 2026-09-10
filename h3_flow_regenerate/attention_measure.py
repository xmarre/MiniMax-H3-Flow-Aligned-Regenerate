from __future__ import annotations

import hashlib
import json
import math
from math import gcd
from typing import Any, Mapping, Sequence

import torch

ATTENTION_MEASURE_KEY = "attention_measure_v1"
ATTENTION_MEASURE_API = 1
ATTENTION_MEASURE_OPERATOR = "key_log_measure"
ATTENTION_MEASURE_NORMALIZATION = "h3_native_source_carrier_v1"
ATTENTION_MEASURE_TOPOLOGY = "mixed_grid_low_suffix"
ATTENTION_MEASURE_COORDINATE_POLICY = "minimax_h3_native_frame_grid_v1"


def _require_int(name: str, value: Any, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _require_grid(name: str, value: Any) -> tuple[int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise TypeError(f"{name} must be a two-element integer grid")
    return (
        _require_int(f"{name}[0]", value[0], minimum=1),
        _require_int(f"{name}[1]", value[1], minimum=1),
    )


def _normalize_ratio(num: Any, den: Any, *, name: str) -> tuple[int, int]:
    num = _require_int(f"{name}.mass_num", num, minimum=1)
    den = _require_int(f"{name}.mass_den", den, minimum=1)
    common = gcd(num, den)
    num //= common
    den //= common
    log_mass = math.log(num) - math.log(den)
    if not math.isfinite(log_mass):
        raise ValueError(f"{name} derives a non-finite key-log-measure")
    return num, den


def _canonical_segments(raw_segments: Sequence[Mapping[str, Any]], *, kv_rows: int) -> list[dict[str, int]]:
    if not raw_segments:
        raise TypeError("attention_measure_v1 segments must be a nonempty list")
    segments: list[dict[str, int]] = []
    cursor = 0
    for index, raw in enumerate(raw_segments):
        if not isinstance(raw, Mapping):
            raise TypeError(f"segments[{index}] must be a mapping")
        start = _require_int(f"segments[{index}].start", raw.get("start"), minimum=0)
        stop = _require_int(f"segments[{index}].stop", raw.get("stop"), minimum=0)
        if start != cursor or stop <= start or stop > kv_rows:
            raise ValueError("attention_measure_v1 segments must be sorted, contiguous, nonempty, and in range")
        mass_num, mass_den = _normalize_ratio(
            raw.get("mass_num"), raw.get("mass_den"), name=f"segments[{index}]"
        )
        if segments and (segments[-1]["mass_num"], segments[-1]["mass_den"]) == (mass_num, mass_den):
            segments[-1]["stop"] = stop
        else:
            segments.append(
                {"start": start, "stop": stop, "mass_num": mass_num, "mass_den": mass_den}
            )
        cursor = stop
    if cursor != kv_rows:
        raise ValueError("attention_measure_v1 segments must cover every key row exactly once")
    return segments


def build_attention_measure_request(
    *,
    q_rows: int,
    kv_rows: int,
    video_start: int,
    temporal: int,
    prefix_t: int,
    source_grid: Sequence[int],
    prefix_grid: Sequence[int],
) -> dict[str, Any]:
    """Build the schema-v1 Mixed-Grid key-measure request.

    The request describes measure only. It never changes Q/K/V topology: all
    rows remain present and the protected target-grid prefix receives a
    per-key mass of ``source_rows / prefix_rows`` so one protected frame has
    the same total spatial softmax measure as one native low-grid suffix frame.
    """

    source_h, source_w = _require_grid("source_grid", source_grid)
    prefix_h, prefix_w = _require_grid("prefix_grid", prefix_grid)
    source_rows = source_h * source_w
    prefix_rows = prefix_h * prefix_w
    prefix_stop = video_start + prefix_t * prefix_rows
    raw_segments = []
    if video_start:
        raw_segments.append({"start": 0, "stop": video_start, "mass_num": 1, "mass_den": 1})
    raw_segments.append(
        {
            "start": video_start,
            "stop": prefix_stop,
            "mass_num": source_rows,
            "mass_den": prefix_rows,
        }
    )
    if prefix_stop < kv_rows:
        raw_segments.append({"start": prefix_stop, "stop": kv_rows, "mass_num": 1, "mass_den": 1})
    return validate_attention_measure_request(
        {
            "api": ATTENTION_MEASURE_API,
            "operator": ATTENTION_MEASURE_OPERATOR,
            "normalization": ATTENTION_MEASURE_NORMALIZATION,
            "topology": ATTENTION_MEASURE_TOPOLOGY,
            "q_rows": q_rows,
            "kv_rows": kv_rows,
            "video_start": video_start,
            "temporal": temporal,
            "prefix_t": prefix_t,
            "source_grid": [source_h, source_w],
            "prefix_grid": [prefix_h, prefix_w],
            "segments": raw_segments,
            "coordinate_policy": ATTENTION_MEASURE_COORDINATE_POLICY,
        }
    )


def validate_attention_measure_request(request: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and canonicalize ``attention_measure_v1`` without mutating it."""

    if not isinstance(request, Mapping):
        raise TypeError("attention_measure_v1 must be a mapping")
    expected_literals = {
        "api": ATTENTION_MEASURE_API,
        "operator": ATTENTION_MEASURE_OPERATOR,
        "normalization": ATTENTION_MEASURE_NORMALIZATION,
        "topology": ATTENTION_MEASURE_TOPOLOGY,
        "coordinate_policy": ATTENTION_MEASURE_COORDINATE_POLICY,
    }
    for key, expected in expected_literals.items():
        if request.get(key) != expected:
            raise ValueError(f"attention_measure_v1 {key} must be {expected!r}")

    q_rows = _require_int("q_rows", request.get("q_rows"), minimum=1)
    kv_rows = _require_int("kv_rows", request.get("kv_rows"), minimum=1)
    if q_rows != kv_rows:
        raise ValueError("Mixed-Grid attention measure retains all self-attention rows, so q_rows must equal kv_rows")
    video_start = _require_int("video_start", request.get("video_start"), minimum=0)
    if video_start >= kv_rows:
        raise ValueError("video_start must identify a nonempty video suffix inside the sequence")
    temporal = _require_int("temporal", request.get("temporal"), minimum=2)
    prefix_t = _require_int("prefix_t", request.get("prefix_t"), minimum=1)
    if prefix_t >= temporal:
        raise ValueError("prefix_t must leave a nonempty generated suffix")
    source_grid = _require_grid("source_grid", request.get("source_grid"))
    prefix_grid = _require_grid("prefix_grid", request.get("prefix_grid"))
    source_rows = math.prod(source_grid)
    prefix_rows = math.prod(prefix_grid)
    if source_rows > prefix_rows:
        raise ValueError("Mixed-Grid source spatial measure cannot exceed the protected-prefix grid")
    expected_rows = video_start + prefix_t * prefix_rows + (temporal - prefix_t) * source_rows
    if q_rows != expected_rows:
        raise ValueError(
            "attention_measure_v1 row count does not match video_start + protected-prefix rows + low-grid suffix rows"
        )

    raw_segments = request.get("segments")
    if not isinstance(raw_segments, (list, tuple)):
        raise TypeError("attention_measure_v1 segments must be a nonempty list")
    segments = _canonical_segments(raw_segments, kv_rows=kv_rows)

    weighted_start = video_start
    weighted_stop = video_start + prefix_t * prefix_rows
    expected_ratio = _normalize_ratio(source_rows, prefix_rows, name="expected_prefix")
    for segment in segments:
        overlap = max(segment["start"], weighted_start) < min(segment["stop"], weighted_stop)
        ratio = (segment["mass_num"], segment["mass_den"])
        if overlap and ratio != expected_ratio:
            raise ValueError("every protected-prefix key row must use source_rows/prefix_rows spatial measure")
        if not overlap and ratio != (1, 1):
            raise ValueError("non-prefix conditioning and generated-suffix key rows must retain unit measure")
        if segment["start"] < weighted_start < segment["stop"] or segment["start"] < weighted_stop < segment["stop"]:
            if expected_ratio != (1, 1):
                raise ValueError("measure segment boundaries must align with the protected-prefix key interval")

    return {
        "api": ATTENTION_MEASURE_API,
        "operator": ATTENTION_MEASURE_OPERATOR,
        "normalization": ATTENTION_MEASURE_NORMALIZATION,
        "topology": ATTENTION_MEASURE_TOPOLOGY,
        "q_rows": q_rows,
        "kv_rows": kv_rows,
        "video_start": video_start,
        "temporal": temporal,
        "prefix_t": prefix_t,
        "source_grid": list(source_grid),
        "prefix_grid": list(prefix_grid),
        "segments": segments,
        "coordinate_policy": ATTENTION_MEASURE_COORDINATE_POLICY,
    }


def attention_measure_semantic_digest(request: Mapping[str, Any]) -> str:
    canonical = validate_attention_measure_request(request)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def materialize_key_log_measure(
    request: Mapping[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Materialize the O(T) natural-log key measure described by the request."""

    canonical = validate_attention_measure_request(request)
    if not dtype.is_floating_point:
        raise TypeError("key log measure requires a floating-point dtype")
    bias = torch.empty(canonical["kv_rows"], device=device, dtype=dtype)
    for segment in canonical["segments"]:
        value = math.log(segment["mass_num"]) - math.log(segment["mass_den"])
        bias[segment["start"] : segment["stop"]] = value
    return bias


def nonunit_exact_key_ranges(request: Mapping[str, Any], *, block_size: int = 64) -> tuple[tuple[int, int], ...]:
    """Return token ranges for every physical key block touching non-unit measure."""

    canonical = validate_attention_measure_request(request)
    block_size = _require_int("block_size", block_size, minimum=1)
    blocks: set[int] = set()
    for segment in canonical["segments"]:
        if (segment["mass_num"], segment["mass_den"]) == (1, 1):
            continue
        first = segment["start"] // block_size
        last = (segment["stop"] - 1) // block_size
        blocks.update(range(first, last + 1))
    if not blocks:
        return ()
    ranges: list[tuple[int, int]] = []
    run_start = run_last = min(blocks)
    for block in sorted(blocks)[1:]:
        if block == run_last + 1:
            run_last = block
            continue
        ranges.append((run_start * block_size, min((run_last + 1) * block_size, canonical["kv_rows"])))
        run_start = run_last = block
    ranges.append((run_start * block_size, min((run_last + 1) * block_size, canonical["kv_rows"])))
    return tuple(ranges)


def _apply_mask(scores: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
    if mask is None:
        return scores
    if mask.device != scores.device:
        raise ValueError("attention mask and scores must be on the same device")
    if mask.dtype == torch.bool:
        return scores.masked_fill(~mask, float("-inf"))
    if not mask.dtype.is_floating_point:
        raise TypeError("attention mask must be boolean or floating point")
    if torch.isnan(mask).any() or torch.isposinf(mask).any():
        raise ValueError("additive attention mask cannot contain NaN or +inf")
    return scores + mask.to(dtype=scores.dtype)


def dense_weighted_attention_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    key_log_measure: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
    key_chunk_size: int | None = None,
) -> torch.Tensor:
    """FP64 dense/streaming oracle for key-log-measure attention.

    Q/K/V use ``(..., Tq|Tk, D)``. ``key_log_measure`` is a natural-log bias
    over Tk and is added *after* QK score scaling. ``mask`` follows SDPA-style
    boolean semantics (True = allowed) or additive logit-mask semantics.
    Supplying ``key_chunk_size`` exercises the same online-softmax algebra in
    bounded key chunks without materializing the complete score matrix.
    """

    if q.ndim < 2 or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("q, k, and v must have matching rank >= 2")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v batch/head dimensions must match")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("incompatible q/k/v attention dimensions")
    if key_log_measure.device != q.device or key_log_measure.ndim != 1 or key_log_measure.shape[0] != k.shape[-2]:
        raise ValueError("key_log_measure must be a device-matched vector with one entry per K/V row")
    if not key_log_measure.dtype.is_floating_point or not torch.isfinite(key_log_measure).all():
        raise ValueError("key_log_measure must be finite floating point")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    scale = float(scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("attention scale must be finite and positive")

    q64, k64, v64 = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
    bias64 = key_log_measure.to(torch.float64)
    tk = k.shape[-2]
    if key_chunk_size is None:
        scores = torch.matmul(q64, k64.transpose(-1, -2)) * scale
        scores = scores + bias64
        scores = _apply_mask(scores, mask)
        row_max = scores.amax(dim=-1, keepdim=True)
        safe_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        p = torch.exp(scores - safe_max)
        p = torch.where(torch.isfinite(scores), p, torch.zeros_like(p))
        den = p.sum(dim=-1, keepdim=True)
        return torch.matmul(p, v64) / den.clamp_min(torch.finfo(torch.float64).tiny)

    key_chunk_size = _require_int("key_chunk_size", key_chunk_size, minimum=1)
    out_shape = (*q64.shape[:-1], v64.shape[-1])
    row_max = torch.full((*q64.shape[:-1], 1), float("-inf"), dtype=torch.float64, device=q.device)
    row_sum = torch.zeros_like(row_max)
    numerator = torch.zeros(out_shape, dtype=torch.float64, device=q.device)
    for start in range(0, tk, key_chunk_size):
        stop = min(start + key_chunk_size, tk)
        scores = torch.matmul(q64, k64[..., start:stop, :].transpose(-1, -2)) * scale
        scores = scores + bias64[start:stop]
        chunk_mask = None if mask is None else mask[..., start:stop]
        scores = _apply_mask(scores, chunk_mask)
        chunk_max = scores.amax(dim=-1, keepdim=True)
        new_max = torch.maximum(row_max, chunk_max)
        safe_old = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        safe_new = torch.where(torch.isfinite(new_max), new_max, torch.zeros_like(new_max))
        old_scale = torch.where(torch.isfinite(row_max), torch.exp(safe_old - safe_new), torch.zeros_like(row_sum))
        p = torch.exp(scores - safe_new)
        p = torch.where(torch.isfinite(scores), p, torch.zeros_like(p))
        numerator = numerator * old_scale + torch.matmul(p, v64[..., start:stop, :])
        row_sum = row_sum * old_scale + p.sum(dim=-1, keepdim=True)
        row_max = new_max
    return numerator / row_sum.clamp_min(torch.finfo(torch.float64).tiny)
