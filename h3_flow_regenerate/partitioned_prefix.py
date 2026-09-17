"""Versioned partitioned exact-prefix attention contract and arithmetic oracle.

This module is intentionally independent from ComfyUI runtime ownership.  It
contains the immutable geometry/measure contract that Flow will publish once the
runtime path is enabled, plus a small dense reference used to prove that
partition-wise attention can be merged without changing softmax semantics.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Iterable, Sequence

import torch
import torch.nn.functional as F

PARTITIONED_PREFIX_KEY = "h3_flow_partitioned_exact_prefix_v1"
PARTITIONED_PREFIX_API = 1
PARTITIONED_PREFIX_TOPOLOGY = "target_prefix_source_suffix"


def _positive_int(value: int, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _digest(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class PartitionedExactPrefixPlan:
    """Immutable row-domain description for exact-prefix progressive attention.

    Packed row order is always ``[nonvideo | prefix_target | suffix_source]``.
    Prefix and suffix rows deliberately retain different physical spatial
    measures.  ``prefix_log_key_measure`` is the additive natural-log softmax
    bias required to make one target-grid prefix frame carry the same total
    spatial measure as one source-grid suffix frame.
    """

    video_start: int
    temporal: int
    prefix_t: int
    source_grid_h: int
    source_grid_w: int
    target_grid_h: int
    target_grid_w: int

    def __post_init__(self) -> None:
        _positive_int(self.video_start, "video_start")
        _positive_int(self.temporal, "temporal")
        _positive_int(self.prefix_t, "prefix_t")
        if self.prefix_t >= self.temporal:
            raise ValueError("partitioned exact-prefix plan requires a generated suffix")
        for name in ("source_grid_h", "source_grid_w", "target_grid_h", "target_grid_w"):
            _positive_int(getattr(self, name), name)
        if self.source_grid_h > self.target_grid_h or self.source_grid_w > self.target_grid_w:
            raise ValueError("source grid must not exceed target grid on either spatial axis")
        if self.source_rows >= self.target_rows:
            raise ValueError("partitioned exact-prefix plan requires a strictly smaller source grid")

    @property
    def source_rows(self) -> int:
        return self.source_grid_h * self.source_grid_w

    @property
    def target_rows(self) -> int:
        return self.target_grid_h * self.target_grid_w

    @property
    def prefix_rows(self) -> int:
        return self.prefix_t * self.target_rows

    @property
    def suffix_rows(self) -> int:
        return (self.temporal - self.prefix_t) * self.source_rows

    @property
    def sequence_rows(self) -> int:
        return self.video_start + self.prefix_rows + self.suffix_rows

    @property
    def prefix_range(self) -> tuple[int, int]:
        return self.video_start, self.video_start + self.prefix_rows

    @property
    def suffix_range(self) -> tuple[int, int]:
        return self.prefix_range[1], self.sequence_rows

    @property
    def prefix_log_key_measure(self) -> float:
        return math.log(self.source_rows / self.target_rows)

    @property
    def semantic_digest(self) -> str:
        return _digest(self.to_contract(include_digest=False))

    def to_contract(self, *, include_digest: bool = True) -> dict:
        payload = {
            "api": PARTITIONED_PREFIX_API,
            "topology": PARTITIONED_PREFIX_TOPOLOGY,
            "sequence_rows": self.sequence_rows,
            "video_start": self.video_start,
            "temporal": self.temporal,
            "prefix_t": self.prefix_t,
            "source_grid_h": self.source_grid_h,
            "source_grid_w": self.source_grid_w,
            "target_grid_h": self.target_grid_h,
            "target_grid_w": self.target_grid_w,
            "source_rows_per_frame": self.source_rows,
            "target_rows_per_frame": self.target_rows,
            "prefix_range": list(self.prefix_range),
            "suffix_range": list(self.suffix_range),
            "prefix_log_key_measure": self.prefix_log_key_measure,
            "nonvideo_log_key_measure": 0.0,
            "suffix_log_key_measure": 0.0,
            "exact_prefix_queries_preserved": True,
            "generated_suffix_queries_preserved": True,
            "heterogeneous_spatial_domains": True,
        }
        if include_digest:
            payload["semantic_digest"] = _digest(payload)
        return payload


def validate_partitioned_contract(contract: dict, *, sequence_rows: int | None = None) -> PartitionedExactPrefixPlan:
    if not isinstance(contract, dict):
        raise ValueError("partitioned exact-prefix contract must be a dictionary")
    if contract.get("api") != PARTITIONED_PREFIX_API or contract.get("topology") != PARTITIONED_PREFIX_TOPOLOGY:
        raise ValueError("unsupported partitioned exact-prefix contract")
    names = (
        "video_start",
        "temporal",
        "prefix_t",
        "source_grid_h",
        "source_grid_w",
        "target_grid_h",
        "target_grid_w",
    )
    if any(type(contract.get(name)) is not int for name in names):
        raise ValueError("partitioned exact-prefix geometry must use integer fields")
    plan = PartitionedExactPrefixPlan(**{name: contract[name] for name in names})
    canonical = plan.to_contract()
    for name in (
        "sequence_rows",
        "source_rows_per_frame",
        "target_rows_per_frame",
        "prefix_range",
        "suffix_range",
        "prefix_log_key_measure",
        "nonvideo_log_key_measure",
        "suffix_log_key_measure",
        "exact_prefix_queries_preserved",
        "generated_suffix_queries_preserved",
        "heterogeneous_spatial_domains",
        "semantic_digest",
    ):
        if contract.get(name) != canonical[name]:
            raise ValueError(f"partitioned exact-prefix contract field {name!r} is inconsistent")
    if sequence_rows is not None and plan.sequence_rows != int(sequence_rows):
        raise ValueError("partitioned exact-prefix contract does not match the current sequence")
    return plan


def merge_partition_attention(
    outputs: Sequence[torch.Tensor],
    lses: Sequence[torch.Tensor],
    log_measures: Sequence[float | torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge independently normalized attention partitions exactly.

    ``outputs[i]`` is the normalized attention result over key partition ``i``;
    ``lses[i]`` is that partition's natural-log normalizer for the same query
    rows/heads.  ``log_measures[i]`` is an additive natural-log physical key
    measure applied uniformly to that partition.
    """
    if not outputs or len(outputs) != len(lses) or len(outputs) != len(log_measures):
        raise ValueError("partition outputs/LSEs/measures must be non-empty and have equal length")
    reference_output = outputs[0]
    reference_lse = lses[0]
    if reference_lse.shape != reference_output.shape[:-1]:
        raise ValueError("partition LSE must match output without the value dimension")
    adjusted_lses = []
    for index, (output, lse, measure) in enumerate(zip(outputs, lses, log_measures)):
        if output.shape != reference_output.shape or output.dtype != reference_output.dtype or output.device != reference_output.device:
            raise ValueError(f"partition output {index} does not match the first output")
        if lse.shape != reference_lse.shape or lse.device != reference_lse.device:
            raise ValueError(f"partition LSE {index} does not match the first LSE")
        adjusted_lses.append(lse.to(torch.float32) + torch.as_tensor(measure, device=lse.device, dtype=torch.float32))
    stacked = torch.stack(adjusted_lses, dim=0)
    merged_lse = torch.logsumexp(stacked, dim=0)
    merged = torch.zeros_like(reference_output, dtype=torch.float32)
    for output, adjusted in zip(outputs, adjusted_lses):
        weight = torch.exp(adjusted - merged_lse).unsqueeze(-1)
        merged.add_(output.to(torch.float32) * weight)
    return merged.to(reference_output.dtype), merged_lse


def dense_partition_oracle(
    q: torch.Tensor,
    key_partitions: Sequence[torch.Tensor],
    value_partitions: Sequence[torch.Tensor],
    log_measures: Sequence[float],
    *,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return explicit dense and partition-merged attention for tiny test domains.

    Inputs use ``[B, H, T, D]``.  This is a correctness oracle only; production
    execution belongs to the Sol/VDN backend and must not call this helper.
    """
    if q.ndim != 4 or not key_partitions or len(key_partitions) != len(value_partitions):
        raise ValueError("dense partition oracle requires BHTD Q and matching non-empty K/V partitions")
    if len(key_partitions) != len(log_measures):
        raise ValueError("dense partition oracle requires one log measure per K/V partition")
    d = q.shape[-1]
    effective_scale = d ** -0.5 if scale is None else float(scale)
    outputs = []
    lses = []
    for k, v in zip(key_partitions, value_partitions):
        if k.shape != v.shape or k.ndim != 4 or k.shape[:2] != q.shape[:2] or k.shape[-1] != d:
            raise ValueError("dense partition oracle received incompatible K/V geometry")
        scores = torch.matmul(q.to(torch.float64), k.to(torch.float64).transpose(-1, -2)) * effective_scale
        lse = torch.logsumexp(scores, dim=-1)
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v.to(torch.float64))
        outputs.append(out)
        lses.append(lse)
    merged, merged_lse = merge_partition_attention(outputs, lses, log_measures)

    full_k = torch.cat(tuple(key_partitions), dim=-2).to(torch.float64)
    full_v = torch.cat(tuple(value_partitions), dim=-2).to(torch.float64)
    full_bias = torch.cat(
        tuple(
            torch.full((k.shape[-2],), float(log_measure), dtype=torch.float64, device=q.device)
            for k, log_measure in zip(key_partitions, log_measures)
        )
    )
    full_scores = torch.matmul(q.to(torch.float64), full_k.transpose(-1, -2)) * effective_scale
    full_scores = full_scores + full_bias.view(1, 1, 1, -1)
    full_lse = torch.logsumexp(full_scores, dim=-1)
    full_out = torch.matmul(torch.softmax(full_scores, dim=-1), full_v)
    return full_out, full_lse, merged.to(torch.float64), merged_lse.to(torch.float64)


__all__ = [
    "PARTITIONED_PREFIX_API",
    "PARTITIONED_PREFIX_KEY",
    "PARTITIONED_PREFIX_TOPOLOGY",
    "PartitionedExactPrefixPlan",
    "dense_partition_oracle",
    "merge_partition_attention",
    "validate_partitioned_contract",
]
