"""Bottleneck report for ``h3-lifecycle-trace-v1`` JSON files.

Durations are reported both *inclusive* (a span and everything nested in it)
and *exclusive* (time not covered by a nested span on the same thread). Only
exclusive times are summed into subsystem totals. These form a partition only
when same-thread spans are nested or disjoint: overlapping sibling spans may
be counted more than once and must be inspected in the raw trace.
GPU activity comes from two independent sources: the CUDA
stream queue state recorded at span boundaries and, when available, NVML
utilization samples.
"""

from __future__ import annotations

import itertools
from collections import defaultdict
from typing import Any

from .lifecycle_trace import STAGE_WALL_EVENTS, TRACE_SCHEMA

_NS = 1e-9
_MIB = float(1 << 20)
IDLE_UTIL_PCT = 10
MIN_IDLE_INTERVAL_S = 1.0


def _duration(span: dict[str, Any]) -> int:
    return max(0, int(span["end_ns"]) - int(span["start_ns"]))


def _seconds(ns: float) -> float:
    return round(ns * _NS, 3)


def _index(spans: list[dict[str, Any]]) -> tuple[dict[int, dict], dict[int, list[dict]]]:
    by_id = {span["id"]: span for span in spans}
    children: dict[int, list[dict]] = defaultdict(list)
    for span in spans:
        parent = span.get("parent")
        if parent in by_id and by_id[parent]["thread"] == span["thread"]:
            children[parent].append(span)
    return by_id, children


def _exclusive(span: dict[str, Any], children: dict[int, list[dict]]) -> int:
    covered = sum(_duration(child) for child in children.get(span["id"], ()) if not child.get("instant"))
    return max(0, _duration(span) - covered)


def _descendants(span: dict[str, Any], children: dict[int, list[dict]]) -> list[dict[str, Any]]:
    stack = list(children.get(span["id"], ()))
    found = []
    while stack:
        item = stack.pop()
        found.append(item)
        stack.extend(children.get(item["id"], ()))
    return found


def _ancestor(span: dict[str, Any], by_id: dict[int, dict], name: str) -> dict[str, Any] | None:
    parent = by_id.get(span.get("parent"))
    while parent is not None:
        if parent["name"] == name:
            return parent
        parent = by_id.get(parent.get("parent"))
    return None


def _nvml_window(samples: list[list[int]], start: int, end: int) -> dict[str, Any] | None:
    window = [sample for sample in samples if start <= sample[0] <= end and sample[1] >= 0]
    if not window:
        return None
    util = [sample[1] for sample in window]
    power = [sample[3] for sample in window if sample[3] >= 0]
    return {
        "samples": len(window),
        "mean_gpu_util_pct": round(sum(util) / len(util), 1),
        "idle_fraction": round(sum(1 for value in util if value < IDLE_UTIL_PCT) / len(util), 3),
        "mean_power_w": round(sum(power) / len(power) / 1000.0, 1) if power else None,
    }


def _idle_intervals(samples: list[list[int]], start: int, end: int) -> list[tuple[int, int]]:
    intervals = []
    run_start = None
    last = None
    for sample in samples:
        t = sample[0]
        if t < start or t > end or sample[1] < 0:
            continue
        if sample[1] < IDLE_UTIL_PCT:
            if run_start is None:
                run_start = t
            last = t
        elif run_start is not None:
            intervals.append((run_start, t))
            run_start = None
    if run_start is not None and last is not None:
        intervals.append((run_start, last))
    return [(a, b) for a, b in intervals if (b - a) * _NS >= MIN_IDLE_INTERVAL_S]


def _covering(spans: list[dict[str, Any]], start: int, end: int, thread: int) -> list[dict[str, Any]]:
    """Spans on ``thread`` overlapping [start, end], with overlap seconds, deepest first."""
    overlaps = []
    for span in spans:
        if span["thread"] != thread or span.get("instant"):
            continue
        overlap = min(end, span["end_ns"]) - max(start, span["start_ns"])
        if overlap > 0 and span["name"] not in {"comfy.prompt"}:
            overlaps.append((span, overlap))
    overlaps.sort(key=lambda item: (item[0]["start_ns"], -_duration(item[0])))
    return [
        {"name": span["name"], "class_type": span["fields"].get("class_type"), "overlap_s": _seconds(overlap)}
        for span, overlap in overlaps
    ]


def _label(span: dict[str, Any] | None) -> str | None:
    if span is None:
        return None
    fields = span.get("fields", {})
    return f"{fields.get('class_type', '?')}#{fields.get('node_id', '?')}"


def analyze_trace(trace: dict[str, Any]) -> dict[str, Any]:
    if trace.get("schema") != TRACE_SCHEMA:
        raise ValueError(f"unsupported lifecycle trace schema {trace.get('schema')!r}")
    spans = [span for span in trace.get("spans", ()) if "start_ns" in span and "end_ns" in span]
    samples = [list(sample) for sample in trace.get("nvml", {}).get("samples", ())]
    by_id, children = _index(spans)
    prompts = [span for span in spans if span["name"] == "comfy.prompt"]
    if prompts:
        root = max(prompts, key=_duration)
        start, end, thread = root["start_ns"], root["end_ns"], root["thread"]
    elif spans:
        root = None
        start = min(span["start_ns"] for span in spans)
        end = max(span["end_ns"] for span in spans)
        thread = max(
            {span["thread"] for span in spans},
            key=lambda tid: sum(_duration(s) for s in spans if s["thread"] == tid),
        )
    else:
        raise ValueError("lifecycle trace contains no spans")
    inside = [span for span in spans if span["start_ns"] >= start and span["end_ns"] <= end]
    main = [span for span in inside if span["thread"] == thread and not span.get("instant")]

    exclusive_by_subsystem: dict[str, int] = defaultdict(int)
    for span in main:
        key = span["subsystem"]
        if key == "prompt":
            key = "between_nodes"
        elif key == "node":
            key = "node_body_other"
        exclusive_by_subsystem[key] += _exclusive(span, children)
    accounted = sum(exclusive_by_subsystem.values())
    wall = end - start
    if root is None:
        exclusive_by_subsystem["untraced"] = max(0, wall - accounted)

    nodes = []
    for span in main:
        if span["subsystem"] != "node":
            continue
        breakdown: dict[str, int] = defaultdict(int)
        for item in _descendants(span, children):
            if not item.get("instant"):
                breakdown[item["subsystem"]] += _exclusive(item, children)
        breakdown["node_body_other"] += _exclusive(span, children)
        nodes.append(
            {
                "node": _label(span),
                "start_s": _seconds(span["start_ns"] - start),
                "inclusive_s": _seconds(_duration(span)),
                "exclusive_s": {key: _seconds(value) for key, value in sorted(breakdown.items()) if value > 0},
                "result": span["fields"].get("result"),
            }
        )
    nodes.sort(key=lambda item: -item["inclusive_s"])

    residency = []
    for span in main:
        if span["subsystem"] != "model_residency":
            continue
        fields = span["fields"]
        moved = fields.get("moved_bytes")
        seconds = _duration(span) * _NS
        entry = {
            "start_s": _seconds(span["start_ns"] - start),
            "event": span["name"],
            "model_class": fields.get("model_class"),
            "moved_mib": None if moved is None else round(moved / _MIB, 1),
            "seconds": round(seconds, 3),
            "gib_per_s": None if not moved or seconds <= 0 else round(moved / seconds / (1 << 30), 2),
            "fully_unloaded": fields.get("fully_unloaded"),
            "node": _label(_ancestor(span, by_id, "comfy.node")),
            "inside_sampler": _ancestor(span, by_id, "sampler.cfg_guider_sample") is not None,
        }
        paging = fields.get("host_pressure_delta")
        if isinstance(paging, dict):
            entry["host_pressure_delta"] = dict(paging)
            before = fields.get("host_pressure_before") or {}
            after = fields.get("host_pressure_after") or {}
            entry["host_memory_available_start_kib"] = before.get("memavailable_kib")
            entry["host_memory_available_end_kib"] = after.get("memavailable_kib")
            entry["host_swap_free_start_kib"] = before.get("swapfree_kib")
            entry["host_swap_free_end_kib"] = after.get("swapfree_kib")
        residency.append(entry)

    samplers = []
    for span in main:
        if span["subsystem"] != "sampler":
            continue
        memory_end = span.get("memory_end") or {}
        memory_start = span.get("memory_start") or {}
        samplers.append(
            {
                "start_s": _seconds(span["start_ns"] - start),
                "seconds": _seconds(_duration(span)),
                "node": _label(_ancestor(span, by_id, "comfy.node")),
                "peak_allocated_mib": (
                    round(memory_end["peak_allocated"] / _MIB, 1)
                    if span.get("peaks_reset") and "peak_allocated" in memory_end
                    else None
                ),
                "peak_reserved_mib": (
                    round(memory_end["peak_reserved"] / _MIB, 1)
                    if span.get("peaks_reset") and "peak_reserved" in memory_end
                    else None
                ),
                "driver_free_start_mib": (
                    round(memory_start["driver_free"] / _MIB, 1) if "driver_free" in memory_start else None
                ),
                "residency_seconds": _seconds(
                    sum(
                        _exclusive(item, children)
                        for item in _descendants(span, children)
                        if item["subsystem"] in {"model_residency", "model_management", "allocator"}
                    )
                ),
                "nvml": _nvml_window(samples, span["start_ns"], span["end_ns"]),
            }
        )

    gaps = []
    node_spans = sorted((span for span in main if span["subsystem"] == "node"), key=lambda span: span["start_ns"])
    for previous, following in itertools.pairwise(node_spans):
        gap = following["start_ns"] - previous["end_ns"]
        if gap * _NS >= 0.05:
            gaps.append(
                {
                    "after": _label(previous),
                    "before": _label(following),
                    "seconds": _seconds(gap),
                    "nvml": _nvml_window(samples, previous["end_ns"], following["start_ns"]),
                }
            )

    stream_idle_boundaries = []
    for span in main:
        if span["subsystem"] not in {"node", "sampler"}:
            continue
        before = (span.get("memory_start") or {}).get("stream_idle")
        after = (span.get("memory_end") or {}).get("stream_idle")
        if before is not None or after is not None:
            stream_idle_boundaries.append({"span": span["name"], "node": _label(span), "start": before, "end": after})

    idle = []
    flow_instants = [span for span in inside if span.get("instant") and span["subsystem"] == "flow_metrics"]
    for idle_start, idle_end in _idle_intervals(samples, start, end):
        idle.append(
            {
                "start_s": _seconds(idle_start - start),
                "seconds": _seconds(idle_end - idle_start),
                "covered_by": _covering(main, idle_start, idle_end, thread)[:12],
                "flow_events": [span["name"] for span in flow_instants if idle_start <= span["start_ns"] <= idle_end][
                    :12
                ],
            }
        )

    flow_stages = []
    peak_start = None
    for span in sorted(inside, key=lambda item: item["start_ns"]):
        if span["subsystem"] == "sampler" and span.get("peaks_reset"):
            peak_start = span["start_ns"]
        if not (span.get("instant") and span["subsystem"] == "flow_metrics"):
            continue
        if span["name"].removeprefix("flow.") not in STAGE_WALL_EVENTS:
            continue
        elapsed_ms = span["fields"].get("elapsed_ms")
        if "memory_end" not in span or not isinstance(elapsed_ms, (int, float)):
            continue
        stage_end = span["start_ns"]
        stage_start = stage_end - int(elapsed_ms * 1e6)
        memory = span.get("memory_end") or {}
        residency_ns = sum(
            max(0, min(stage_end, item["end_ns"]) - max(stage_start, item["start_ns"]))
            for item in main
            if item["subsystem"] in {"model_residency", "allocator", "python_gc"}
        )
        flow_stages.append(
            {
                "stage": span["name"].removeprefix("flow."),
                "start_s": _seconds(stage_start - start),
                "seconds": round(elapsed_ms / 1000.0, 3),
                "peak_allocated_mib": (
                    round(memory["peak_allocated"] / _MIB, 1)
                    if peak_start is not None and "peak_allocated" in memory
                    else None
                ),
                "peak_reserved_mib": (
                    round(memory["peak_reserved"] / _MIB, 1)
                    if peak_start is not None and "peak_reserved" in memory
                    else None
                ),
                "peak_window_start_s": None if peak_start is None else _seconds(peak_start - start),
                "allocated_at_end_mib": (round(memory["allocated"] / _MIB, 1) if "allocated" in memory else None),
                "driver_free_at_end_mib": (round(memory["driver_free"] / _MIB, 1) if "driver_free" in memory else None),
                "residency_gc_seconds": _seconds(residency_ns),
                "nvml": _nvml_window(samples, stage_start, stage_end),
            }
        )
        if span.get("peaks_reset"):
            peak_start = stage_end

    gc_spans = [span for span in inside if span["subsystem"] == "python_gc"]
    return {
        "prompt_id": trace.get("prompt_id"),
        "wall_s": _seconds(wall),
        "dropped_spans": trace.get("dropped_spans", 0),
        "nvml_status": trace.get("nvml", {}).get("status"),
        "exclusive_by_subsystem_s": {
            key: _seconds(value) for key, value in sorted(exclusive_by_subsystem.items(), key=lambda kv: -kv[1])
        },
        "nodes": nodes,
        "inter_node_gaps": gaps,
        "residency": residency,
        "samplers": samplers,
        "flow_stages": flow_stages,
        "gpu_idle_intervals": idle,
        "stream_idle_boundaries": stream_idle_boundaries,
        "python_gc": {
            "count": len(gc_spans),
            "total_s": _seconds(sum(_duration(span) for span in gc_spans)),
            "max_s": _seconds(max((_duration(span) for span in gc_spans), default=0)),
        },
        "nvml_prompt": _nvml_window(samples, start, end),
    }


def format_report(summary: dict[str, Any], *, top: int = 15) -> str:
    lines = [
        f"# H3 lifecycle trace — prompt {summary.get('prompt_id')}",
        "",
        f"Prompt wall: **{summary['wall_s']:.3f} s**; NVML: {summary.get('nvml_status')}; "
        f"dropped spans: {summary.get('dropped_spans', 0)}",
    ]
    nvml = summary.get("nvml_prompt")
    if nvml:
        lines.append(
            f"Whole prompt: mean GPU util {nvml['mean_gpu_util_pct']}%, "
            f"idle (<{IDLE_UTIL_PCT}%) fraction {nvml['idle_fraction']}"
        )
    lines += ["", "## Exclusive time by subsystem (non-overlapping)", "", "| subsystem | s |", "|---|---:|"]
    lines += [f"| {key} | {value:.3f} |" for key, value in summary["exclusive_by_subsystem_s"].items()]
    lines += [
        "",
        "## Slowest nodes (inclusive)",
        "",
        "| node | start s | inclusive s | exclusive breakdown s |",
        "|---|---:|---:|---|",
    ]
    for node in summary["nodes"][:top]:
        breakdown = ", ".join(f"{key} {value:.2f}" for key, value in node["exclusive_s"].items())
        lines.append(f"| {node['node']} | {node['start_s']:.2f} | {node['inclusive_s']:.3f} | {breakdown} |")
    lines += [
        "",
        "## Sampler spans",
        "",
        "| start s | s | node | peak alloc MiB | peak reserved MiB | driver free at start MiB "
        "| residency s | mean util % | idle frac |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for sampler in summary["samplers"]:
        nv = sampler.get("nvml") or {}
        lines.append(
            f"| {sampler['start_s']:.2f} | {sampler['seconds']:.3f} | {sampler['node']} "
            f"| {sampler['peak_allocated_mib']} | {sampler['peak_reserved_mib']} "
            f"| {sampler['driver_free_start_mib']} | {sampler['residency_seconds']:.3f} "
            f"| {nv.get('mean_gpu_util_pct')} | {nv.get('idle_fraction')} |"
        )
    lines += [
        "",
        "## Flow stages",
        "",
        "Peaks cover the interval since the previous stage end (or sampler start), so a",
        "stage's peak may include hand-off work immediately before it.",
        "These occupancy peaks (allocated/reserved) are NOT the free/reclaimable-memory",
        "target in VDN admission required_mib; do not subtract or compare them directly.",
        "CUDA peak-counter resets affect process-wide telemetry while tracing is enabled.",
        "",
        "| stage | start s | s | peak alloc MiB | peak reserved MiB | alloc at end MiB "
        "| driver free at end MiB | residency+alloc+gc s | mean util % | idle frac |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in summary.get("flow_stages", ()):
        nv = stage.get("nvml") or {}
        lines.append(
            f"| {stage['stage']} | {stage['start_s']:.2f} | {stage['seconds']:.3f} | {stage['peak_allocated_mib']} "
            f"| {stage['peak_reserved_mib']} | {stage['allocated_at_end_mib']} | {stage['driver_free_at_end_mib']} "
            f"| {stage['residency_gc_seconds']:.3f} | {nv.get('mean_gpu_util_pct')} | {nv.get('idle_fraction')} |"
        )
    lines += [
        "",
        "## Model load / offload",
        "",
        "| start s | event | model | MiB moved | s | GiB/s | full unload | node | in sampler |",
        "|---:|---|---|---:|---:|---:|---|---|---|",
    ]
    for item in summary["residency"]:
        lines.append(
            f"| {item['start_s']:.2f} | {item['event']} | {item['model_class']} | {item['moved_mib']} "
            f"| {item['seconds']:.3f} | {item['gib_per_s']} | {item['fully_unloaded']} | {item['node']} "
            f"| {item['inside_sampler']} |"
        )
    paging_rows = [item for item in summary["residency"] if item.get("host_pressure_delta") is not None]
    if paging_rows:
        lines += [
            "",
            "## Host paging during model loads (Linux/WSL)",
            "",
            "Process faults cover all process threads; VM swap counters are system-wide,",
            "not proof of paging in a particular model-load operation.",
            "",
            "| model | load s | process major faults | system swap-in | system swap-out | MemAvailable start→end KiB |",
            "|---|---:|---:|---:|---:|---|",
        ]
        for item in paging_rows:
            paging = item["host_pressure_delta"]
            memory = f"{item.get('host_memory_available_start_kib')}→{item.get('host_memory_available_end_kib')}"
            lines.append(
                f"| {item['model_class']} | {item['seconds']:.3f} "
                f"| {paging.get('process_major_faults')} | {paging.get('system_pswpin')} "
                f"| {paging.get('system_pswpout')} | {memory} |"
            )
    lines += ["", "## GPU-idle intervals (NVML)", ""]
    if not summary["gpu_idle_intervals"]:
        lines.append("None detected (or NVML unavailable).")
    for interval in summary["gpu_idle_intervals"]:
        covered = "; ".join(
            f"{item['name']}{'(' + item['class_type'] + ')' if item['class_type'] else ''} {item['overlap_s']:.2f}s"
            for item in interval["covered_by"]
        )
        lines.append(f"- t={interval['start_s']:.2f}s for {interval['seconds']:.2f}s: {covered or 'no traced span'}")
        if interval["flow_events"]:
            lines.append(f"  - Flow events: {', '.join(interval['flow_events'])}")
    lines += ["", "## Gaps between nodes", ""]
    for gap in summary["inter_node_gaps"][:top]:
        nv = gap.get("nvml") or {}
        lines.append(f"- {gap['after']} → {gap['before']}: {gap['seconds']:.3f}s (util {nv.get('mean_gpu_util_pct')}%)")
    gc_summary = summary["python_gc"]
    lines += [
        "",
        f"Python GC: {gc_summary['count']} collections, {gc_summary['total_s']:.3f}s total, "
        f"max pause {gc_summary['max_s']:.3f}s",
    ]
    return "\n".join(lines) + "\n"


__all__ = ["analyze_trace", "format_report"]
