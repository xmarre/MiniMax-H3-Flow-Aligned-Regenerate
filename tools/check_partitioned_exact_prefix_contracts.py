#!/usr/bin/env python3
"""CPU-only cross-repo oracle for partitioned exact-prefix development contracts."""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace


def _root(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise SystemExit(f"missing dependency checkout: {path}")
    return path


def _namespace_package(name: str, package_dir: Path) -> None:
    """Load source-contract modules without executing custom-node __init__.py."""
    if not package_dir.is_dir():
        raise SystemExit(f"missing package directory: {package_dir}")
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(package_dir)]
    sys.modules[name] = package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sol", required=True)
    parser.add_argument("--vdn", required=True)
    args = parser.parse_args()
    sol_root = _root(args.sol)
    vdn_root = _root(args.vdn)
    local_root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, local_root)
    _namespace_package("sol_h3", sol_root / "sol_h3")
    _namespace_package("vdn_h3", vdn_root / "vdn_h3")

    from h3_flow_regenerate.partitioned_prefix import (
        PARTITIONED_PREFIX_KEY as FLOW_CONTRACT_KEY,
        PARTITIONED_PREFIX_TOPOLOGY,
        PartitionedExactPrefixPlan,
    )
    from h3_flow_regenerate.partitioned_transformer import (
        PARTITIONED_PREFIX_KEY as FLOW_RUNTIME_KEY,
        _vdn_external_contract,
    )
    from sol_h3.mapped_neighbors import compile_descriptor, validate_wire_map
    from sol_h3.partitioned_history import (
        PARTITIONED_FLOW_IDENTITY,
        VDN_EXTERNAL_SEQUENCE_KEY,
        _partitioned_history_layout_valid,
    )
    from sol_h3.partitioned_request import PARTITIONED_REQUEST_ABI
    from vdn_h3.partitioned_grouped import build_partitioned_grouped_plan
    from vdn_h3.partitioned_sequence import (
        PARTITIONED_PREFIX_KEY as VDN_FLOW_KEY,
        PARTITIONED_PREFIX_TOPOLOGY as VDN_TOPOLOGY,
        VDN_PARTITIONED_SEQUENCE_API,
        make_vdn_partitioned_external_contract,
        validate_flow_partition_contract,
    )

    keys = {
        FLOW_RUNTIME_KEY,
        FLOW_CONTRACT_KEY,
        PARTITIONED_FLOW_IDENTITY,
        VDN_FLOW_KEY,
    }
    if len(keys) != 1:
        raise SystemExit(f"partitioned Flow key mismatch: {sorted(keys)}")
    if PARTITIONED_PREFIX_TOPOLOGY != VDN_TOPOLOGY:
        raise SystemExit("Flow/VDN partition topology mismatch")
    if VDN_PARTITIONED_SEQUENCE_API != 3:
        raise SystemExit("unexpected partitioned VDN external-sequence API")
    if not isinstance(PARTITIONED_REQUEST_ABI, str) or not PARTITIONED_REQUEST_ABI:
        raise SystemExit("Sol partitioned request ABI is missing")

    flow = PartitionedExactPrefixPlan(
        video_start=7,
        temporal=5,
        prefix_t=2,
        source_grid_h=2,
        source_grid_w=3,
        target_grid_h=3,
        target_grid_w=4,
    )
    contract = flow.to_contract()
    vdn_plan = validate_flow_partition_contract(contract, sequence_rows=flow.sequence_rows)
    if (
        vdn_plan.sequence_rows != flow.sequence_rows
        or vdn_plan.source_rows != flow.source_rows
        or vdn_plan.target_rows != flow.target_rows
    ):
        raise SystemExit("VDN changed Flow partition geometry")

    flow_external = _vdn_external_contract(flow)
    vdn_external = make_vdn_partitioned_external_contract(vdn_plan)
    if flow_external != vdn_external:
        raise SystemExit(f"Flow/VDN external-sequence binding mismatch:\nFlow={flow_external!r}\nVDN={vdn_external!r}")

    grouped = build_partitioned_grouped_plan(
        vdn_plan,
        bounds=((0, 1), (0, 2), (1, 3), (2, 4), (3, 4)),
        anchor_frames="none",
        semantic_digest=contract["semantic_digest"],
    )
    wires = grouped.wires("vdn-ci-owner")
    if len(wires) != len(grouped.groups) or not wires:
        raise SystemExit("VDN partitioned grouped plan did not produce provider-v4 wires")

    saw_suffix_group = False
    saw_mapped_descriptor = False
    for group, wire in zip(grouped.groups, wires, strict=True):
        validated = validate_wire_map(
            wire,
            q_rows=group.q_rows,
            kv_rows=group.kv_rows,
            sink_rows=group.sink_rows,
        )
        descriptor = compile_descriptor(validated)
        if group.prefix_k_range is not None and group.prefix_k_range[0] != group.sink_rows:
            raise SystemExit("biased target-prefix K rows are not contiguous with the global sink")
        if not group.query_prefix_domain:
            saw_suffix_group = True
            if descriptor is not None:
                saw_mapped_descriptor = True
    if not saw_suffix_group:
        raise SystemExit("VDN grouped oracle produced no generated-suffix query group")
    if not saw_mapped_descriptor:
        raise SystemExit("VDN grouped oracle did not exercise mapped-neighbor metadata")

    layout = SimpleNamespace(
        seq_len=flow.sequence_rows,
        segments=[(0, flow.video_start, "nonvideo"), (flow.video_start, flow.sequence_rows, "video")],
        signature=(FLOW_CONTRACT_KEY, "cross-repo-ci"),
    )
    options = {
        FLOW_CONTRACT_KEY: contract,
        VDN_EXTERNAL_SEQUENCE_KEY: flow_external,
    }
    if not _partitioned_history_layout_valid(options, layout):
        raise SystemExit("Sol history rejected the canonical Flow/VDN partitioned layout")
    stale = SimpleNamespace(
        seq_len=flow.sequence_rows - 1,
        segments=layout.segments,
        signature=layout.signature,
    )
    if _partitioned_history_layout_valid(options, stale):
        raise SystemExit("Sol history accepted stale partitioned layout geometry")

    print(
        "partitioned exact-prefix contracts: OK ",
        f"abi={PARTITIONED_REQUEST_ABI} groups={len(grouped.groups)} sequence_rows={flow.sequence_rows}",
    )


if __name__ == "__main__":
    main()
