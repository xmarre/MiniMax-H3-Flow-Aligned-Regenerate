#!/usr/bin/env python3
"""CPU-only cross-repo oracle for partitioned exact-prefix development contracts."""
# ruff: noqa: I001

from __future__ import annotations

import argparse
import inspect
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace

import torch


def _validate_boundary_witness_transport() -> None:
    from h3_flow_regenerate.boundary_witness import BoundaryWitness, WITNESS_API, WITNESS_KEY
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from vdn_h3.boundary_witness import (
        WITNESS_API as VDN_WITNESS_API,
        WITNESS_KEY as VDN_WITNESS_KEY,
        claim_feature_witness,
    )
    from vdn_h3.partitioned_linear import _frame_offsets, _variable_features

    if (WITNESS_API, WITNESS_KEY) != (VDN_WITNESS_API, VDN_WITNESS_KEY):
        raise SystemExit("Flow/VDN bounded witness ABI differs")
    sizes = ((4, 5),) * 4 + ((3, 4),) * 5
    offsets = _frame_offsets(sizes)
    tokens = torch.arange(offsets[-1][1], dtype=torch.float32).reshape(-1, 1, 1) / 100
    weights = {
        "short_conv.k_sp.weight": torch.ones(1, 1, 5, 5) / 25,
        "short_conv.k_tm.weight": torch.ones(1, 1, 5) / 5,
    }
    branch = SimpleNamespace(short_conv=("k",))
    with tempfile.TemporaryDirectory() as directory:
        metrics = H3FlowMetrics()
        sink = BoundaryWitness(directory, metrics)
        options = {WITNESS_KEY: sink, "h3_flow_stage": "low"}
        observation = claim_feature_witness(options, block_index=0, plan_digest="a" * 64, mode="normal")
        reference = _variable_features(branch, weights, tokens, tokens, tokens, sizes, offsets)
        observed = _variable_features(branch, weights, tokens, tokens, tokens, sizes, offsets, observation=observation)
        if not all(torch.equal(old, new) for old, new in zip(reference, observed, strict=True)):
            raise SystemExit("Flow/VDN observation mutated actual features")
        observation.finish()
        if not sink.completed or len(metrics.events) != 1:
            raise SystemExit("Flow/VDN bounded witness did not finish")


def _root(value: str) -> Path:
    path = Path(value).resolve()
    if not path.is_dir():
        raise SystemExit(f"missing dependency checkout: {path}")
    return path


def _validate_same_grid_history_transport() -> None:
    """Prove canonical Flow/VDN control metadata reaches both Sol classifiers."""
    from h3_flow_regenerate.partitioned_prefix import PartitionedExactPrefixPlan
    from h3_flow_regenerate.partitioned_stage import PartitionedStagePlan
    from h3_flow_regenerate.partitioned_transformer import _vdn_external_contract
    from sol_h3 import interop
    from sol_h3.partitioned_history import (
        PARTITIONED_FLOW_IDENTITY,
        VDN_EXTERNAL_SEQUENCE_KEY,
        _partitioned_flow_replacement_identity,
        _partitioned_history_layout_valid,
        install_partitioned_history_bridge,
    )
    from vdn_h3.partitioned_runtime import _partitioned_query_summary
    from vdn_h3.partitioned_sequence import validate_flow_partition_contract

    flow = PartitionedExactPrefixPlan(
        video_start=7,
        temporal=5,
        prefix_t=2,
        source_grid_h=4,
        source_grid_w=6,
        target_grid_h=4,
        target_grid_w=6,
        same_grid_control=True,
    )
    contract = flow.to_contract()
    vdn = validate_flow_partition_contract(contract, sequence_rows=flow.sequence_rows)
    if vdn.source_rows != vdn.target_rows or vdn.sequence_rows != flow.sequence_rows:
        raise SystemExit("VDN changed canonical equal-grid control geometry")
    layout = SimpleNamespace(
        seq_len=flow.sequence_rows,
        segments=[(0, flow.video_start, "nonvideo"), (flow.video_start, flow.sequence_rows, "video")],
        signature=(PARTITIONED_FLOW_IDENTITY, "same-grid-ci"),
    )
    external = _vdn_external_contract(flow)
    options = {PARTITIONED_FLOW_IDENTITY: contract, VDN_EXTERNAL_SEQUENCE_KEY: external}
    if not _partitioned_history_layout_valid(options, layout):
        raise SystemExit("Sol rejected canonical equal-grid Flow/VDN history")
    vdn_values = {
        "state": SimpleNamespace(softmax_backend="grouped", query_position_owner_generation="same-grid-ci"),
        "cfg": {"radius": 1, "chunk": 1, "anchor_frames": "none"},
    }
    forward = SimpleNamespace(
        vdn_query_position_plan_v1=lambda opts, packed: _partitioned_query_summary(None, vdn_values, opts, packed),
    )
    install_partitioned_history_bridge()
    has_hook, mapped_identity = interop._mapped_vdn_history_identity(forward, options, layout)
    if not has_hook or mapped_identity is None:
        raise SystemExit("Sol rejected VDN's equal-grid mapped query-position history")
    prefix = torch.zeros((1, 24, 2, 8, 12))
    plan = PartitionedStagePlan(
        prefix=prefix,
        temporal=5,
        source_h=8,
        source_w=12,
        prefix_noise=prefix.clone(),
    )
    previous = object()
    values = {
        "layer": 0,
        "previous": previous,
        "plan": plan,
        "layout": SimpleNamespace(
            seq_len=flow.sequence_rows,
            segments=layout.segments,
            signature=("native-carrier",),
        ),
        "partitioned_layout": layout,
        "video_start": flow.video_start,
        "video_end": flow.sequence_rows,
        "carrier_prefix_rows": plan.prefix_t * plan.source_rows,
        "inner": SimpleNamespace(blocks=[object()]),
        "partition_contract": contract,
    }

    def patch():
        return None

    patch.__module__ = "h3_flow_regenerate.partitioned_transformer"
    patch.__qualname__ = "partitioned_diffusion_wrapper.<locals>.wrap.<locals>.call"
    classifier = SimpleNamespace(_closure_values=lambda _patch: values)
    identity = _partitioned_flow_replacement_identity(classifier, patch, 0)
    if identity is None or identity[1] is not previous or identity[0][1] != flow.semantic_digest:
        raise SystemExit("Sol rejected equal-grid Flow closure geometry or changed inherited ownership")
    stale = SimpleNamespace(seq_len=layout.seq_len - 1, segments=layout.segments, signature=layout.signature)
    if _partitioned_history_layout_valid(options, stale):
        raise SystemExit("Sol accepted a stale equal-grid control layout")
    external["flow_semantic_digest"] = "d" * 64
    if _partitioned_history_layout_valid(options, layout):
        raise SystemExit("Sol accepted a mismatched equal-grid VDN binding")


def _namespace_package(name: str, package_dir: Path) -> None:
    """Load source-contract modules without executing custom-node __init__.py."""
    if not package_dir.is_dir():
        raise SystemExit(f"missing package directory: {package_dir}")
    package = ModuleType(name)
    package.__package__ = name
    package.__path__ = [str(package_dir)]
    sys.modules[name] = package


def _validate_preprocess_transport() -> None:
    """Prove Flow metadata is consumable by Sol and published under VDN's key."""
    from h3_flow_regenerate.partitioned_stage import PartitionedStageRuntime
    from h3_flow_regenerate.partitioned_transformer import _stage_partitioned_attention_override
    from sol_h3.interop import VDN_PREPROCESS_KEY, provider_identity
    from sol_h3.runtime import BlockPatch, _preprocess_chain
    from vdn_h3.softmax_provider import PREPROCESS_KEY, preprocess as vdn_preprocess

    if VDN_PREPROCESS_KEY != PREPROCESS_KEY:
        raise SystemExit(f"Sol/VDN preprocessing key mismatch: Sol={VDN_PREPROCESS_KEY!r} VDN={PREPROCESS_KEY!r}")

    block_source = inspect.getsource(BlockPatch.__call__)
    required_publication = (
        'hasattr(previous, "attention_preprocess_v1")',
        "options[VDN_PREPROCESS_KEY] = vdn_preprocess",
    )
    missing = tuple(marker for marker in required_publication if marker not in block_source)
    if missing:
        raise SystemExit(f"Sol BlockPatch no longer publishes inherited preprocessing to VDN: {missing!r}")

    calls = []

    def terminal(original, q, k, v, heads, mask=None, **kw):
        del original, heads, mask, kw
        return q + k + v

    def inherited_transform(q, k, v, *, heads, **kw):
        del kw
        calls.append(int(heads))
        return q + 1, k + 2, v + 3

    def inherited(original, q, k, v, heads, mask=None, **kw):
        return terminal(original, q, k, v, heads, mask=mask, **kw)

    inherited.attention_preprocess_v1 = (inherited_transform, terminal)
    metrics = SimpleNamespace(increment=lambda *_args, **_kwargs: None)
    runtime = PartitionedStageRuntime(plan=None, metrics=metrics)
    flow_override = _stage_partitioned_attention_override(runtime, inherited, metrics)
    if getattr(flow_override, "_h3_flow_partitioned_provider_identity", None) != provider_identity(inherited):
        raise SystemExit("Flow partitioned provider identity diverged from Sol history-v1 semantics")
    contract = getattr(flow_override, "attention_preprocess_v1", None)
    if not isinstance(contract, tuple) or len(contract) != 2 or not callable(contract[0]):
        raise SystemExit("Flow partitioned override did not preserve attention_preprocess_v1 metadata")
    expected_leaf = contract[1]

    def sol_vdn_bridge(q, k, v, *, heads, transformer_options=None):
        processed_q, processed_k, processed_v, leaf = _preprocess_chain(
            flow_override,
            q,
            k,
            v,
            heads,
            {"transformer_options": transformer_options or {}},
        )
        if leaf is not expected_leaf:
            raise SystemExit("Sol preprocessing chain did not terminate at Flow's partition leaf")
        return processed_q, processed_k, processed_v

    q = torch.zeros((4, 2, 3), dtype=torch.float32)
    k = torch.zeros_like(q)
    v = torch.zeros_like(q)
    processed_q, processed_k, processed_v = vdn_preprocess(
        {PREPROCESS_KEY: sol_vdn_bridge},
        q,
        k,
        v,
        q.shape[1],
    )
    if calls != [q.shape[1]]:
        raise SystemExit(f"inherited Flow preprocessing executed {len(calls)} times instead of once")
    if not torch.equal(processed_q, q + 1):
        raise SystemExit("Flow->Sol->VDN preprocessing changed Q unexpectedly")
    if not torch.equal(processed_k, k + 2):
        raise SystemExit("Flow->Sol->VDN preprocessing changed K unexpectedly")
    if not torch.equal(processed_v, v + 3):
        raise SystemExit("Flow->Sol->VDN preprocessing changed V unexpectedly")


def _validate_high_attention_transport() -> None:
    from h3_flow_regenerate.metrics import H3FlowMetrics
    from h3_flow_regenerate.partitioned_scheduler import (
        VDN_PARTITIONED_BOUNDARY_QUERY_POLICY,
        _partitioned_high_stage_contract,
    )
    from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan
    from h3_flow_regenerate.partitioned_prefix import PartitionedExactPrefixPlan
    from h3_flow_regenerate.runtime import _flow_stage_contract, _high_stage_contract
    from sol_h3.contracts import Config
    from sol_h3.interop import dense_evaluation_warmup
    from vdn_h3.partitioned_grouped import build_partitioned_grouped_plan
    from vdn_h3.partitioned_runtime import _partitioned_local_force_dense
    from vdn_h3.partitioned_sequence import validate_flow_partition_contract
    from vdn_h3.window import window_bounds

    prefix = torch.zeros(1, 24, 12, 8, 12)
    plan = PartitionedStagePlan(prefix, 47, 4, 6, prefix.clone())
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    options = guider.model_options["transformer_options"]
    with _flow_stage_contract(guider, "probe"), _high_stage_contract(guider):
        if not dense_evaluation_warmup(Config(), 0, options):
            raise SystemExit("endpoint probe unexpectedly consumed the high startup exemption")
    with _flow_stage_contract(guider, "high"), _partitioned_high_stage_contract(guider, plan, H3FlowMetrics()):
        high = options[PARTITIONED_STAGE_KEY].plan
        if high.source_grid != high.target_grid or high.prefix is not plan.prefix:
            raise SystemExit("high attention changed target-grid prefix ownership")
        for count in (0, 1, 2):
            config = Config(dense_evaluations=count)
            for evaluation in (0, 1, 2):
                if dense_evaluation_warmup(config, evaluation, options) != (evaluation < count):
                    raise SystemExit("high attention bypassed the configured Sol startup policy")
        flow = PartitionedExactPrefixPlan(
            7, high.temporal, high.prefix_t, *high.source_grid, *high.target_grid, same_grid_control=True
        )
        vdn = validate_flow_partition_contract(flow.to_contract(), sequence_rows=flow.sequence_rows)
        grouped = build_partitioned_grouped_plan(
            vdn, bounds=window_bounds(47, 1, 5), anchor_frames="both", semantic_digest=flow.semantic_digest
        )
        saw_boundary_suffix = False
        saw_later_suffix = False
        for group in grouped.groups:
            force_dense, diagnostic, boundary_suffix = _partitioned_local_force_dense(
                group,
                "normal",
                prefix_t=vdn.prefix_t,
            )
            expected_boundary_suffix = bool(not group.query_prefix_domain and vdn.prefix_t in group.query_frames)
            if boundary_suffix != expected_boundary_suffix or diagnostic:
                raise SystemExit("high VDN boundary-query discriminator drifted")
            if group.query_prefix_domain:
                if not force_dense:
                    raise SystemExit("high VDN protected-prefix group lost dense ownership")
            elif expected_boundary_suffix:
                saw_boundary_suffix = True
                if not force_dense:
                    raise SystemExit("high VDN first generated boundary group lost dense continuity")
            else:
                saw_later_suffix = True
                if force_dense:
                    raise SystemExit("high VDN later generated group lost native sparse routing")
        if not saw_boundary_suffix or not saw_later_suffix:
            raise SystemExit("high VDN contract did not exercise boundary and later suffix groups")
        if VDN_PARTITIONED_BOUNDARY_QUERY_POLICY != "boundary_suffix_local_group_dense_v1":
            raise SystemExit("Flow boundary-query policy identity drifted")
    if options:
        raise SystemExit("high attention left sampler-stage state behind")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-boundary-witness", action="store_true")
    parser.add_argument("--sol", required=True)
    parser.add_argument("--vdn", required=True)
    args = parser.parse_args()
    sol_root = _root(args.sol)
    vdn_root = _root(args.vdn)
    local_root = str(Path(__file__).resolve().parents[1])
    sys.path.insert(0, local_root)
    _namespace_package("sol_h3", sol_root / "sol_h3")
    _namespace_package("vdn_h3", vdn_root / "vdn_h3")

    from h3_flow_regenerate.partitioned_diagnostics import (
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_BYPASS as FLOW_VDN_LINEAR_BYPASS,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API as FLOW_VDN_LINEAR_DIAGNOSTIC_API,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_KEY as FLOW_VDN_LINEAR_DIAGNOSTIC_KEY,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_NORMAL as FLOW_VDN_LINEAR_NORMAL,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE as FLOW_VDN_LINEAR_RAW_MEASURE,
        PARTITIONED_VDN_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL as FLOW_VDN_LINEAR_SUPPRESS,
        PARTITIONED_SOFTMAX_DIAGNOSTIC_API as FLOW_SOFTMAX_DIAGNOSTIC_API,
        PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX as FLOW_SOFTMAX_DENSE_SUFFIX,
        PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY as FLOW_SOFTMAX_DIAGNOSTIC_KEY,
        PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL as FLOW_SOFTMAX_NORMAL,
        PARTITIONED_VDN_TEMPORAL_CARRIER_API as FLOW_CARRIER_API,
        PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION as FLOW_CARRIER_DESTINATION,
        PARTITIONED_VDN_TEMPORAL_CARRIER_KEY as FLOW_CARRIER_KEY,
        PARTITIONED_VDN_TEMPORAL_CARRIER_MAPPING_POLICY as FLOW_CARRIER_MAPPING_POLICY,
        PARTITIONED_VDN_TEMPORAL_CARRIER_NATIVE as FLOW_CARRIER_NATIVE,
        build_vdn_temporal_carrier_contract,
    )
    from h3_flow_regenerate.partitioned_prefix import (
        PARTITIONED_PREFIX_KEY as FLOW_CONTRACT_KEY,
        PARTITIONED_PREFIX_TOPOLOGY,
        PartitionedExactPrefixPlan,
    )
    from h3_flow_regenerate.partitioned_scheduler import (
        VDN_PARTITIONED_BOUNDARY_QUERY_API as FLOW_BOUNDARY_QUERY_API,
        VDN_PARTITIONED_BOUNDARY_QUERY_POLICY as FLOW_BOUNDARY_QUERY_POLICY,
    )
    from h3_flow_regenerate.partitioned_transformer import (
        PARTITIONED_PREFIX_KEY as FLOW_RUNTIME_KEY,
        VDN_PARTITIONED_SEQUENCE_API as FLOW_VDN_API,
        VDN_PARTITIONED_SEQUENCE_MODE as FLOW_VDN_MODE,
        _vdn_external_contract,
    )
    from sol_h3.mapped_neighbors import compile_descriptor, validate_wire_map
    from sol_h3.partitioned_history import (
        PARTITIONED_FLOW_IDENTITY,
        _partitioned_flow_replacement_identity,
        VDN_EXTERNAL_SEQUENCE_KEY,
        VDN_PARTITIONED_SEQUENCE_API as SOL_VDN_API,
        VDN_PARTITIONED_SEQUENCE_MODE as SOL_VDN_MODE,
        _partitioned_history_layout_valid,
    )
    from sol_h3.partitioned_request import PARTITIONED_REQUEST_ABI, partitioned_request_attention
    from vdn_h3.partitioned_grouped import build_partitioned_grouped_plan
    from vdn_h3.partitioned_linear import partitioned_frame_contract
    from vdn_h3.partitioned_runtime import (
        VDN_PARTITIONED_BOUNDARY_QUERY_API,
        VDN_PARTITIONED_BOUNDARY_QUERY_POLICY,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE,
        VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_API,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY,
        VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL,
        _partitioned_local_force_dense,
    )
    from vdn_h3.partitioned_sequence import (
        PARTITIONED_PREFIX_KEY as VDN_FLOW_KEY,
        PARTITIONED_PREFIX_TOPOLOGY as VDN_TOPOLOGY,
        VDN_PARTITIONED_SEQUENCE_API,
        VDN_PARTITIONED_SEQUENCE_MODE,
        VDN_TEMPORAL_CARRIER_API,
        VDN_TEMPORAL_CARRIER_DESTINATION,
        VDN_TEMPORAL_CARRIER_KEY,
        VDN_TEMPORAL_CARRIER_MAPPING_POLICY,
        VDN_TEMPORAL_CARRIER_NATIVE,
        make_temporal_carrier_contract,
        make_vdn_partitioned_external_contract,
        validate_flow_partition_contract,
        validate_temporal_carrier_contract,
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
    if VDN_PARTITIONED_SEQUENCE_API != 4:
        raise SystemExit("unexpected partitioned VDN external-sequence API")
    if not (
        FLOW_VDN_API == SOL_VDN_API == VDN_PARTITIONED_SEQUENCE_API
        and FLOW_VDN_MODE == SOL_VDN_MODE == VDN_PARTITIONED_SEQUENCE_MODE
    ):
        raise SystemExit("Flow/Sol/VDN partitioned external-sequence API or mode identity diverged")
    if not isinstance(PARTITIONED_REQUEST_ABI, str) or not PARTITIONED_REQUEST_ABI:
        raise SystemExit("Sol partitioned request ABI is missing")
    if "force_dense" not in inspect.signature(partitioned_request_attention).parameters:
        raise SystemExit("Sol partitioned request no longer exposes the force_dense discriminator")
    if (
        FLOW_VDN_LINEAR_DIAGNOSTIC_API != VDN_PARTITIONED_LINEAR_DIAGNOSTIC_API
        or FLOW_VDN_LINEAR_DIAGNOSTIC_KEY != VDN_PARTITIONED_LINEAR_DIAGNOSTIC_KEY
        or FLOW_VDN_LINEAR_NORMAL != VDN_PARTITIONED_LINEAR_DIAGNOSTIC_NORMAL
        or FLOW_VDN_LINEAR_BYPASS != VDN_PARTITIONED_LINEAR_DIAGNOSTIC_BYPASS
        or FLOW_VDN_LINEAR_RAW_MEASURE != VDN_PARTITIONED_LINEAR_DIAGNOSTIC_RAW_TOKEN_MEASURE
        or FLOW_VDN_LINEAR_SUPPRESS != VDN_PARTITIONED_LINEAR_DIAGNOSTIC_SUPPRESS_CROSS_GRID_TEMPORAL
    ):
        raise SystemExit("Flow/VDN partitioned linear diagnostic contract diverged")

    if (
        FLOW_SOFTMAX_DIAGNOSTIC_API != VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_API
        or FLOW_SOFTMAX_DIAGNOSTIC_KEY != VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_KEY
        or FLOW_SOFTMAX_NORMAL != VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_NORMAL
        or FLOW_SOFTMAX_DENSE_SUFFIX != VDN_PARTITIONED_SOFTMAX_DIAGNOSTIC_DENSE_SUFFIX
    ):
        raise SystemExit("Flow/VDN partitioned softmax diagnostic contract diverged")
    if (
        FLOW_BOUNDARY_QUERY_API != VDN_PARTITIONED_BOUNDARY_QUERY_API
        or FLOW_BOUNDARY_QUERY_POLICY != VDN_PARTITIONED_BOUNDARY_QUERY_POLICY
    ):
        raise SystemExit("Flow/VDN continuation boundary-query contract diverged")

    if (
        FLOW_CARRIER_API != VDN_TEMPORAL_CARRIER_API
        or FLOW_CARRIER_KEY != VDN_TEMPORAL_CARRIER_KEY
        or FLOW_CARRIER_NATIVE != VDN_TEMPORAL_CARRIER_NATIVE
        or FLOW_CARRIER_DESTINATION != VDN_TEMPORAL_CARRIER_DESTINATION
        or FLOW_CARRIER_MAPPING_POLICY != VDN_TEMPORAL_CARRIER_MAPPING_POLICY
    ):
        raise SystemExit("Flow/VDN temporal-carrier numerical-policy ABI diverged")

    carrier_args = {
        "policy": FLOW_CARRIER_DESTINATION,
        "flow_semantic_digest": "a" * 64,
        "diagnostic_mode": FLOW_VDN_LINEAR_NORMAL,
        "short_conv_spec": "vdn_solve_short_conv_v1|heads=56|head_dim=128|conv=k,v|spatial=5x5|temporal=5|a_fp32=1",
    }
    flow_carrier = build_vdn_temporal_carrier_contract(**carrier_args)
    vdn_carrier = make_temporal_carrier_contract(**carrier_args)
    if flow_carrier != vdn_carrier:
        raise SystemExit(f"Flow/VDN temporal-carrier numerical digest diverged: {flow_carrier!r} != {vdn_carrier!r}")
    policy, validated_carrier = validate_temporal_carrier_contract(
        flow_carrier,
        flow_semantic_digest=carrier_args["flow_semantic_digest"],
        diagnostic_mode=carrier_args["diagnostic_mode"],
        short_conv_spec=carrier_args["short_conv_spec"],
    )
    if policy != FLOW_CARRIER_DESTINATION or validated_carrier != flow_carrier:
        raise SystemExit("VDN rejected the canonical Flow temporal-carrier policy leaf")
    native_policy, native_contract = validate_temporal_carrier_contract(
        None,
        flow_semantic_digest=carrier_args["flow_semantic_digest"],
        diagnostic_mode=carrier_args["diagnostic_mode"],
        short_conv_spec=carrier_args["short_conv_spec"],
    )
    if native_policy != FLOW_CARRIER_NATIVE or native_contract is not None:
        raise SystemExit("VDN absence semantics no longer preserve the native temporal-carrier policy")

    _validate_preprocess_transport()
    if args.require_boundary_witness:
        _validate_boundary_witness_transport()
    history_source = inspect.getsource(_partitioned_flow_replacement_identity)
    if "repr(partitioned_layout.signature)" not in history_source:
        raise SystemExit("Sol history no longer keys partitioned numerical identity by full layout signature")

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

    def fake_partition_patch():
        return None

    fake_partition_patch.__module__ = "h3_flow_regenerate.partitioned_transformer"
    fake_partition_patch.__qualname__ = "partitioned_diffusion_wrapper.<locals>.wrap.<locals>.call"
    stage_plan = SimpleNamespace(
        prefix_t=flow.prefix_t,
        temporal=flow.temporal,
        source_rows=flow.source_rows,
        target_rows=flow.target_rows,
        prefix_rows=flow.prefix_t * flow.target_rows,
        partitioned_rows=flow.prefix_t * flow.target_rows + (flow.temporal - flow.prefix_t) * flow.source_rows,
        target_hw=(flow.target_grid_h * 2, flow.target_grid_w * 2),
    )
    native_rows = flow.video_start + flow.temporal * flow.source_rows
    partitioned_rows = flow.sequence_rows
    native_layout = SimpleNamespace(
        seq_len=native_rows,
        segments=[(0, flow.video_start, "nonvideo"), (flow.video_start, native_rows, "video")],
        signature=("native-carrier",),
    )
    legacy_partitioned_layout = SimpleNamespace(
        seq_len=partitioned_rows,
        segments=[(0, flow.video_start, "nonvideo"), (flow.video_start, partitioned_rows, "video")],
        signature=(PARTITIONED_FLOW_IDENTITY, "legacy"),
    )
    candidate_partitioned_layout = SimpleNamespace(
        seq_len=partitioned_rows,
        segments=legacy_partitioned_layout.segments,
        signature=(
            PARTITIONED_FLOW_IDENTITY,
            "legacy",
            ("h3_flow_partitioned_position_policy_v1", "source_carrier", 12345),
        ),
    )
    closure_values = {
        "layer": 0,
        "previous": None,
        "plan": stage_plan,
        "layout": native_layout,
        "partitioned_layout": legacy_partitioned_layout,
        "video_start": flow.video_start,
        "video_end": native_rows,
        "carrier_prefix_rows": flow.prefix_t * flow.source_rows,
        "inner": SimpleNamespace(blocks=[object()]),
        "partition_contract": contract,
    }

    class _Interop:
        @staticmethod
        def _closure_values(_patch):
            return closure_values

    legacy_identity = _partitioned_flow_replacement_identity(_Interop(), fake_partition_patch, 0)
    if legacy_identity is None:
        raise SystemExit("Sol history rejected the baseline partitioned Flow replacement identity")
    closure_values["partitioned_layout"] = candidate_partitioned_layout
    candidate_identity = _partitioned_flow_replacement_identity(_Interop(), fake_partition_patch, 0)
    if candidate_identity is None:
        raise SystemExit("Sol history rejected the candidate position-policy layout signature")
    if candidate_identity[0] == legacy_identity[0]:
        raise SystemExit("Sol history failed to distinguish the candidate position policy")

    vdn_plan = validate_flow_partition_contract(contract, sequence_rows=flow.sequence_rows)
    if (
        vdn_plan.sequence_rows != flow.sequence_rows
        or vdn_plan.source_rows != flow.source_rows
        or vdn_plan.target_rows != flow.target_rows
    ):
        raise SystemExit("VDN changed Flow partition geometry")

    frame_sizes, measure_scales = partitioned_frame_contract(vdn_plan)
    if frame_sizes != ((3, 4), (3, 4), (2, 3), (2, 3), (2, 3)):
        raise SystemExit(f"VDN variable-grid linear frame contract drifted: {frame_sizes!r}")
    if measure_scales != (0.5, 0.5, 1.0, 1.0, 1.0):
        raise SystemExit(f"VDN variable-grid linear physical measure drifted: {measure_scales!r}")

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
        expected_boundary_suffix = bool(not group.query_prefix_domain and vdn_plan.prefix_t in group.query_frames)
        normal_route = _partitioned_local_force_dense(
            group,
            FLOW_SOFTMAX_NORMAL,
            prefix_t=vdn_plan.prefix_t,
        )
        diagnostic_route = _partitioned_local_force_dense(
            group,
            FLOW_SOFTMAX_DENSE_SUFFIX,
            prefix_t=vdn_plan.prefix_t,
        )
        if group.query_prefix_domain:
            if normal_route != (True, False, False):
                raise SystemExit("normal partitioned prefix group lost dense ownership")
            if diagnostic_route != (True, False, False):
                raise SystemExit("dense-suffix diagnostic changed prefix-query ownership")
        else:
            saw_suffix_group = True
            if expected_boundary_suffix:
                if normal_route != (True, False, True):
                    raise SystemExit("normal boundary suffix group lost dense continuity")
                if diagnostic_route != (True, False, True):
                    raise SystemExit("dense-suffix diagnostic misclassified boundary suffix ownership")
            else:
                if normal_route != (False, False, False):
                    raise SystemExit("later generated-suffix group no longer uses sparse Sol selection")
                if diagnostic_route != (True, True, False):
                    raise SystemExit("dense-suffix diagnostic did not force only later suffix groups dense")
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

    _validate_same_grid_history_transport()
    _validate_high_attention_transport()
    print(
        "partitioned exact-prefix contracts: OK ",
        f"abi={PARTITIONED_REQUEST_ABI} groups={len(grouped.groups)} sequence_rows={flow.sequence_rows}",
        "same_grid_history=True",
        "high_prefix_attention=True",
    )


if __name__ == "__main__":
    main()
