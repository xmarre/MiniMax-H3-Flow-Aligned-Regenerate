"""Backend capabilities survive the actual block-options publication boundary."""

import builtins
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_attention import PARTITIONED_ATTENTION_PROVIDER_KEY
from h3_flow_regenerate.partitioned_prefix import PartitionedExactPrefixPlan
from h3_flow_regenerate.partitioned_stage import PartitionedStagePlan, PartitionedStageRuntime
from h3_flow_regenerate.partitioned_transformer import (
    _partitioned_transformer_options,
    _stage_partitioned_attention_override,
)


def setup_options():
    prefix = torch.zeros(1, 24, 1, 4, 4)
    runtime = PartitionedStageRuntime(
        plan=PartitionedStagePlan(prefix, 3, 2, 2, prefix.clone()), metrics=H3FlowMetrics()
    )
    override = _stage_partitioned_attention_override(runtime, None, runtime.metrics)
    contract = PartitionedExactPrefixPlan(3, 3, 1, 1, 1, 2, 2).to_contract()
    layout = SimpleNamespace(seq_len=9)
    options = {"optimized_attention_override": override, "sol_h3_runtime_v1": {"backend": "sol"}}
    return runtime, options, layout, contract


def publish(runtime, options, layout, contract):
    return _partitioned_transformer_options(options, layout, contract, runtime.metrics, runtime=runtime, block_index=0)


def install_sol(monkeypatch, attention):
    package = ModuleType("sol_h3")
    module = ModuleType("sol_h3.partitioned_request")
    module.partitioned_request_attention = attention
    monkeypatch.setitem(sys.modules, "sol_h3", package)
    monkeypatch.setitem(sys.modules, "sol_h3.partitioned_request", module)


def test_published_provider_transports_group_producer_readonly_and_actual_dispatch(monkeypatch):
    calls = []

    def attention(q, k, v, **kwargs):
        calls.append(kwargs)
        return q + k[: len(q)] + v[: len(q)]

    producer = object()

    def execute(*args, **kwargs):
        return producer

    attention.grouped_kv_v1 = execute
    attention._vdn_readonly_query_contract_v1 = True
    install_sol(monkeypatch, attention)
    runtime, options, layout, contract = setup_options()
    bound = publish(runtime, options, layout, contract)
    provider = bound[PARTITIONED_ATTENTION_PROVIDER_KEY]
    assert provider is publish(runtime, dict(options), layout, contract)[PARTITIONED_ATTENTION_PROVIDER_KEY]
    assert provider.grouped_kv_v1 is execute
    assert provider._vdn_readonly_query_contract_v1 is True
    assert provider.grouped_kv_v1() is producer
    q = torch.ones(2, 1, 8)
    torch.testing.assert_close(provider(q, q, q, transformer_options=bound), q * 3)
    assert calls[0]["transformer_options"] is bound
    assert runtime.metrics.counters["partitioned_sol_kernel_calls"] == 1
    with pytest.raises(RuntimeError, match="Sol-selected"):
        provider(q, q, q, transformer_options={})
    assert len(calls) == 1


def test_old_sol_keeps_unmodified_provider_and_capability_changes_rebind(monkeypatch):
    def attention(*args, **kwargs):
        raise AssertionError("binding must not execute attention")

    install_sol(monkeypatch, attention)
    runtime, options, layout, contract = setup_options()
    raw = options["optimized_attention_override"].partitioned_attention_v1
    assert publish(runtime, options, layout, contract)[PARTITIONED_ATTENTION_PROVIDER_KEY] is raw
    attention._vdn_readonly_query_contract_v1 = True
    first = publish(runtime, options, layout, contract)[PARTITIONED_ATTENTION_PROVIDER_KEY]
    attention.grouped_kv_v1 = lambda *args, **kwargs: None
    second = publish(runtime, options, layout, contract)[PARTITIONED_ATTENTION_PROVIDER_KEY]
    assert first is not second
    assert not hasattr(first, "grouped_kv_v1")
    assert second.grouped_kv_v1 is attention.grouped_kv_v1
    del attention.grouped_kv_v1
    del attention._vdn_readonly_query_contract_v1
    assert publish(runtime, options, layout, contract)[PARTITIONED_ATTENTION_PROVIDER_KEY] is raw


def test_native_backend_never_imports_or_inherits_sol_capabilities(monkeypatch):
    runtime, options, layout, contract = setup_options()
    options.pop("sol_h3_runtime_v1")
    original_import = builtins.__import__

    def no_sol(name, *args, **kwargs):
        if name == "sol_h3" or name.startswith("sol_h3."):
            raise AssertionError("native provider must remain independent of Sol")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_sol)
    provider = publish(runtime, options, layout, contract)[PARTITIONED_ATTENTION_PROVIDER_KEY]
    assert provider is options["optimized_attention_override"].partitioned_attention_v1
    assert not hasattr(provider, "grouped_kv_v1")
    assert not hasattr(provider, "_vdn_readonly_query_contract_v1")
