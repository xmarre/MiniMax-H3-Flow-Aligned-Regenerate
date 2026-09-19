from types import SimpleNamespace

import torch

from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_scheduler import _partitioned_stage_contract
from h3_flow_regenerate.partitioned_stage import (
    PARTITIONED_STAGE_KEY,
    PartitionedStagePlan,
    PartitionedStageRuntime,
)
from h3_flow_regenerate.partitioned_transformer import _stage_partitioned_attention_override


def _provider_a(*_args, **_kwargs):
    raise AssertionError("provider should not execute in identity regression")


def _provider_b(*_args, **_kwargs):
    raise AssertionError("provider should not execute in identity regression")


def _dynamic_preprocess_wrapper(previous, delta):
    # Model-function wrappers such as MiniMax H3 Untwist rebuild this outer
    # callable every H3 evaluation while retaining the same terminal provider.
    # The transform reads per-call state at execution time; use a captured delta
    # here to prove the partitioned owner rebinds to the newest callable rather
    # than accidentally freezing the first one.
    def preprocess(q, k, v, heads, **_kwargs):
        assert heads == 1
        return q + float(delta), k, v

    def override(original, q, k, v, heads, *args, **kwargs):
        q, k, v = preprocess(q, k, v, heads, **kwargs)
        if previous is not None:
            return previous(original, q, k, v, heads, *args, **kwargs)
        return original(q, k, v, heads, *args, **kwargs)

    override.attention_preprocess_v1 = (preprocess, previous)
    return override


def _copy_nested_dicts(value):
    """Mirror ComfyUI's recursive transformer-options copy semantics."""
    copied = value.copy()
    for key, item in value.items():
        if isinstance(item, dict):
            copied[key] = _copy_nested_dicts(item)
        elif isinstance(item, list):
            copied[key] = item.copy()
    return copied


def _plan():
    prefix = torch.zeros((1, 24, 1, 4, 4), dtype=torch.float32)
    return PartitionedStagePlan(
        prefix=prefix,
        temporal=2,
        source_h=2,
        source_w=2,
        prefix_noise=prefix.clone(),
    )


def test_partitioned_stage_owner_survives_comfy_recursive_option_copies():
    metrics = H3FlowMetrics()
    guider = SimpleNamespace(model_options={"transformer_options": {}})

    with _partitioned_stage_contract(guider, _plan(), metrics):
        published = guider.model_options["transformer_options"]
        owner = published[PARTITIONED_STAGE_KEY]
        assert isinstance(owner, PartitionedStageRuntime)

        first_call_options = _copy_nested_dicts(published)
        second_call_options = _copy_nested_dicts(published)
        first_owner = first_call_options[PARTITIONED_STAGE_KEY]
        second_owner = second_call_options[PARTITIONED_STAGE_KEY]
        assert first_owner is owner
        assert second_owner is owner

        first = _stage_partitioned_attention_override(first_owner, _provider_a, metrics)
        repeated = _stage_partitioned_attention_override(second_owner, _provider_a, metrics)
        assert repeated is first
        assert first._h3_flow_partitioned_previous is _provider_a

        changed_provider = _stage_partitioned_attention_override(owner, _provider_b, metrics)
        changed_provider_repeated = _stage_partitioned_attention_override(owner, _provider_b, metrics)
        assert changed_provider is changed_provider_repeated
        assert changed_provider is not first
        assert changed_provider._h3_flow_partitioned_previous is _provider_b
        assert len(owner.attention_provider_cache) == 2

    assert PARTITIONED_STAGE_KEY not in guider.model_options["transformer_options"]

    with _partitioned_stage_contract(guider, _plan(), metrics):
        fresh_owner = guider.model_options["transformer_options"][PARTITIONED_STAGE_KEY]
        assert fresh_owner is not owner
        fresh = _stage_partitioned_attention_override(fresh_owner, _provider_a, metrics)
        assert fresh is not first

    assert metrics.counters["partitioned_attention_provider_creations"] == 3
    assert metrics.counters["partitioned_attention_provider_reuses"] == 2


def test_equivalent_rebuilt_preprocess_wrapper_reuses_partition_owner_and_rebinds_latest_transform():
    metrics = H3FlowMetrics()
    runtime = PartitionedStageRuntime(plan=_plan(), metrics=metrics)

    first_outer = _dynamic_preprocess_wrapper(_provider_a, 1.0)
    first = _stage_partitioned_attention_override(runtime, first_outer, metrics)

    second_outer = _dynamic_preprocess_wrapper(_provider_a, 2.0)
    repeated = _stage_partitioned_attention_override(runtime, second_outer, metrics)

    # Sol history-v1 sees these as the same numerical provider: the preprocess
    # implementation name is unchanged and the terminal provider object is the
    # same. Partitioned Flow must therefore keep one visible provider identity.
    assert repeated is first
    assert repeated._h3_flow_partitioned_previous is second_outer
    assert repeated._h3_flow_partitioned_provider_identity[2] == id(_provider_a)

    # The stable partition owner must nevertheless execute the CURRENT call's
    # preprocessing function. This prevents caching from freezing dynamic
    # per-step configuration carried by a reconstructed outer wrapper.
    apply_preprocess, _partition_leaf = repeated.attention_preprocess_v1
    q = torch.zeros((1, 1, 1, 1), dtype=torch.float32)
    k = q.clone()
    v = q.clone()
    q_out, k_out, v_out = apply_preprocess(q, k, v, 1)
    assert torch.equal(q_out, torch.full_like(q, 2.0))
    assert torch.equal(k_out, k)
    assert torch.equal(v_out, v)

    # A genuinely different terminal dense provider remains a numerical history
    # transition and receives a new partitioned owner.
    changed_outer = _dynamic_preprocess_wrapper(_provider_b, 3.0)
    changed = _stage_partitioned_attention_override(runtime, changed_outer, metrics)
    assert changed is not first
    assert changed._h3_flow_partitioned_provider_identity[2] == id(_provider_b)

    assert metrics.counters["partitioned_attention_provider_creations"] == 2
    assert metrics.counters["partitioned_attention_provider_reuses"] == 1
    assert metrics.counters["partitioned_attention_equivalent_provider_rebindings"] == 1
    assert metrics.counters["partitioned_attention_inherited_provider_transitions"] == 1


def test_partitioned_attention_provider_cache_rejects_malformed_runtime_owner():
    metrics = H3FlowMetrics()
    runtime = PartitionedStageRuntime(plan=_plan(), metrics=metrics)
    runtime.attention_provider_cache = ()

    try:
        _stage_partitioned_attention_override(runtime, _provider_a, metrics)
    except RuntimeError as exc:
        assert "provider cache is malformed" in str(exc)
    else:
        raise AssertionError("malformed partitioned provider cache was accepted")
