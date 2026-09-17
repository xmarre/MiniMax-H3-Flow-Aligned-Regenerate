from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_transformer import (
    PARTITIONED_ATTENTION_CACHE_KEY,
    _stage_partitioned_attention_override,
)


def _provider_a(*_args, **_kwargs):
    raise AssertionError("provider should not execute in identity regression")


def _provider_b(*_args, **_kwargs):
    raise AssertionError("provider should not execute in identity regression")


def test_partitioned_attention_provider_identity_is_stable_only_within_stage_lifetime():
    metrics = H3FlowMetrics()
    stage = {}

    first = _stage_partitioned_attention_override(stage, _provider_a, metrics)
    repeated = _stage_partitioned_attention_override(stage, _provider_a, metrics)
    assert repeated is first
    assert first._h3_flow_partitioned_previous is _provider_a

    changed_provider = _stage_partitioned_attention_override(stage, _provider_b, metrics)
    changed_provider_repeated = _stage_partitioned_attention_override(stage, _provider_b, metrics)
    assert changed_provider is changed_provider_repeated
    assert changed_provider is not first
    assert changed_provider._h3_flow_partitioned_previous is _provider_b

    fresh_stage = {}
    fresh = _stage_partitioned_attention_override(fresh_stage, _provider_a, metrics)
    assert fresh is not first
    assert fresh._h3_flow_partitioned_previous is _provider_a

    assert len(stage[PARTITIONED_ATTENTION_CACHE_KEY]) == 2
    assert len(fresh_stage[PARTITIONED_ATTENTION_CACHE_KEY]) == 1
    assert metrics.counters["partitioned_attention_provider_creations"] == 3
    assert metrics.counters["partitioned_attention_provider_reuses"] == 2


def test_partitioned_attention_provider_cache_rejects_malformed_stage_state():
    metrics = H3FlowMetrics()
    stage = {PARTITIONED_ATTENTION_CACHE_KEY: ()}

    try:
        _stage_partitioned_attention_override(stage, _provider_a, metrics)
    except RuntimeError as exc:
        assert "provider cache is malformed" in str(exc)
    else:
        raise AssertionError("malformed partitioned provider cache was accepted")
