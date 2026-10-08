from types import SimpleNamespace

import pytest

from h3_flow_regenerate.partitioned_scheduler import (
    PARTITIONED_SOL_REQUIRED_METADATA,
    SOL_RUNTIME_KEY,
    PartitionedPreflightUnsupported,
    _validate_partitioned_sol_compat,
)


def _guider(metadata):
    transformer_options = {}
    if metadata is not None:
        transformer_options[SOL_RUNTIME_KEY] = metadata
    return SimpleNamespace(model_options={"transformer_options": transformer_options})


def test_partitioned_sol_preflight_accepts_coordinated_runtime_contract():
    _validate_partitioned_sol_compat(_guider(dict(PARTITIONED_SOL_REQUIRED_METADATA)))


def test_partitioned_sol_preflight_allows_native_attention_without_sol():
    assert _validate_partitioned_sol_compat(_guider(None)) is False
    metadata = dict(PARTITIONED_SOL_REQUIRED_METADATA, backend="inherit", attention_ownership="inherit")
    assert _validate_partitioned_sol_compat(_guider(metadata)) is False


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("api", 2),
        ("owner", "other"),
        ("exact", False),
        ("backend", "unknown"),
        ("attention_ownership", "inherit"),
        ("kernel_contract", "old-kernel"),
        ("history_policy", "other-history"),
    ],
)
def test_partitioned_sol_preflight_rejects_incompatible_runtime(field, invalid):
    metadata = dict(PARTITIONED_SOL_REQUIRED_METADATA)
    metadata[field] = invalid
    with pytest.raises(PartitionedPreflightUnsupported, match="compatible Sol-H3 native runtime metadata"):
        _validate_partitioned_sol_compat(_guider(metadata))


def test_target_sink_requires_paired_sol_capability(monkeypatch):
    import sys
    from types import ModuleType

    from h3_flow_regenerate.partitioned_scheduler import _validate_partitioned_sol_sink_measure

    package = ModuleType("sol_h3")
    request = ModuleType("sol_h3.partitioned_request")
    package.partitioned_request = request
    monkeypatch.setitem(sys.modules, "sol_h3", package)
    monkeypatch.setitem(sys.modules, "sol_h3.partitioned_request", request)
    _validate_partitioned_sol_sink_measure("normal")
    with pytest.raises(PartitionedPreflightUnsupported, match="sink-measure API 1"):
        _validate_partitioned_sol_sink_measure("target_query_sink_measure")
    request.PARTITIONED_SINK_MEASURE_API = 1
    _validate_partitioned_sol_sink_measure("target_query_sink_measure")
