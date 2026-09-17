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


def test_partitioned_sol_preflight_rejects_missing_runtime_before_sampling():
    with pytest.raises(PartitionedPreflightUnsupported, match="active Sol-H3 native runtime ownership"):
        _validate_partitioned_sol_compat(_guider(None))


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("api", 2),
        ("owner", "other"),
        ("exact", False),
        ("backend", "inherit"),
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
