import math

import pytest
import torch

from h3_flow_regenerate.partitioned_prefix import (
    PartitionedExactPrefixPlan,
    dense_partition_oracle,
    merge_partition_attention,
    validate_partitioned_contract,
)


def test_partitioned_plan_roundtrips_canonical_contract():
    plan = PartitionedExactPrefixPlan(
        video_start=7,
        temporal=5,
        prefix_t=2,
        source_grid_h=3,
        source_grid_w=4,
        target_grid_h=5,
        target_grid_w=6,
    )
    assert plan.source_rows == 12
    assert plan.target_rows == 30
    assert plan.prefix_range == (7, 67)
    assert plan.suffix_range == (67, 103)
    assert plan.sequence_rows == 103
    assert plan.prefix_log_key_measure == pytest.approx(math.log(12 / 30))

    contract = plan.to_contract()
    restored = validate_partitioned_contract(contract, sequence_rows=103)
    assert restored == plan
    assert contract["semantic_digest"] == plan.semantic_digest


def test_partitioned_contract_rejects_tampered_measure_or_geometry():
    plan = PartitionedExactPrefixPlan(
        video_start=5,
        temporal=4,
        prefix_t=1,
        source_grid_h=2,
        source_grid_w=3,
        target_grid_h=4,
        target_grid_w=4,
    )
    contract = plan.to_contract()
    contract["prefix_log_key_measure"] = 0.0
    with pytest.raises(ValueError, match="prefix_log_key_measure"):
        validate_partitioned_contract(contract)

    with pytest.raises(ValueError, match="strictly smaller"):
        PartitionedExactPrefixPlan(
            video_start=5,
            temporal=4,
            prefix_t=1,
            source_grid_h=4,
            source_grid_w=4,
            target_grid_h=4,
            target_grid_w=4,
        )


def test_partitioned_lse_merge_matches_explicit_dense_attention():
    generator = torch.Generator(device="cpu").manual_seed(1234)
    q = torch.randn((1, 2, 3, 8), generator=generator, dtype=torch.float64)
    keys = [
        torch.randn((1, 2, 5, 8), generator=generator, dtype=torch.float64),
        torch.randn((1, 2, 7, 8), generator=generator, dtype=torch.float64),
        torch.randn((1, 2, 4, 8), generator=generator, dtype=torch.float64),
    ]
    values = [torch.randn(key.shape, generator=generator, dtype=torch.float64) for key in keys]
    measures = [0.0, math.log(0.4), math.log(1.25)]

    full_out, full_lse, merged_out, merged_lse = dense_partition_oracle(
        q,
        keys,
        values,
        measures,
    )
    assert torch.allclose(merged_out, full_out, rtol=1e-12, atol=1e-12)
    assert torch.allclose(merged_lse, full_lse, rtol=1e-12, atol=1e-12)


def test_physical_measure_cancels_duplicate_carrier_density():
    generator = torch.Generator(device="cpu").manual_seed(7)
    q = torch.randn((1, 1, 3, 6), generator=generator, dtype=torch.float64)
    physical_k = torch.randn((1, 1, 4, 6), generator=generator, dtype=torch.float64)
    physical_v = torch.randn((1, 1, 4, 6), generator=generator, dtype=torch.float64)
    suffix_k = torch.randn((1, 1, 5, 6), generator=generator, dtype=torch.float64)
    suffix_v = torch.randn((1, 1, 5, 6), generator=generator, dtype=torch.float64)

    baseline_out, baseline_lse, _, _ = dense_partition_oracle(
        q,
        [physical_k, suffix_k],
        [physical_v, suffix_v],
        [0.0, 0.0],
    )
    doubled_k = torch.repeat_interleave(physical_k, 2, dim=-2)
    doubled_v = torch.repeat_interleave(physical_v, 2, dim=-2)
    weighted_out, weighted_lse, _, _ = dense_partition_oracle(
        q,
        [doubled_k, suffix_k],
        [doubled_v, suffix_v],
        [-math.log(2.0), 0.0],
    )
    assert torch.allclose(weighted_out, baseline_out, rtol=1e-12, atol=1e-12)
    assert torch.allclose(weighted_lse, baseline_lse, rtol=1e-12, atol=1e-12)


def test_partition_merge_uses_fp32_for_bfloat16_production_inputs():
    outputs = [
        torch.tensor([[[[1.0, 3.0]]]], dtype=torch.bfloat16),
        torch.tensor([[[[5.0, 7.0]]]], dtype=torch.bfloat16),
    ]
    lses = [
        torch.tensor([[[0.0]]], dtype=torch.float32),
        torch.tensor([[[0.0]]], dtype=torch.float32),
    ]
    merged, merged_lse = merge_partition_attention(outputs, lses, [0.0, 0.0])
    assert merged.dtype == torch.bfloat16
    assert merged_lse.dtype == torch.float32
    assert torch.equal(
        merged,
        torch.tensor([[[[3.0, 5.0]]]], dtype=torch.bfloat16),
    )
    assert merged_lse.item() == pytest.approx(math.log(2.0), abs=1e-7)
