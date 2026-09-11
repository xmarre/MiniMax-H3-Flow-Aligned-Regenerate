from __future__ import annotations

import pytest
import torch

from h3_flow_regenerate.geometry import resize_spatial_5d
from h3_flow_regenerate.state_transport import (
    HANDOFF_STATE_POLICY_LEGACY,
    HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1,
    resolve_handoff_state_policy,
    transport_velocity_bicubic_v1,
)


def _video(shape, *, dtype=torch.float32, seed=0):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(shape, generator=generator, dtype=dtype)


def test_policy_resolution_preserves_omitted_legacy_behavior():
    assert resolve_handoff_state_policy(None) == HANDOFF_STATE_POLICY_LEGACY
    assert resolve_handoff_state_policy(HANDOFF_STATE_POLICY_LEGACY) == HANDOFF_STATE_POLICY_LEGACY
    assert (
        resolve_handoff_state_policy(HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1)
        == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1
    )
    with pytest.raises(ValueError, match="unsupported handoff_state_policy"):
        resolve_handoff_state_policy("unknown")
    with pytest.raises(TypeError, match="string or None"):
        resolve_handoff_state_policy(True)


def test_velocity_transport_matches_bicubic_linear_state_lift():
    prefix_t = 2
    source_clean = _video((1, 24, 5, 4, 6), dtype=torch.float32, seed=1)
    displacement = _video(source_clean.shape, dtype=torch.float32, seed=2) * 0.2
    source_state = source_clean + displacement
    target_clean = resize_spatial_5d(source_clean, 8, 10, mode="bicubic")

    transported, metrics = transport_velocity_bicubic_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=prefix_t,
    )
    expected = resize_spatial_5d(source_state, 8, 10, mode="bicubic")

    # Prefix is staging clean and deliberately not source-state transported.
    assert torch.equal(transported[:, :, :prefix_t], target_clean[:, :, :prefix_t])
    torch.testing.assert_close(
        transported[:, :, prefix_t:],
        expected[:, :, prefix_t:],
        rtol=2e-5,
        atol=2e-5,
    )
    assert metrics["handoff_state_policy"] == HANDOFF_STATE_POLICY_VELOCITY_BICUBIC_V1
    assert metrics["state_transport_added_rng"] is False


def test_velocity_transport_preserves_reanchored_displacement_not_endpoint_policy():
    prefix_t = 1
    source_clean = torch.zeros((1, 24, 3, 4, 4), dtype=torch.float32)
    source_state = source_clean.clone()
    source_state[:, :, 1:] = 0.75
    target_clean = torch.zeros((1, 24, 3, 6, 8), dtype=torch.float32)
    target_clean[:, :, 1:] = 2.0

    transported, _ = transport_velocity_bicubic_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=prefix_t,
    )

    # U(X-C) is the constant 0.75 field, so velocity/displacement transport
    # re-anchors that field on C_target and yields 2.75, not a re-noised state.
    torch.testing.assert_close(
        transported[:, :, 1:],
        torch.full_like(transported[:, :, 1:], 2.75),
        rtol=0,
        atol=1e-6,
    )


def test_velocity_transport_preserves_fp64_oracle_and_one_axis_growth():
    prefix_t = 1
    source_clean = _video((1, 24, 3, 4, 6), dtype=torch.float64, seed=3)
    source_state = source_clean + _video(source_clean.shape, dtype=torch.float64, seed=4) * 0.1
    target_clean = torch.nn.functional.interpolate(
        source_clean.permute(0, 2, 1, 3, 4).reshape(3, 24, 4, 6),
        size=(4, 10),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    ).reshape(1, 3, 24, 4, 10).permute(0, 2, 1, 3, 4)

    transported, metrics = transport_velocity_bicubic_v1(
        source_state,
        source_clean,
        target_clean,
        prefix_t=prefix_t,
    )

    assert transported.dtype == torch.float64
    assert metrics["state_transport_compute_dtype"] == "torch.float64"
    expected_displacement = torch.nn.functional.interpolate(
        (source_state[:, :, prefix_t:] - source_clean[:, :, prefix_t:])
        .permute(0, 2, 1, 3, 4)
        .reshape(2, 24, 4, 6),
        size=(4, 10),
        mode="bicubic",
        align_corners=False,
        antialias=False,
    ).reshape(1, 2, 24, 4, 10).permute(0, 2, 1, 3, 4)
    torch.testing.assert_close(
        transported[:, :, prefix_t:] - target_clean[:, :, prefix_t:],
        expected_displacement,
        rtol=1e-12,
        atol=1e-12,
    )


def test_velocity_transport_does_not_mutate_inputs():
    source_clean = _video((1, 24, 4, 4, 4), seed=5)
    source_state = source_clean + 0.1
    target_clean = _video((1, 24, 4, 6, 6), seed=6)
    snapshots = tuple(tensor.clone() for tensor in (source_state, source_clean, target_clean))

    transport_velocity_bicubic_v1(source_state, source_clean, target_clean, prefix_t=2)

    for tensor, snapshot in zip((source_state, source_clean, target_clean), snapshots, strict=True):
        assert torch.equal(tensor, snapshot)


def test_velocity_transport_rejects_invalid_geometry_and_nonfinite_state():
    source_clean = torch.zeros((1, 24, 4, 6, 6), dtype=torch.float32)
    source_state = source_clean.clone()
    target_clean = torch.zeros((1, 24, 4, 4, 8), dtype=torch.float32)
    with pytest.raises(ValueError, match="must not shrink"):
        transport_velocity_bicubic_v1(source_state, source_clean, target_clean, prefix_t=2)

    target_clean = torch.zeros((1, 24, 4, 8, 8), dtype=torch.float32)
    source_state = source_state.clone()
    source_state[0, 0, 2, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="NaN or Inf"):
        transport_velocity_bicubic_v1(source_state, source_clean, target_clean, prefix_t=2)

    with pytest.raises(ValueError, match="nonempty exact prefix"):
        transport_velocity_bicubic_v1(source_clean, source_clean, target_clean, prefix_t=0)
