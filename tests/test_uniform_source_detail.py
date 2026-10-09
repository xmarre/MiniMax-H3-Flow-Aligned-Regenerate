from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_UNIFORM_SOURCE_DETAIL_TRANSPORT_KEY,
    resolve_partitioned_uniform_source_detail_transport,
)
from h3_flow_regenerate.uniform_source_detail import (
    UNIFORM_SOURCE_DETAIL_TRANSPORT_POLICY,
    apply_uniform_source_detail_transport,
)

C, T, H, W, PREFIX_T = 8, 26, 36, 40, 12
OBJECT_ROWS = slice(14, 22)


def _lowpass(frame: torch.Tensor) -> torch.Tensor:
    flat = frame.reshape(-1, 1, H, W)
    return F.avg_pool2d(flat, 5, 1, 2, count_include_pad=False).reshape(frame.shape)


def _scene(seed: int = 0):
    """Static detailed background, a moving object, and an upscaler that loses detail.

    The learned rendering keeps only low frequencies of the truth and adds its own
    synthesized detail, like a learned upscaler re-synthesizing texture. The
    synthesized detail belongs to the content: it stays put on the static
    background and moves with the object.
    """

    generator = torch.Generator().manual_seed(seed)
    detail = torch.randn(1, C, H, W, generator=generator)
    base = _lowpass(torch.randn(1, C, H, W, generator=generator)) * 3.0
    obj = torch.randn(1, C, 8, 8, generator=generator) * 2.0
    background_synthesized = torch.randn(1, C, H, W, generator=generator) * 0.8
    object_synthesized = torch.randn(1, C, 8, 8, generator=generator) * 0.8
    truth = torch.empty(1, C, T, H, W)
    learned = torch.empty(1, C, T, H, W)
    for t in range(T):
        frame = base + detail
        frame[:, :, OBJECT_ROWS, t : t + 8] = obj
        truth[:, :, t] = frame
        synthesized = background_synthesized.clone()
        synthesized[:, :, OBJECT_ROWS, t : t + 8] = object_synthesized
        noise = 0.05 * torch.randn(1, C, H, W, generator=generator)
        learned[:, :, t] = _lowpass(frame) + synthesized + noise
    return truth, learned


def _rms(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt())


def test_static_detail_is_recovered_without_touching_prefix_or_moving_content():
    truth, learned = _scene()
    exact_prefix = truth[:, :, :PREFIX_T].clone()

    transported, receipt = apply_uniform_source_detail_transport(learned, exact_prefix, prefix_t=PREFIX_T)

    assert receipt["policy"] == UNIFORM_SOURCE_DETAIL_TRANSPORT_POLICY
    assert receipt["applied"] is True
    assert receipt["holdout_error_ratio"] < 0.5
    assert torch.equal(transported[:, :, :PREFIX_T], learned[:, :, :PREFIX_T])

    static = torch.ones(H, W, dtype=torch.bool)
    static[OBJECT_ROWS.start - 2 : OBJECT_ROWS.stop + 2] = False
    for t in (PREFIX_T, PREFIX_T + 6, T - 1):
        before = _rms((learned[:, :, t] - truth[:, :, t])[..., static])
        after = _rms((transported[:, :, t] - truth[:, :, t])[..., static])
        assert after < 0.1 * before
        # The object has moved since the anchor frame; its current location and the
        # area it vacated must not receive the anchor frame's detail as a ghost.
        current = (slice(None), slice(None), OBJECT_ROWS, slice(t, t + 8))
        assert _rms(transported[:, :, t][current] - truth[:, :, t][current]) <= 1.02 * _rms(
            learned[:, :, t][current] - truth[:, :, t][current]
        )
        # Ghost amplitude: what the transport adds on the moving object stays far below
        # the correction it applies to static content.
        added = transported[:, :, t] - learned[:, :, t]
        assert _rms(added[current]) < 0.1 * _rms(added[..., static])
    vacated = (slice(None), slice(None), OBJECT_ROWS, slice(PREFIX_T - 1, PREFIX_T + 7))
    t = T - 1
    assert _rms(transported[:, :, t][vacated] - truth[:, :, t][vacated]) <= 1.02 * _rms(
        learned[:, :, t][vacated] - truth[:, :, t][vacated]
    )


def test_transport_is_skipped_when_the_error_does_not_persist_in_held_out_prefix_frames():
    generator = torch.Generator().manual_seed(3)
    learned = torch.randn(1, C, T, H, W, generator=generator)
    # The exact prefix differs from the learned rendering by independent noise per
    # frame, so the anchor frame's error predicts nothing about later frames.
    exact_prefix = learned[:, :, :PREFIX_T] + torch.randn(1, C, PREFIX_T, H, W, generator=generator)

    transported, receipt = apply_uniform_source_detail_transport(learned, exact_prefix, prefix_t=PREFIX_T)

    if receipt["applied"]:
        assert receipt["holdout_error_ratio"] < 1.0
        assert receipt["delta_rms"] < 0.05 * receipt["anchor_residual_rms"]
    else:
        assert receipt["reason"] == "holdout_not_improved"
        assert transported is learned


def test_short_prefix_is_not_calibrated():
    truth, learned = _scene()
    transported, receipt = apply_uniform_source_detail_transport(learned, truth[:, :, :3], prefix_t=3)
    assert receipt["applied"] is False
    assert receipt["reason"] == "prefix_too_short_for_calibration"
    assert transported is learned


def test_geometry_and_dtype_contract():
    truth, learned = _scene()
    with pytest.raises(ValueError, match="prefix geometry"):
        apply_uniform_source_detail_transport(learned, truth[:, :, : PREFIX_T - 1], prefix_t=PREFIX_T)
    with pytest.raises(ValueError, match="valid prefix boundary"):
        apply_uniform_source_detail_transport(learned, truth, prefix_t=T)
    half, receipt = apply_uniform_source_detail_transport(
        learned.to(torch.bfloat16), truth[:, :, :PREFIX_T].to(torch.bfloat16), prefix_t=PREFIX_T
    )
    assert half.dtype == torch.bfloat16
    assert receipt["applied"] is True


def test_option_defaults_off_and_requires_a_boolean_opt_in():
    assert resolve_partitioned_uniform_source_detail_transport({}) is False
    assert (
        resolve_partitioned_uniform_source_detail_transport({PARTITIONED_UNIFORM_SOURCE_DETAIL_TRANSPORT_KEY: False})
        is False
    )
    assert resolve_partitioned_uniform_source_detail_transport({PARTITIONED_UNIFORM_SOURCE_DETAIL_TRANSPORT_KEY: True})
    for value in (1, 0, None, "true"):
        with pytest.raises(ValueError, match="boolean"):
            resolve_partitioned_uniform_source_detail_transport(
                {PARTITIONED_UNIFORM_SOURCE_DETAIL_TRANSPORT_KEY: value}
            )


def test_prefix_holdout_does_not_establish_safety_for_motion_lost_by_projection():
    # A one-pixel translation changes this fine pattern, but neither phase is
    # visible in a 2x2 projection. The prefix-only calibration cannot detect
    # this suffix motion; accepting its fit must not imply rendered safety.
    learned = 0.001 * torch.randn(1, C, T, H, W, generator=torch.Generator().manual_seed(0))
    y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    detail = ((x + y) % 2 * 2 - 1).float()[None, None].expand(1, C, -1, -1)
    shifted = torch.roll(detail, shifts=1, dims=-1)
    assert torch.count_nonzero(F.avg_pool2d(detail, 2, 2)) == 0
    assert torch.count_nonzero(F.avg_pool2d(shifted, 2, 2)) == 0
    exact = learned[:, :, :PREFIX_T] + detail.unsqueeze(2)
    transported, receipt = apply_uniform_source_detail_transport(learned, exact, prefix_t=PREFIX_T)
    truth = learned[:, :, PREFIX_T:] + shifted.unsqueeze(2)

    assert receipt["applied"] and receipt["holdout_error_ratio"] < 0.001
    assert _rms(transported[:, :, PREFIX_T:] - truth) > 1.5 * _rms(learned[:, :, PREFIX_T:] - truth)
    assert resolve_partitioned_uniform_source_detail_transport({}) is False


@pytest.mark.parametrize("prefix_t", [4, 5, 6])
def test_minimal_prefixes_never_fail(prefix_t):
    truth, learned = _scene()
    transported, receipt = apply_uniform_source_detail_transport(learned, truth[:, :, :prefix_t], prefix_t=prefix_t)
    assert torch.equal(transported[:, :, :prefix_t], learned[:, :, :prefix_t])
    if not receipt["applied"]:
        assert transported is learned
