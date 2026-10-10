"""Opt-in paired source-grid low/probe experiment using a frozen caller input.

The normal A output remains the production handoff. B and A' are shadow
low/probe lifetimes with independent Flow trajectory/metrics owners. An A-A'
reproducibility test is required before interpreting the A-B difference.
"""
from __future__ import annotations

import copy
import random
from contextlib import contextmanager
from dataclasses import dataclass

import torch

from .geometry import pack_streams, unpack_streams
from .partitioned_stage import tensor_sha256

LOW_SAMPLER_AB_KEY = "h3_flow_low_sampler_aba_v1"
POLICY = "h3_frozen_low_source_aba_v1"


@dataclass(frozen=True)
class FrozenRNG:
    python: object
    numpy: object
    cpu: torch.Tensor
    cuda: tuple[torch.Tensor, ...] | None


def capture_rng() -> FrozenRNG:
    import numpy as np

    cuda = tuple(t.clone() for t in torch.cuda.get_rng_state_all()) if torch.cuda.is_initialized() else None
    return FrozenRNG(copy.deepcopy(random.getstate()), copy.deepcopy(np.random.get_state()), torch.get_rng_state().clone(), cuda)


def restore_rng(saved: FrozenRNG) -> None:
    import numpy as np

    random.setstate(copy.deepcopy(saved.python))
    np.random.set_state(copy.deepcopy(saved.numpy))
    torch.set_rng_state(saved.cpu.clone())
    if saved.cuda is not None:
        if not torch.cuda.is_initialized() or len(saved.cuda) != torch.cuda.device_count():
            raise RuntimeError("frozen low A/B CUDA RNG device topology changed")
        torch.cuda.set_rng_state_all([v.clone() for v in saved.cuda])


@contextmanager
def frozen_rng(initial: FrozenRNG):
    """Rewind Python/NumPy/torch RNGs for one arm and restore caller RNGs."""
    after = capture_rng()
    try:
        restore_rng(initial)
        yield
    finally:
        restore_rng(after)


def counterfactual_low_inputs(
    original: torch.Tensor,
    low_shapes: list[tuple[int, ...]],
    original_prefix: torch.Tensor,
    alternative_prefix: torch.Tensor,
    prefix_t: int,
) -> torch.Tensor:
    video, audio = unpack_streams(original, low_shapes)
    if (
        video.ndim != 5
        or video.shape[2] <= prefix_t
        or prefix_t < 2
        or (prefix_t - 2) % 5
        or (video.shape[2] - 2) % 5
        or video[:, :, :prefix_t].shape != original_prefix.shape
        or alternative_prefix.shape != original_prefix.shape
        or original_prefix.dtype != alternative_prefix.dtype
        or not torch.equal(video[:, :, :prefix_t], original_prefix.to(device=video.device))
    ):
        raise ValueError("frozen low source A/B prefix or native phase does not match actual sampler input")
    candidate_video = video.clone()
    candidate_video[:, :, :prefix_t] = alternative_prefix.to(device=video.device)
    output, _ = pack_streams((candidate_video, audio.clone()))
    if not torch.equal(candidate_video[:, :, prefix_t:], video[:, :, prefix_t:]):
        raise RuntimeError("frozen low source A/B modified generated suffix input")
    if not torch.equal(unpack_streams(output, low_shapes)[1], audio):
        raise RuntimeError("frozen low source A/B modified carried audio input")
    return output


def difference(a: torch.Tensor, b: torch.Tensor) -> dict:
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError("paired low predictions have incompatible domains")
    delta = a.detach().to(device="cpu", dtype=torch.float32) - b.detach().to(device="cpu", dtype=torch.float32)
    return {
        "rms": float(delta.square().mean().sqrt()),
        "abs_max": float(delta.abs().max()),
        "exact": bool(torch.equal(a, b)),
        "a_sha256": tensor_sha256(a),
        "b_sha256": tensor_sha256(b),
    }


def compare_aba(
    baseline_clean: torch.Tensor,
    changed_clean: torch.Tensor,
    repeat_clean: torch.Tensor,
    baseline_first: torch.Tensor,
    changed_first: torch.Tensor,
    repeat_first: torch.Tensor,
    *,
    prefix_t: int,
    tolerance: float = 1e-5,
) -> dict:
    """Reject divergent A' replay before treating B as a conditioning result."""
    originals = [baseline_clean, changed_clean, repeat_clean]
    if any(t.ndim != 5 or t.shape != originals[0].shape for t in originals):
        raise ValueError("A/B/A' low clean videos must share native [B,C,T,H,W] geometry")
    if not 0 < prefix_t < originals[0].shape[2]:
        raise ValueError("invalid low generated suffix split")
    clean_a_repeat = difference(baseline_clean[:, :, prefix_t:], repeat_clean[:, :, prefix_t:])
    clean_a_b = difference(baseline_clean[:, :, prefix_t:], changed_clean[:, :, prefix_t:])
    first_a_repeat = difference(baseline_first, repeat_first)
    first_a_b = difference(baseline_first, changed_first)
    # A reproduction must match both first actual prediction and final
    # low/probe clean suffix; one small scalar is not sufficient.
    invariant = (
        clean_a_repeat["abs_max"] <= tolerance
        and first_a_repeat["abs_max"] <= tolerance
    )
    return {
        "policy": POLICY,
        "repeat_a_clean_suffix": clean_a_repeat,
        "repeat_a_first_actual_prediction": first_a_repeat,
        "changed_b_clean_suffix": clean_a_b,
        "changed_b_first_actual_prediction": first_a_b,
        "reproduction_abs_tolerance": tolerance,
        "reproduction_verified": invariant,
        "prefix_is_only_altered_low_input": True,
        "causal_low_input_effect_qualified": bool(invariant),
        "qualification_scope": "within this one frozen low/probe invocation, not camera motion or final render",
        "model_execution_history_strictly_isolated": False,
        "a_b_a_order_controls_for_observable_model_state_leakage": True,
    }
