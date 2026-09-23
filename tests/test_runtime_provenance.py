from types import SimpleNamespace

import torch

from h3_flow_regenerate.runtime import (
    _bounded_tensor_provenance,
    _trajectory_sample_provenance,
)


def test_bounded_tensor_provenance_is_stable_and_rng_neutral():
    torch.manual_seed(1234)
    tensor = torch.arange(256, dtype=torch.float32).reshape(1, 1, 1, 16, 16)
    rng_before = torch.random.get_rng_state().clone()

    first = _bounded_tensor_provenance(tensor)
    second = _bounded_tensor_provenance(tensor.clone())

    assert first == second
    assert len(first) == 64
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_bounded_tensor_provenance_detects_sampled_content_change():
    tensor = torch.zeros(1, 1, 1, 16, 16, dtype=torch.float32)
    baseline = _bounded_tensor_provenance(tensor)

    changed = tensor.clone()
    changed.reshape(-1)[0] = 1.0

    assert _bounded_tensor_provenance(changed) != baseline
    assert _bounded_tensor_provenance(None) is None


def test_trajectory_sample_provenance_records_bounded_anchor_receipts():
    sample = SimpleNamespace(
        coordinate=0.375,
        video_sigma=0.878,
        audio_sigma=0.643,
        outer_step=5,
        call_index=0,
        phase="single",
        provenance="actual",
        video_x0=torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2),
    )
    run = SimpleNamespace(samples=(sample,))

    receipts = _trajectory_sample_provenance(run)

    assert receipts == [
        {
            "coordinate": 0.375,
            "video_sigma": 0.878,
            "audio_sigma": 0.643,
            "outer_step": 5,
            "call_index": 0,
            "phase": "single",
            "provenance": "actual",
            "video_x0_signature": _bounded_tensor_provenance(sample.video_x0),
        }
    ]
    assert _trajectory_sample_provenance(None) == []
