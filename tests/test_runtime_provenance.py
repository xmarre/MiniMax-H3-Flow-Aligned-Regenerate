from types import SimpleNamespace

import torch

from h3_flow_regenerate.runtime import (
    _bounded_tensor_provenance,
    _conditioning_signature_from_original,
    _nested_tensor_provenance,
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


def test_trajectory_sample_provenance_bounds_large_runs_with_endpoint_anchors():
    samples = tuple(
        SimpleNamespace(
            coordinate=float(index) / 16.0,
            video_sigma=1.0 - float(index) / 32.0,
            audio_sigma=1.0 - float(index) / 64.0,
            outer_step=index,
            call_index=index,
            phase="model",
            provenance="actual",
            video_x0=torch.full((1, 24, 1, 2, 2), float(index)),
        )
        for index in range(17)
    )

    receipts = _trajectory_sample_provenance(SimpleNamespace(samples=samples), max_samples=5)

    assert [receipt["call_index"] for receipt in receipts] == [0, 4, 8, 12, 16]
    assert len(receipts) == 5


def test_nested_tensor_provenance_reports_paths_without_rng_use():
    first = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4)
    second = torch.arange(8, dtype=torch.int64)
    payload = {"z": [second], "a": {"tensor": first}}
    rng_before = torch.random.get_rng_state().clone()

    receipts = _nested_tensor_provenance(payload, path="conditioning")

    assert [receipt["path"] for receipt in receipts] == [
        "conditioning.a.tensor",
        "conditioning.z[0]",
    ]
    assert receipts[0]["shape"] == (1, 4, 4)
    assert receipts[0]["dtype"] == "torch.float32"
    assert receipts[1]["dtype"] == "torch.int64"
    assert all(len(receipt["signature"]) == 64 for receipt in receipts)
    assert torch.equal(torch.random.get_rng_state(), rng_before)



def test_supported_conditioning_wrapper_payload_affects_receipt_and_signature():
    CondRegular = type("CONDRegular", (), {})
    CondRegular.__module__ = "comfy.conds"

    first = CondRegular()
    first.cond = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    second = CondRegular()
    second.cond = first.cond.clone()
    second.cond[0, 0, 0] += 1.0

    first_payload = {"positive": [{"model_conds": {"context": first}}]}
    second_payload = {"positive": [{"model_conds": {"context": second}}]}

    first_receipts = _nested_tensor_provenance(first_payload, path="conditioning")
    second_receipts = _nested_tensor_provenance(second_payload, path="conditioning")

    assert [receipt["path"] for receipt in first_receipts] == [
        "conditioning.positive[0].model_conds.context.cond"
    ]
    assert first_receipts[0]["signature"] != second_receipts[0]["signature"]
    assert _conditioning_signature_from_original(first_payload) != _conditioning_signature_from_original(second_payload)


def test_arbitrary_cond_attribute_is_not_traversed():
    class Arbitrary:
        pass

    value = Arbitrary()
    value.cond = torch.ones(1, 2, 3)

    assert _nested_tensor_provenance({"value": value}, path="conditioning") == []
