from types import SimpleNamespace

import torch

from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.runtime import (
    FLOW_BINDING_KEY,
    FLOW_STAGE_KEY,
    FlowBinding,
    _ActiveCapture,
    _bounded_tensor_provenance,
    _conditioning_signature_from_original,
    _nested_tensor_provenance,
    _packed_av_tensor_provenance,
    _trajectory_sample_provenance,
    flow_predict_wrapper,
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


def test_supported_conditioning_wrapper_payload_affects_receipt_without_changing_historical_routing_signature():
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

    assert [receipt["path"] for receipt in first_receipts] == ["conditioning.positive[0].model_conds.context.cond"]
    assert first_receipts[0]["signature"] != second_receipts[0]["signature"]
    # Diagnostic traversal must not change the exact 00625 routing/signature semantics.
    assert _conditioning_signature_from_original(first_payload) == _conditioning_signature_from_original(second_payload)


def test_arbitrary_cond_attribute_is_not_traversed():
    class Arbitrary:
        pass

    value = Arbitrary()
    value.cond = torch.ones(1, 2, 3)

    assert _nested_tensor_provenance({"value": value}, path="conditioning") == []


def test_packed_av_tensor_provenance_is_bounded_stable_and_rng_neutral():
    video = torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2)
    audio = torch.arange(64, dtype=torch.float32).reshape(1, 32, 2, 1)
    packed, shapes = pack_streams((video, audio))
    rng_before = torch.random.get_rng_state().clone()

    first = _packed_av_tensor_provenance(packed, shapes)
    second = _packed_av_tensor_provenance(packed.clone(), shapes)

    assert first == second
    assert first["video_signature"] == _bounded_tensor_provenance(video)
    assert first["audio_signature"] == _bounded_tensor_provenance(audio)
    assert first["packed_shape"] == tuple(packed.shape)
    assert torch.equal(torch.random.get_rng_state(), rng_before)


def test_first_low_model_provenance_records_pre_executor_input_and_output_without_rng_use():
    video = torch.arange(96, dtype=torch.float32).reshape(1, 24, 1, 2, 2)
    audio = torch.arange(64, dtype=torch.float32).reshape(1, 32, 2, 1)
    packed, shapes = pack_streams((video, audio))
    output = packed + 1.0

    class Trajectory:
        def append(self, *_args, **_kwargs):
            return None

    binding = FlowBinding(trajectory=Trajectory())
    binding.active_capture = _ActiveCapture(
        run_id="diagnostic",
        shapes=shapes,
        phases=((0, "single"),),
    )
    guider = SimpleNamespace(
        model_options={
            FLOW_BINDING_KEY: binding,
            "transformer_options": {FLOW_STAGE_KEY: "low"},
        }
    )

    class Executor:
        class_obj = guider

        def __call__(self, x, timestep, model_options, seed):
            assert x is packed
            assert seed == 123
            return output

    rng_before = torch.random.get_rng_state().clone()
    result = flow_predict_wrapper(
        Executor(),
        packed,
        torch.tensor([1.0], dtype=torch.float32),
        guider.model_options,
        seed=123,
    )

    assert result is output
    events = [event for event in binding.metrics.events if event.kind == "first_low_model_provenance"]
    assert len(events) == 1
    fields = events[0].fields
    input_receipt = _packed_av_tensor_provenance(packed, shapes)
    output_receipt = _packed_av_tensor_provenance(output, shapes)
    assert fields["actual"] is True
    assert fields["stage"] == "low"
    assert fields["call_index"] == 0
    assert fields["resolved_seed"] == 123
    assert fields["packed_input_signature"] == input_receipt["packed_signature"]
    assert fields["video_input_signature"] == input_receipt["video_signature"]
    assert fields["audio_input_signature"] == input_receipt["audio_signature"]
    assert fields["packed_output_signature"] == output_receipt["packed_signature"]
    assert fields["video_output_signature"] == output_receipt["video_signature"]
    assert fields["audio_output_signature"] == output_receipt["audio_signature"]
    assert fields["rng_state_consumed"] is False
    assert torch.equal(torch.random.get_rng_state(), rng_before)
