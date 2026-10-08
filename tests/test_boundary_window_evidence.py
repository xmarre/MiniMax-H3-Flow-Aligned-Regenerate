import copy
import hashlib
import json
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_stage_boundary import high_boundary_contract
from h3_flow_regenerate.residual_evidence import BoundaryWindowEvidence, export_residual_geometry_evidence
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FLOW_STAGE_KEY, FlowBinding, flow_predict_wrapper


def test_full_stage_capture_keeps_late_changes_and_owns_cpu_storage():
    video = torch.zeros(1, 24, 22, 4, 4)
    evidence = BoundaryWindowEvidence(video[:, :, :12], 22, capture_full_video=True)
    video[:, :, 21] = 3.0
    evidence.observe_prediction(video, point="before_flow", call_index=0, sigma=0.8, actual=True)
    assert torch.count_nonzero(evidence.tensors["first_high_before_flow"]) == 0
    full = evidence.tensors["first_high_before_flow_full"]
    assert torch.equal(full, video)
    video.zero_()
    assert torch.all(full[:, :, 21] == 3.0)
    assert full.device.type == "cpu"
    assert full.untyped_storage().nbytes() == full.numel() * full.element_size()
    evidence.close()
    assert evidence.tensors == {}


def test_full_stage_capture_checks_budget_before_copying_or_recording_a_stage():
    video = torch.zeros(1, 24, 22, 4, 4)
    # Enough for the authoritative prefix window, not a full stage.
    evidence = BoundaryWindowEvidence(video[:, :, :12], 22, capture_full_video=True, max_bytes=24000)
    names = set(evidence.tensors)
    with pytest.raises(RuntimeError, match="CPU byte budget"):
        evidence.capture("provider_native_clean", video)
    assert set(evidence.tensors) == names


def test_full_stage_capture_rejects_a_different_timeline():
    video = torch.zeros(1, 24, 22, 4, 4)
    evidence = BoundaryWindowEvidence(video[:, :, :12], 22, capture_full_video=True)
    with pytest.raises(RuntimeError, match="declared timeline"):
        evidence.capture("provider_native_clean", video[:, :, :-1])
    assert set(evidence.tensors) == {"authoritative_prefix", "authoritative_prefix_full"}


def test_high_call_windows_keep_forecasts_and_pair_provenance_without_full_copies():
    video = torch.zeros(1, 24, 22, 4, 4)
    evidence = BoundaryWindowEvidence(video[:, :, :12], 22, capture_full_video=True)
    for index, actual in enumerate((True, False, True)):
        video[:, :, 12:] = index
        before = video.clone()
        evidence.observe_prediction(video, point="before_flow", call_index=index, sigma=0.8, actual=actual)
        assert torch.equal(video, before)
        video[:, :, 12:] += 0.1
        after = video.clone()
        evidence.observe_prediction(video, point="after_flow", call_index=index, sigma=0.8, actual=actual)
        assert torch.equal(video, after)
    calls = evidence.plan["high_prediction_trace"]["calls"]
    assert [call["actual"] for call in calls] == [True, False, True]
    assert calls[0]["before_flow"] == "first_high_before_flow"
    assert calls[0]["after_flow"] == "first_high_after_flow"
    assert evidence.pending_high_prediction is None
    before_window = evidence.tensors[calls[1]["before_flow"]]
    after_window = evidence.tensors[calls[1]["after_flow"]]
    assert before_window.shape[2] == after_window.shape[2] == 7
    assert torch.all(before_window[:, :, 2:] == 1)
    torch.testing.assert_close(after_window[:, :, 2:], torch.full_like(after_window[:, :, 2:], 1.1))
    video.zero_()
    assert torch.all(before_window[:, :, 2:] == 1)
    assert not any(name.startswith("high_prediction_") and name.endswith("_full") for name in evidence.tensors)
    assert evidence.cpu_tensor_bytes == sum(t.numel() * t.element_size() for t in evidence.tensors.values())
    evidence.observe_prediction(video, point="before_flow", call_index=16, sigma=0.2, actual=True)
    assert len(calls) == 3


def test_optional_high_windows_reserve_final_snapshot_and_skip_budget_exhaustion():
    video = torch.zeros(1, 24, 22, 4, 4)
    evidence = BoundaryWindowEvidence(video[:, :, :12], 22, capture_full_video=True)
    for point in ("before_flow", "after_flow"):
        evidence.observe_prediction(video, point=point, call_index=0, sigma=0.8, actual=True)
    window_bytes = 7 * 24 * 4 * 4 * 4
    final_bytes = video.numel() * video.element_size() + window_bytes
    evidence.max_bytes = evidence.cpu_tensor_bytes + final_bytes
    names = set(evidence.tensors)
    for point in ("before_flow", "after_flow"):
        evidence.observe_prediction(video, point=point, call_index=1, sigma=0.7, actual=False)
    assert set(evidence.tensors) == names
    assert evidence.plan["high_prediction_trace"]["omitted_call_indices"] == [1]
    evidence.capture("final_post_high_internal_clean", video)
    assert evidence.cpu_tensor_bytes == evidence.max_bytes


@pytest.mark.parametrize("change", ["sigma", "actual", "index"])
def test_high_call_window_pair_rejects_changed_provenance(change):
    video = torch.zeros(1, 24, 22, 4, 4)
    evidence = BoundaryWindowEvidence(video[:, :, :12], 22, capture_full_video=True)
    evidence.observe_prediction(video, point="before_flow", call_index=0, sigma=0.8, actual=True)
    args = {"call_index": 0, "sigma": 0.8, "actual": True}
    args[change if change != "index" else "call_index"] = {"sigma": 0.7, "actual": False, "index": 1}[change]
    with pytest.raises(RuntimeError, match="pair changed provenance"):
        evidence.observe_prediction(video, point="after_flow", **args)


@pytest.mark.parametrize("measure", [False, True])
def test_rejected_registration_captures_first_actual_window_without_changing_predictions(
    tmp_path, monkeypatch, measure
):
    generator = torch.Generator().manual_seed(1052)
    video = torch.randn(1, 24, 22, 8, 10, generator=generator)
    audio = torch.randn(1, 32, 2, 40, generator=generator)
    original_video, original_audio = video.clone(), audio.clone()
    packed, shapes = pack_streams((video, audio))
    sampler_input = packed + 0.25
    evidence = BoundaryWindowEvidence(video[:, :, :12], 22) if measure else None
    if evidence is not None:
        evidence.capture("pre_high_exact_restored", video)
    binding = FlowBinding()
    guider = SimpleNamespace(
        model_options={FLOW_BINDING_KEY: binding}, inner_model=SimpleNamespace(latent_shapes=list(shapes))
    )

    class Executor:
        class_obj = guider
        calls = 0

        def __call__(self, x, timestep, model_options, seed):
            del x, timestep, model_options, seed
            self.calls += 1
            return packed + self.calls / 10

    executor = Executor()
    with high_boundary_contract(binding, video[:, :, :12], shapes, measure=measure, window_evidence=evidence):
        for index, actual in enumerate((False, True, False, True)):
            options = {"transformer_options": {FLOW_STAGE_KEY: "high", "spectrum_h3_actual": actual}}
            result = flow_predict_wrapper(
                executor, sampler_input, torch.tensor([0.8 - index / 10]), model_options=options, seed=1
            )
            assert torch.equal(result, packed + (index + 1) / 10)
        trace = binding.high_boundary_trace
    assert executor.calls == 4
    assert torch.equal(video, original_video) and torch.equal(audio, original_audio)
    if evidence is None:
        assert trace is None
        return

    assert trace.window_evidence is None
    assert evidence.first_high_call_index == 1
    assert evidence.first_high_sigma == pytest.approx(0.7)
    assert copy.deepcopy(evidence) is evidence
    start, stop = 10, 17
    expected_prediction = original_video[:, :, start:stop] + 0.2
    assert torch.equal(evidence.tensors["first_high_before_flow"], expected_prediction)
    assert torch.equal(evidence.tensors["first_high_after_flow"], expected_prediction)
    input_video, _ = unpack_streams(sampler_input, shapes)
    assert torch.equal(evidence.tensors["first_high_sampler_input"], input_video[:, :, start:stop])
    assert torch.equal(evidence.tensors["authoritative_prefix"], original_video[:, :, start:12])
    assert not torch.equal(
        evidence.tensors["first_high_before_flow"][:, :, :2], evidence.tensors["authoritative_prefix"]
    )
    for tensor in evidence.tensors.values():
        assert tensor.device.type == "cpu"
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    # Reused production buffers cannot change an already-captured actual prediction.
    packed.add_(3)
    assert torch.equal(evidence.tensors["first_high_before_flow"], expected_prediction)
    monkeypatch.setitem(sys.modules, "folder_paths", SimpleNamespace(get_output_directory=lambda: str(tmp_path)))
    receipt = export_residual_geometry_evidence(
        evidence.tensors,
        session_id="session",
        chunk_id="2",
        seed=1,
        sigma=0.8,
        evidence_kind="h3_flow_native_boundary_decoder_window_evidence",
        metadata={
            "window": evidence.plan,
            "registration_result": "rejected",
            "registration_acceptance_required": False,
        },
    )
    bundle = tmp_path / receipt["bundle"]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["metadata"]["registration_result"] == "rejected"
    assert manifest["kind"] == "h3_flow_native_boundary_decoder_window_evidence"
    assert manifest["metadata"]["window"]["first_retained_local_frame"] == 5
    assert receipt["tensor_count"] == 5
    for name, details in manifest["tensor_bytes"].items():
        raw = (bundle / details["file"]).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == details["sha256"]
        decoded = torch.frombuffer(bytearray(raw), dtype=torch.float32).reshape(details["shape"])
        assert torch.equal(decoded, evidence.tensors[name])
    evidence.close()
    assert evidence.tensors == {}


@pytest.mark.parametrize("prefix_t", [2, 7, 12])
@pytest.mark.parametrize("generated_t", [5, 10])
def test_native_window_snapshots_include_two_real_context_tokens_and_five_generated_tokens(prefix_t, generated_t):
    temporal = prefix_t + generated_t
    video = torch.arange(temporal * 24 * 4 * 4, dtype=torch.float32).reshape(1, 24, temporal, 4, 4)
    evidence = BoundaryWindowEvidence(video[:, :, :prefix_t], temporal)
    evidence.capture("final_post_high_internal_clean", video)
    window = evidence.tensors["final_post_high_internal_clean"]
    assert torch.equal(window, video[:, :, prefix_t - 2 : prefix_t + 5])
    assert torch.equal(window[:, :, :2], evidence.tensors["authoritative_prefix"])
    assert window.shape[2] == 7


@pytest.mark.parametrize("prefix_t", [2, 7, 12])
def test_saved_window_maps_first_retained_frames_to_the_native_continuous_decoder(prefix_t):
    decode = runpy.run_path(str(Path(__file__).with_name("test_decode_context.py")))["native_temporal_decoder"]()
    generator = torch.Generator().manual_seed(1052)
    video = torch.randn(1, 24, prefix_t + 10, 4, 4, generator=generator)
    evidence = BoundaryWindowEvidence(video[:, :, :prefix_t], video.shape[2])
    evidence.capture("pre_high_exact_restored", video)
    window_pixels = decode(evidence.tensors["pre_high_exact_restored"])
    timeline_pixels = decode(video)
    trim = evidence.plan["decoded_trim_frames"]
    local_start = evidence.plan["first_retained_local_frame"]
    # These twelve retained frames precede the next decoder-window blend.
    assert torch.equal(window_pixels[:, :, local_start:17], timeline_pixels[:, :, trim : trim + 12])


@pytest.mark.parametrize("prefix_t,temporal", [(3, 18), (12, 16)])
def test_evidence_does_not_invent_decoder_phase_or_pad_missing_generated_context(prefix_t, temporal):
    with pytest.raises(ValueError):
        BoundaryWindowEvidence(torch.zeros(1, 24, prefix_t, 4, 4), temporal)
