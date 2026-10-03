import copy
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_stage_boundary import high_boundary_contract
from h3_flow_regenerate.residual_evidence import BoundaryWindowEvidence, export_residual_geometry_evidence
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, FLOW_STAGE_KEY, FlowBinding, flow_predict_wrapper


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
def test_native_window_snapshots_include_two_real_context_tokens_and_five_generated_tokens(prefix_t):
    video = torch.arange((prefix_t + 10) * 24 * 4 * 4, dtype=torch.float32).reshape(1, 24, prefix_t + 10, 4, 4)
    evidence = BoundaryWindowEvidence(video[:, :, :prefix_t], prefix_t + 10)
    evidence.capture("final_post_high_internal_clean", video)
    window = evidence.tensors["final_post_high_internal_clean"]
    assert torch.equal(window, video[:, :, prefix_t - 2 : prefix_t + 5])
    assert torch.equal(window[:, :, :2], evidence.tensors["authoritative_prefix"])
    assert window.shape[2] == 7


@pytest.mark.parametrize("prefix_t,temporal", [(3, 18), (12, 16)])
def test_evidence_does_not_invent_decoder_phase_or_pad_missing_generated_context(prefix_t, temporal):
    with pytest.raises(ValueError):
        BoundaryWindowEvidence(torch.zeros(1, 24, prefix_t, 4, 4), temporal)
