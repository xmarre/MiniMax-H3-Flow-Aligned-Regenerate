import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.attention import AttentionConfig, layout_summary, make_attention_override, video_local_mask
from h3_flow_regenerate.comfy_compat import patch_flow_model, reconfigure_binding
from h3_flow_regenerate.contracts import H3FlowTrajectory
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.handoff import ProgressiveHandoffConfig
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.reference import apply_reference_budget
from h3_flow_regenerate.runtime import FlowBinding, _run_progressive, _sampler_phases


def layout():
    # text=2, ref=2, audio=4, video=2*2*3=12
    return SimpleNamespace(
        segments=[(0, 2, "text"), (2, 4, "ref_img"), (4, 8, "audio"), (8, 20, "video")],
        signature=(2, 2, 4, 6, 2),
        seq_len=20,
    )


def test_sparse_mask_keeps_nonvideo_global_and_video_spatiotemporal_local():
    mask = video_local_mask(layout(), torch.tensor([0, 8]), radius=1, device=torch.device("cpu"))
    assert mask[0].all()
    assert mask[1, :8].all()
    # Same spatial location in both frames remains visible.
    assert mask[1, 8]
    assert mask[1, 14]
    # Far spatial corner is hidden.
    assert not mask[1, 13]


def test_layout_summary_counts_packed_modalities():
    summary = layout_summary(layout())
    assert summary["text_rows"] == 2
    assert summary["reference_rows"] == 2
    assert summary["audio_rows"] == 4
    assert summary["video_rows"] == 12
    assert summary["sequence_rows"] == 20


def test_attention_config_is_guarded():
    with pytest.raises(ValueError):
        AttentionConfig(mode="experimental_sparse", layers=(-1,))


def test_attention_diagnostic_preserves_existing_override():
    metrics = H3FlowMetrics()
    calls = []

    def previous(backend, q, k, v, heads, mask=None, skip_reshape=False, **kwargs):
        calls.append((mask, skip_reshape))
        return backend(q, k, v, heads, mask=mask, skip_reshape=skip_reshape, **kwargs)

    config = AttentionConfig(mode="diagnostic", layers=(0,))
    override = make_attention_override(config, metrics, previous_override=previous)
    q = torch.randn(1, 2, 20, 4)

    def backend(q, _k, _v, heads, mask=None, skip_reshape=False, **_kwargs):
        assert mask is None
        assert skip_reshape
        return q.transpose(1, 2).reshape(1, q.shape[2], heads * q.shape[-1])

    result = override(
        backend,
        q,
        q,
        q,
        2,
        skip_reshape=True,
        transformer_options={"h3_flow_attention_context": {"layout": layout(), "layer": 0}},
    )
    assert result.shape == (1, 20, 8)
    assert calls == [(None, True)]
    assert metrics.events[-1].kind == "attention_diagnostic"


def test_sparse_backend_error_falls_back_to_full_attention():
    metrics = H3FlowMetrics()
    config = AttentionConfig(
        mode="experimental_sparse",
        layers=(0,),
        global_heads=0,
        query_chunk=4,
    )
    override = make_attention_override(config, metrics)
    q = torch.randn(1, 2, 20, 4)
    calls = []

    def backend(q, _k, _v, heads, mask=None, **_kwargs):
        calls.append(mask)
        if mask is not None:
            raise RuntimeError("mask unavailable")
        return q.transpose(1, 2).reshape(1, q.shape[2], heads * q.shape[-1])

    result = override(
        backend,
        q,
        q,
        q,
        2,
        skip_reshape=True,
        transformer_options={"h3_flow_attention_context": {"layout": layout(), "layer": 0}},
    )
    assert result.shape == (1, 20, 8)
    assert calls[-1] is None
    assert metrics.events[-1].fields["reason"] == "backend_rejected_sparse_mask"


def test_binding_reconfiguration_preserves_shared_metrics():
    original = FlowBinding(trajectory=H3FlowTrajectory())
    updated = reconfigure_binding(original, capture_forecasts=True)
    assert updated.trajectory is original.trajectory
    assert updated.metrics is original.metrics
    assert updated.capture_forecasts


class FakeH3Base:
    def __init__(self):
        self.diffusion_model = SimpleNamespace(
            patch_size=(1, 2, 2),
            latents_dim=24,
            audio_latents_dim=32,
            sigma_shift_video=12.0,
            sigma_shift_audio=3.0,
            blocks=[object() for _ in range(50)],
        )
        self.latent_shapes = None

    def process_latent_in(self, value):
        return value


class MiniMaxH3:
    def __init__(self):
        self.diffusion_model = FakeH3Base().diffusion_model

    def process_latent_in(self, value):
        return value


class FakePatcher:
    def __init__(self):
        self.model = MiniMaxH3()
        self.model_options = {
            "transformer_options": {
                "wrappers": {
                    "outer_sample": {"spectrum": [object()]},
                    "predict_noise": {"spectrum": [object()]},
                }
            }
        }
        self.wrappers = self.model_options["transformer_options"]["wrappers"]

    def clone(self):
        cloned = FakePatcher()
        cloned.model = self.model
        cloned.model_options = {
            **self.model_options,
            "transformer_options": dict(self.model_options["transformer_options"]),
        }
        cloned.wrappers = {key: dict(value) for key, value in self.wrappers.items()}
        cloned.model_options["transformer_options"]["wrappers"] = cloned.wrappers
        return cloned

    def remove_wrappers_with_key(self, wrapper_type, key):
        self.wrappers.get(wrapper_type, {}).pop(key, None)

    def set_model_patch_replace(self, patch, name, block_name, number):
        patches = self.model_options["transformer_options"].setdefault("patches_replace", {})
        patches.setdefault(name, {})[(block_name, number)] = patch


def test_model_patch_preserves_existing_wrapper_order_and_binding(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)
    trajectory = H3FlowTrajectory()
    patched, binding = patch_flow_model(FakePatcher(), trajectory=trajectory)
    assert binding.trajectory is trajectory
    assert next(iter(patched.wrappers["outer_sample"])) == "h3_flow_regenerate.outer.v1"
    assert "spectrum" in patched.wrappers["outer_sample"]
    assert next(iter(patched.wrappers["predict_noise"])) == "spectrum"
    assert list(patched.wrappers["predict_noise"])[-1] == "h3_flow_regenerate.predict.v1"
    assert ("double_block", 0) in patched.model_options["transformer_options"]["patches_replace"]["dit"]


def test_patch_can_explicitly_disable_inherited_forecast_capture(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)

    first, first_binding = patch_flow_model(FakePatcher(), capture_forecasts=True)
    assert first_binding.capture_forecasts
    second, second_binding = patch_flow_model(first, capture_forecasts=False)
    assert not second_binding.capture_forecasts
    _, inherited = patch_flow_model(second)
    assert not inherited.capture_forecasts


def test_attention_layers_are_validated_against_loaded_block_count(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)

    model = FakePatcher()
    model.model.diffusion_model.blocks.append(object())
    patched, _ = patch_flow_model(
        model,
        attention=AttentionConfig(mode="diagnostic", layers=(50,)),
    )
    assert ("double_block", 50) in patched.model_options["transformer_options"]["patches_replace"]["dit"]


def test_progressive_runtime_uses_three_fresh_downstream_calls_and_preserves_audio(monkeypatch):
    class KSampler:
        def __init__(self, function):
            self.sampler_function = function
            self.extra_options = {}

    fake_samplers = ModuleType("comfy.samplers")
    fake_samplers.KSAMPLER = KSampler
    fake_comfy = ModuleType("comfy")
    fake_comfy.samplers = fake_samplers
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake_samplers)

    source_video = torch.randn(1, 24, 1, 4, 4)
    source_audio = torch.randn(1, 32, 2, 5)
    source_raw, source_shapes = __import__("h3_flow_regenerate.geometry", fromlist=["pack_streams"]).pack_streams(
        (source_video, source_audio)
    )
    source_x0 = source_raw.clone()
    sigmas = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])

    def native():
        pass

    native.__name__ = "sample_sa_solver_pece"
    sampler = SimpleNamespace(sampler_function=native, extra_options={})
    original_cond = {"cross_attn": torch.zeros(1, 2, 4)}
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=MiniMaxH3()),
        original_conds={"positive": [original_cond]},
        conds={"positive": [original_cond.copy()]},
    )
    calls = []

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent, call_sampler, call_sigmas, *args, latent_shapes):
            calls.append(
                (
                    call_sampler.sampler_function.__name__,
                    list(latent_shapes),
                    dict(guider.model_options["transformer_options"]),
                    call_sigmas.clone(),
                    "stage_mutated" in guider.conds["positive"][0],
                )
            )
            guider.conds["positive"][0]["stage_mutated"] = True
            if call_sampler.sampler_function.__name__ == "_h3_flow_exact_probe":
                return source_x0
            if len(calls) == 1:
                return source_raw / (1.0 - call_sigmas[-1])
            return noise

    binding = FlowBinding(trajectory=H3FlowTrajectory(), guidance=GuidanceConfig(mode="off"))
    mutable_shapes = list(source_shapes)
    result = _run_progressive(
        Executor(),
        guider,
        binding,
        ProgressiveHandoffConfig(8, 6, handoff_coordinate=0.3),
        source_raw,
        torch.zeros_like(source_raw),
        sampler,
        sigmas,
        None,
        None,
        True,
        7,
        mutable_shapes,
    )
    assert [call[0] for call in calls] == ["sample_sa_solver_pece", "_h3_flow_exact_probe", "sample_sa_solver_pece"]
    assert calls[-1][2]["h3_refinement"]["min_actual_prefix_steps"] == 1
    assert [call[4] for call in calls] == [False, False, False]
    assert mutable_shapes[0] == (1, 24, 1, 8, 6)
    _, result_audio = __import__("h3_flow_regenerate.geometry", fromlist=["unpack_streams"]).unpack_streams(
        result * float(calls[-1][3][0]), mutable_shapes
    )
    assert torch.allclose(result_audio, source_audio)


def test_reference_native_parity_and_direct_only_decoupling():
    cross = torch.randn(1, 9, 12)
    ref = torch.randn(1, 24, 2, 16, 16)
    conditioning = [[cross, {"minimax_refs": [{"kind": "video", "latent": ref}]}]]
    native, native_report = apply_reference_budget(conditioning, mode="native", max_direct_video_rows=16)
    assert native is conditioning
    assert native_report.qwen_rows == 9
    changed, report = apply_reference_budget(
        conditioning,
        mode="decoupled_direct_experimental",
        max_direct_video_rows=16,
    )
    assert changed is not conditioning
    assert changed[0][0] is cross
    changed_shape = changed[0][1]["minimax_refs"][0]["latent"].shape[-2:]
    assert changed_shape[0] % 2 == changed_shape[1] % 2 == 0
    assert conditioning[0][1]["minimax_refs"][0]["latent"] is ref
    assert report.direct_video_rows_after <= 16


def test_reference_budget_handles_thin_aspect_ratios():
    thin = torch.randn(1, 24, 1, 2, 40)
    conditioning = [[torch.randn(1, 3, 4), {"minimax_refs": [{"kind": "video", "latent": thin}]}]]
    changed, report = apply_reference_budget(
        conditioning,
        mode="decoupled_direct_experimental",
        max_direct_video_rows=5,
    )
    fitted = changed[0][1]["minimax_refs"][0]["latent"]
    assert fitted.shape[-2] == 2
    assert report.direct_video_rows_after <= 5


@pytest.mark.parametrize(
    ("name", "calls", "phases"),
    [
        ("sample_sa_solver", 3, ["predicted"] * 3),
        ("sample_sa_solver_pece", 5, ["predicted", "predicted", "corrected", "predicted", "corrected"]),
        ("sample_seeds_3", 7, ["stage_1", "stage_2", "stage_3", "stage_1", "stage_2", "stage_3", "stage_1"]),
    ],
)
def test_sampler_topology(name, calls, phases):
    def function():
        pass

    function.__name__ = name
    sampler = SimpleNamespace(sampler_function=function, extra_options={})
    schedule = torch.tensor([1.0, 0.7, 0.3, 0.0])
    result = _sampler_phases(sampler, schedule)
    assert len(result) == calls
    assert [phase for _, phase in result] == phases
