import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.attention import AttentionConfig, layout_summary, make_attention_override, video_local_mask
from h3_flow_regenerate.comfy_compat import patch_flow_model, reconfigure_binding
from h3_flow_regenerate.contracts import H3FlowTrajectory
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.handoff import ProgressiveHandoffConfig, ProgressiveTargetInputConfig
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.nodes import H3FlowAlignedRefineState
from h3_flow_regenerate.reference import apply_reference_budget
from h3_flow_regenerate.runtime import (
    FLOW_BINDING_KEY,
    PROGRESSIVE_KEY,
    SPECTRUM_ACTUAL_KEY,
    SPECTRUM_OUTER_STEP_KEY,
    SPECTRUM_PHASE_KEY,
    FlowBinding,
    _begin_capture,
    _conditioning_signature,
    _finish_capture,
    _merge_preserved_noise,
    _noise_argument,
    _resize_packed_mask,
    _run_progressive,
    _sampler_phases,
    conditioning_signature_from_conditioning,
    flow_predict_wrapper,
)


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


def test_sparse_attention_uses_query_key_additive_mask():
    test_layout = layout()
    q = torch.randn(1, 2, 20, 4)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    observed = []

    def backend(q, _k, _v, heads, mask=None, skip_reshape=False, **_kwargs):
        assert skip_reshape
        b, h, q_len, d = q.shape
        if mask is not None:
            assert mask.dtype == q.dtype
            assert mask.ndim == 2
            assert mask.shape == (q_len, 20)
            assert bool((mask <= 0).all())
            observed.append(mask.detach().clone())
        return torch.zeros(b, q_len, h * d, dtype=q.dtype)

    config = AttentionConfig(
        mode="experimental_sparse",
        layers=(0,),
        global_heads=1,
        sparse_window=1,
        query_chunk=7,
        max_sequence=64,
    )
    metrics = H3FlowMetrics()
    override = make_attention_override(config, metrics)
    result = override(
        backend,
        q,
        k,
        v,
        2,
        mask=None,
        skip_reshape=True,
        transformer_options={"h3_flow_attention_context": {"layout": test_layout, "layer": 0}},
    )
    assert result.shape == (1, 20, 8)
    assert [mask.shape[0] for mask in observed] == [7, 7, 6]
    first_video = observed[1][1]
    assert first_video[:8].eq(0).all()
    assert first_video[8].item() == 0
    assert first_video[13].item() < 0


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


def test_repatch_preserves_existing_flow_attention_context_wrapper(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)

    attention_metrics = H3FlowMetrics()
    attention_model, _ = patch_flow_model(
        FakePatcher(),
        attention=AttentionConfig(mode="diagnostic", layers=(0,)),
        metrics=attention_metrics,
    )
    attention_wrapper = attention_model.model_options["transformer_options"]["patches_replace"]["dit"][
        ("double_block", 0)
    ]
    assert attention_wrapper._h3_flow_layout_scope == "attention"

    flow_metrics = H3FlowMetrics()
    repatched, _ = patch_flow_model(attention_model, metrics=flow_metrics)
    outer = repatched.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    assert outer._h3_flow_layout_scope == "layout"
    assert outer._h3_flow_metrics is flow_metrics
    assert outer._h3_flow_previous is attention_wrapper

    replacement_metrics = H3FlowMetrics()
    replaced, _ = patch_flow_model(
        repatched,
        attention=AttentionConfig(mode="diagnostic", layers=(0,)),
        metrics=replacement_metrics,
    )
    replacement = replaced.model_options["transformer_options"]["patches_replace"]["dit"][("double_block", 0)]
    assert replacement._h3_flow_layout_scope == "attention"
    assert replacement._h3_flow_metrics is replacement_metrics
    assert not getattr(replacement._h3_flow_previous, "_h3_flow_layout_wrapper", False)


def test_patch_can_explicitly_disable_inherited_capture(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)

    first, first_binding = patch_flow_model(FakePatcher(), capture_enabled=True)
    assert first_binding.capture_enabled
    _, second_binding = patch_flow_model(first, capture_enabled=False)
    assert not second_binding.capture_enabled


def test_patch_can_explicitly_clear_inherited_progressive_handoff(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)

    progressive = ProgressiveTargetInputConfig(source_latent_h=4, source_latent_w=4)
    first, _ = patch_flow_model(FakePatcher(), progressive=progressive)
    assert first.model_options[PROGRESSIVE_KEY] is progressive

    inherited, _ = patch_flow_model(first)
    assert inherited.model_options[PROGRESSIVE_KEY] is progressive

    cleared, _ = patch_flow_model(inherited, clear_progressive=True)
    assert PROGRESSIVE_KEY not in cleared.model_options

    with pytest.raises(ValueError, match="cannot set and clear"):
        patch_flow_model(first, progressive=progressive, clear_progressive=True)


def test_patch_can_explicitly_clear_inherited_guidance_signature(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)

    first, first_binding = patch_flow_model(FakePatcher(), guidance_conditioning_signature="source")
    assert first_binding.guidance_conditioning_signature == "source"

    inherited, inherited_binding = patch_flow_model(first)
    assert inherited_binding.guidance_conditioning_signature == "source"

    _, cleared = patch_flow_model(inherited, clear_guidance_conditioning_signature=True)
    assert cleared.guidance_conditioning_signature is None

    with pytest.raises(ValueError, match="cannot set and clear guidance conditioning signature"):
        patch_flow_model(
            first,
            guidance_conditioning_signature="other",
            clear_guidance_conditioning_signature=True,
        )


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


def test_conditioning_signature_tracks_content_across_tensor_clones():
    cross = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    ref = torch.arange(1 * 24 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 24, 1, 4, 4)

    def guider(cross_tensor, ref_tensor):
        return SimpleNamespace(
            original_conds={
                "positive": [
                    {
                        "cross_attn": cross_tensor,
                        "minimax_refs": [{"kind": "video", "latent": ref_tensor}],
                    }
                ]
            }
        )

    signature = _conditioning_signature(guider(cross, ref))
    assert signature == _conditioning_signature(guider(cross.clone(), ref.clone()))
    assert signature != _conditioning_signature(guider(cross + 1, ref))
    assert signature != _conditioning_signature(guider(cross, ref + 1))


def test_conditioning_signature_is_strict_for_keyframe_content_and_geometry():
    cross = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    low = torch.arange(1 * 24 * 1 * 4 * 4, dtype=torch.float32).reshape(1, 24, 1, 4, 4)
    high = torch.nn.functional.interpolate(
        low.reshape(24, 1, 4, 4),
        size=(8, 8),
        mode="bilinear",
        align_corners=False,
    ).reshape(1, 24, 1, 8, 8)

    def guider(keyframe):
        return SimpleNamespace(
            original_conds={
                "positive": [
                    {
                        "cross_attn": cross,
                        "model_conds": {},
                        "minimax_keyframes": [{"latent": keyframe}],
                    }
                ]
            }
        )

    signature = _conditioning_signature(guider(low))
    assert signature != _conditioning_signature(guider(high))
    assert signature != _conditioning_signature(guider(low + 1))


def test_raw_conditioning_signature_matches_cfg_guider_conversion():
    cross = torch.arange(24, dtype=torch.float32).reshape(1, 3, 8)
    keyframe = torch.randn(1, 24, 1, 4, 4)
    raw = [[cross, {"minimax_keyframes": [{"latent": keyframe}], "tag": "chunk"}]]
    converted = SimpleNamespace(
        original_conds={
            "positive": [
                {
                    "cross_attn": cross,
                    "model_conds": {},
                    "minimax_keyframes": [{"latent": keyframe}],
                    "tag": "chunk",
                }
            ]
        }
    )
    assert conditioning_signature_from_conditioning(raw) == _conditioning_signature(converted)


def test_continuum_refine_state_patch_preserves_payload_and_disables_capture(monkeypatch):
    fake_extension = ModuleType("comfy.patcher_extension")
    fake_extension.WrappersMP = SimpleNamespace(OUTER_SAMPLE="outer_sample", PREDICT_NOISE="predict_noise")
    fake_comfy = ModuleType("comfy")
    fake_comfy.patcher_extension = fake_extension
    monkeypatch.setitem(sys.modules, "comfy", fake_comfy)
    monkeypatch.setitem(sys.modules, "comfy.patcher_extension", fake_extension)

    trajectory = H3FlowTrajectory()
    base_model, _ = patch_flow_model(FakePatcher(), trajectory=trajectory, capture_enabled=True)
    positive = [[torch.zeros(1, 2, 4), {"tag": "chunk"}]]
    opaque = object()
    state = {"api": 1, "model": base_model, "positive": positive, "opaque": opaque}

    patched_state, metrics = H3FlowAlignedRefineState().patch(
        state,
        trajectory,
        "direction",
        0.35,
        0.0,
        0.0,
        0.25,
    )
    binding = patched_state["model"].model_options[FLOW_BINDING_KEY]
    assert patched_state is not state
    assert patched_state["positive"] is positive
    assert patched_state["opaque"] is opaque
    assert binding.trajectory is trajectory
    assert binding.guidance.mode == "direction"
    assert not binding.capture_enabled
    assert binding.guidance_conditioning_signature == conditioning_signature_from_conditioning(positive)
    assert binding.metrics is metrics
    layout_wrapper = patched_state["model"].model_options["transformer_options"]["patches_replace"]["dit"][
        ("double_block", 0)
    ]
    assert layout_wrapper._h3_flow_metrics is metrics


def test_spectrum_forecasts_never_become_exact_trajectory_anchors():
    video = torch.full((1, 24, 1, 4, 4), 2.0)
    audio = torch.full((1, 32, 2, 5), 3.0)
    packed, shapes = pack_streams((video, audio))
    trajectory = H3FlowTrajectory()
    binding = FlowBinding(trajectory=trajectory, capture_enabled=True, capture_forecasts=False)
    guider = SimpleNamespace(
        model_options={FLOW_BINDING_KEY: binding, "transformer_options": {}},
        original_conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
    )

    def native():
        pass

    native.__name__ = "sample_sa_solver_pece"
    sampler = SimpleNamespace(sampler_function=native, extra_options={})
    _begin_capture(binding, guider, sampler, torch.tensor([1.0, 0.5, 0.0]), list(shapes))

    class Executor:
        class_obj = guider

        def __call__(self, x, timestep, model_options=None, seed=None):
            return packed

    forecast_options = {
        "transformer_options": {
            SPECTRUM_ACTUAL_KEY: False,
            SPECTRUM_PHASE_KEY: "predicted",
            SPECTRUM_OUTER_STEP_KEY: 0,
        }
    }
    actual_options = {
        "transformer_options": {
            SPECTRUM_ACTUAL_KEY: True,
            SPECTRUM_PHASE_KEY: "corrected",
            SPECTRUM_OUTER_STEP_KEY: 1,
        }
    }
    flow_predict_wrapper(Executor(), packed, torch.tensor([0.5]), forecast_options, 7)
    flow_predict_wrapper(Executor(), packed, torch.tensor([0.4]), actual_options, 7)
    _finish_capture(binding)

    run = trajectory.latest
    assert len(run.samples) == 1
    assert run.samples[0].provenance == "actual"
    assert run.samples[0].phase == "corrected"
    assert run.samples[0].call_index == 1
    assert binding.metrics.counters["spectrum_forecast_calls"] == 1
    assert binding.metrics.counters["transformer_actual_nfe"] == 1


def test_noise_reconstruction_includes_preserved_latent_image():
    base_model = SimpleNamespace(model_sampling=SimpleNamespace(noise_scale=2.0))
    state = torch.randn(1, 1, 32)
    latent = torch.randn_like(state)
    sigma = 0.4
    noise = _noise_argument(base_model, state, sigma, latent)
    reconstructed = sigma * 2.0 * noise + (1.0 - sigma) * latent
    assert torch.allclose(reconstructed, state)


def test_progressive_mask_resize_uses_prepared_full_channel_av_geometry():
    source_shapes = [(1, 24, 1, 4, 4), (1, 32, 2, 5)]
    target_shapes = [(1, 24, 1, 8, 6), (1, 32, 2, 5)]
    video_mask = torch.arange(16, dtype=torch.float32).reshape(1, 1, 1, 4, 4).repeat(1, 24, 1, 1, 1)
    audio_mask = torch.arange(10, dtype=torch.float32).reshape(1, 1, 2, 5).repeat(1, 32, 1, 1)
    packed_mask, _ = pack_streams((video_mask, audio_mask))

    resized = _resize_packed_mask(packed_mask, source_shapes, target_shapes)
    out_video, out_audio = unpack_streams(resized, target_shapes)
    assert out_video.shape == (1, 24, 1, 8, 6)
    assert torch.equal(out_audio, audio_mask)


def test_preserved_noise_merge_keeps_native_noise_only_under_mask():
    generated = torch.full((1, 1, 8), 7.0)
    native = torch.arange(8, dtype=torch.float32).reshape(1, 1, 8)
    mask = torch.tensor([0.0, 0.0, 0.5, 1.0, 1.0, 0.25, 0.75, 0.0]).reshape(1, 1, 8)
    merged = _merge_preserved_noise(generated, native, mask)
    expected = generated * mask + native * (1.0 - mask)
    assert torch.equal(merged, expected)
    assert torch.equal(merged[..., :2], native[..., :2])
    assert torch.equal(merged[..., 3:5], generated[..., 3:5])


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
    original_cond = {
        "cross_attn": torch.zeros(1, 2, 4),
        "minimax_keyframes": [{"latent": source_video.clone(), "latent_h": 4, "latent_w": 4}],
    }
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
                    tuple(guider.conds["positive"][0]["minimax_keyframes"][0]["latent"].shape[-2:]),
                )
            )
            guider.conds["positive"][0]["stage_mutated"] = True
            if call_sampler.sampler_function.__name__ == "_h3_flow_exact_probe":
                return source_x0
            if len(calls) == 1:
                return source_raw / (1.0 - call_sigmas[-1])
            if len(calls) == 3:
                binding.metrics.event("model_call", actual=True)
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
    assert calls[1][2]["h3_refinement"]["sigma_reference"] == 1.0
    assert calls[-1][2]["h3_refinement"]["min_actual_prefix_steps"] == 1
    assert [call[4] for call in calls] == [False, False, False]
    assert [call[5] for call in calls] == [(4, 4), (4, 4), (8, 6)]
    assert mutable_shapes[0] == (1, 24, 1, 8, 6)
    _, result_audio = __import__("h3_flow_regenerate.geometry", fromlist=["unpack_streams"]).unpack_streams(
        result * float(calls[-1][3][0]), mutable_shapes
    )
    assert torch.allclose(result_audio, source_audio)


def test_reference_native_parity_and_direct_only_decoupling():
    cross = torch.randn(1, 9, 12)
    ref = torch.randn(1, 24, 2, 16, 16)
    conditioning = [
        [
            cross,
            {
                "minimax_refs": [
                    {
                        "kind": "video",
                        "latent_t": 2,
                        "latent_h": 16,
                        "latent_w": 16,
                        "latent": ref,
                    }
                ]
            },
        ]
    ]
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
    changed_ref = changed[0][1]["minimax_refs"][0]
    changed_shape = changed_ref["latent"].shape[-2:]
    assert changed_shape[0] % 2 == changed_shape[1] % 2 == 0
    assert changed_ref["latent_t"] == changed_ref["latent"].shape[2]
    assert changed_ref["latent_h"] == changed_shape[0]
    assert changed_ref["latent_w"] == changed_shape[1]
    assert conditioning[0][1]["minimax_refs"][0]["latent"] is ref
    assert conditioning[0][1]["minimax_refs"][0]["latent_h"] == 16
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


def test_target_input_progressive_keeps_continuum_target_geometry(monkeypatch):
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

    target_video = torch.randn(1, 24, 1, 8, 6)
    target_audio = torch.randn(1, 32, 2, 5)
    target_latent, target_shapes = pack_streams((target_video, target_audio))
    target_noise = torch.randn_like(target_latent)
    video_mask = torch.ones(1, 24, 1, 8, 6)
    video_mask[:, :, :, :2] = 0
    audio_mask = torch.ones(1, 32, 2, 5)
    packed_mask, _ = pack_streams((video_mask, audio_mask))
    source_video = torch.randn(1, 24, 1, 4, 4)
    source_audio = target_audio.clone()
    source_raw, _source_shapes = pack_streams((source_video, source_audio))
    source_x0 = source_raw.clone()
    sigmas = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])

    def native():
        pass

    native.__name__ = "sample_sa_solver_pece"
    sampler = SimpleNamespace(sampler_function=native, extra_options={})
    original_cond = {
        "cross_attn": torch.zeros(1, 2, 4),
        "minimax_keyframes": [{"latent": target_video.clone(), "latent_h": 8, "latent_w": 6}],
    }
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=MiniMaxH3()),
        original_conds={"positive": [original_cond]},
        conds={"positive": [original_cond.copy()]},
    )
    calls = []

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent, call_sampler, call_sigmas, mask, *args, latent_shapes):
            mask_shapes = None
            if mask is not None:
                mask_shapes = tuple(
                    tensor.shape
                    for tensor in unpack_streams(
                        mask,
                        latent_shapes,
                    )
                )
            calls.append(
                {
                    "name": call_sampler.sampler_function.__name__,
                    "shapes": list(latent_shapes),
                    "keyframe_hw": tuple(guider.conds["positive"][0]["minimax_keyframes"][0]["latent"].shape[-2:]),
                    "mask_shapes": mask_shapes,
                }
            )
            if call_sampler.sampler_function.__name__ == "_h3_flow_exact_probe":
                return source_x0
            if len(calls) == 1:
                return source_raw / (1.0 - call_sigmas[-1])
            if len(calls) == 3:
                binding.metrics.event("model_call", actual=True)
            return noise

    mutable_shapes = list(target_shapes)
    binding = FlowBinding(trajectory=H3FlowTrajectory(), guidance=GuidanceConfig(mode="off"), capture_enabled=True)
    result = _run_progressive(
        Executor(),
        guider,
        binding,
        ProgressiveTargetInputConfig(source_latent_h=4, source_latent_w=4, handoff_coordinate=0.3),
        target_noise,
        target_latent,
        sampler,
        sigmas,
        packed_mask,
        None,
        True,
        7,
        mutable_shapes,
    )

    assert [call["name"] for call in calls] == [
        "sample_sa_solver_pece",
        "_h3_flow_exact_probe",
        "sample_sa_solver_pece",
    ]
    assert [call["shapes"][0][-2:] for call in calls] == [(4, 4), (4, 4), (8, 6)]
    assert [call["keyframe_hw"] for call in calls] == [(4, 4), (4, 4), (8, 6)]
    assert calls[0]["mask_shapes"] == (torch.Size([1, 24, 1, 4, 4]), torch.Size([1, 32, 2, 5]))
    assert calls[2]["mask_shapes"] == (torch.Size([1, 24, 1, 8, 6]), torch.Size([1, 32, 2, 5]))
    assert mutable_shapes == target_shapes
    assert result.shape == target_latent.shape
    assert binding.trajectory.latest.geometry.latent_h == 4
    assert binding.trajectory.latest.geometry.latent_w == 4
    handoff = [event for event in binding.metrics.events if event.kind == "handoff_complete"][-1]
    assert handoff.fields["high_stage_first_call_actual"] is True
    assert handoff.fields["high_stage_model_calls"] == 1


def test_progressive_high_failure_invalidates_committed_low_trajectory(monkeypatch):
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

    video = torch.randn(1, 24, 1, 4, 4)
    audio = torch.randn(1, 32, 2, 5)
    packed, shapes = pack_streams((video, audio))
    sigmas = torch.tensor([1.0, 0.8, 0.4, 0.0])

    def native():
        pass

    native.__name__ = "sample_euler"
    sampler = SimpleNamespace(sampler_function=native, extra_options={})
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=MiniMaxH3()),
        original_conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
        conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
    )
    calls = 0

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent, call_sampler, call_sigmas, *args, latent_shapes):
            nonlocal calls
            calls += 1
            if call_sampler.sampler_function.__name__ == "_h3_flow_exact_probe":
                return packed
            if calls == 1:
                return packed / (1.0 - call_sigmas[-1])
            raise RuntimeError("synthetic high failure")

    trajectory = H3FlowTrajectory()
    binding = FlowBinding(
        trajectory=trajectory,
        guidance=GuidanceConfig(mode="off"),
        capture_enabled=True,
    )
    with pytest.raises(RuntimeError, match="synthetic high failure"):
        _run_progressive(
            Executor(),
            guider,
            binding,
            ProgressiveHandoffConfig(8, 6, handoff_coordinate=0.3),
            packed,
            torch.zeros_like(packed),
            sampler,
            sigmas,
            None,
            None,
            True,
            7,
            list(shapes),
        )
    assert len(trajectory.runs) == 1
    assert not trajectory.runs[0].complete
    assert "progressive continuation failed" in trajectory.runs[0].abort_reason
    assert any(event.kind == "trajectory_invalidate" for event in binding.metrics.events)


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
