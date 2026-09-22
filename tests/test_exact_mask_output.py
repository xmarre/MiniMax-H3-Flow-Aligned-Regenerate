from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import torch

from h3_flow_regenerate.comfy_compat import (
    _ProgressiveExactMaskExecutor,
    _canonicalize_exact_masked_output,
    flow_outer_wrapper_with_exact_mask,
)
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, PROGRESSIVE_KEY, FlowBinding


def _res_multistep_endpoint_roundoff(reference: torch.Tensor) -> torch.Tensor:
    """Reproduce x + ((x - denoised) / sigma) * -sigma rounding."""
    x = reference.new_tensor(-0.4338788092136383)
    sigma = reference.new_tensor(0.4)
    return x + ((x - reference) / sigma) * (-sigma)


def _packed_exact_prefix(t=7, prefix=2, h=8, w=12):
    video = torch.arange(24 * t * h * w, dtype=torch.float32).reshape(1, 24, t, h, w)
    video[0, 0, 0, 0, 0] = 0.7935083508491516
    audio = torch.randn(1, 32, 2, 11)
    packed, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, :prefix] = 0
    mask = pack_streams((video_mask, torch.ones_like(audio)))[0]
    return packed, shapes, mask


def _with_one_roundoff_value(packed: torch.Tensor) -> torch.Tensor:
    result = packed.clone()
    rounded = _res_multistep_endpoint_roundoff(result[..., 0])
    assert not torch.equal(rounded, result[..., 0])
    result[..., 0] = rounded
    return result


def test_exact_mask_canonicalization_repairs_res_multistep_roundoff_only():
    latent = torch.tensor([[[0.7935083508491516, 2.0, 3.0, 4.0]]], dtype=torch.float32)
    mask = torch.tensor([[[0.0, 1.0, 1.0, 1.0]]], dtype=torch.float32)
    result = latent.clone()
    result[..., 0] = _res_multistep_endpoint_roundoff(result[..., 0])
    unprotected_before = result[..., 1:].clone()

    canonical, stats = _canonicalize_exact_masked_output(result, latent, mask)

    assert torch.equal(canonical[..., :1], latent[..., :1])
    assert torch.equal(canonical[..., 1:], unprotected_before)
    assert stats["protected_elements"] == 1
    assert stats["pre_restore_exact"] is False
    assert stats["canonicalized"] is True
    assert stats["changed_elements"] == 1
    assert stats["nonfinite_changed_elements"] == 0
    assert stats["max_abs_delta"] > 0
    assert stats["rms_delta"] > 0
    assert stats["final_exact"] is True


def test_exact_mask_canonicalization_is_zero_copy_when_already_exact():
    latent = torch.tensor([[[1.0, 2.0, 3.0]]], dtype=torch.float32)
    mask = torch.tensor([[[0.0, 1.0, 1.0]]], dtype=torch.float32)
    result = latent.clone()
    pointer = result.data_ptr()

    canonical, stats = _canonicalize_exact_masked_output(result, latent, mask)

    assert canonical.data_ptr() == pointer
    assert stats["protected_elements"] == 1
    assert stats["pre_restore_exact"] is True
    assert stats["canonicalized"] is False
    assert stats["changed_elements"] == 0
    assert stats["max_abs_delta"] == 0.0
    assert stats["rms_delta"] == 0.0


def test_mixed_grid_runtime_sees_canonical_prefix_before_hard_postcondition(monkeypatch):
    from test_handoff import FakeLearnedProvider

    fake = ModuleType("comfy")
    fake.samplers = ModuleType("comfy.samplers")
    fake.samplers.KSAMPLER = lambda function, **kw: SimpleNamespace(sampler_function=function, extra_options={})
    monkeypatch.setitem(sys.modules, "comfy", fake)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake.samplers)

    packed, shapes, mask = _packed_exact_prefix()
    binding = FlowBinding()
    provider = FakeLearnedProvider()
    config = ProgressiveTargetInputConfig(
        source_latent_h=4,
        source_latent_w=6,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
    )
    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))
    patcher = SimpleNamespace(model=base, object_patches={})
    guider = SimpleNamespace(
        model_options={
            "transformer_options": {},
            FLOW_BINDING_KEY: binding,
            PROGRESSIVE_KEY: config,
        },
        model_patcher=patcher,
        conds={"positive": []},
    )
    calls = []

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):
            stage = guider.model_options["transformer_options"].get("h3_flow_stage", "single")
            calls.append(stage)
            if stage == "high":
                assert torch.equal(call_mask, mask)
                binding.metrics.event("model_call", actual=True)
                return _with_one_roundoff_value(packed)
            if stage == "probe":
                return latent.clone()
            return latent / (1 - sigmas[-1])

    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    result = flow_outer_wrapper_with_exact_mask(
        Executor(),
        torch.randn_like(packed),
        packed,
        sampler,
        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        mask,
        None,
        True,
        7,
        latent_shapes=list(shapes),
    )

    assert torch.equal(result, packed)
    assert calls == ["low", "probe", "high"]
    events = [event for event in binding.metrics.events if event.kind == "exact_mask_output"]
    assert len(events) == 1
    event = events[0]
    assert event.fields["source"] == "mixed_grid_low_suffix_high"
    assert event.fields["canonicalized"] is True
    assert event.fields["changed_elements"] == 1
    assert binding.metrics.counters["exact_mask_output_canonicalizations"] == 1


def test_target_input_fallback_canonicalizes_single_sampler_return():
    packed, shapes, mask = _packed_exact_prefix()
    binding = FlowBinding()
    config = ProgressiveTargetInputConfig(source_latent_h=4, source_latent_w=6)
    base = SimpleNamespace(process_latent_in=lambda value: value, diffusion_model=SimpleNamespace(blocks=[]))
    guider = SimpleNamespace(
        model_options={
            "transformer_options": {},
            FLOW_BINDING_KEY: binding,
            PROGRESSIVE_KEY: config,
        },
        model_patcher=SimpleNamespace(model=base),
        conds={"positive": []},
    )

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):
            assert latent_shapes == list(shapes)
            assert torch.equal(call_mask, mask)
            return _with_one_roundoff_value(packed)

    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    result = flow_outer_wrapper_with_exact_mask(
        Executor(),
        torch.randn_like(packed),
        packed,
        sampler,
        torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
        mask,
        None,
        True,
        7,
        latent_shapes=list(shapes),
    )

    assert torch.equal(result, packed)
    result_video, result_audio = unpack_streams(result, shapes)
    source_video, source_audio = unpack_streams(packed, shapes)
    assert torch.equal(result_video, source_video)
    assert torch.equal(result_audio, source_audio)
    events = [event for event in binding.metrics.events if event.kind == "exact_mask_output"]
    assert len(events) == 1
    assert events[0].fields["source"] == "target_input_fallback_return"
    assert events[0].fields["canonicalized"] is True
    assert events[0].fields["changed_elements"] == 1


def test_exact_mask_adapter_transfers_audio_transition_before_partitioned_restore():
    video = torch.zeros(1, 24, 3, 2, 2)
    audio = torch.zeros(1, 32, 2, 10)
    audio[..., :6] = 1.0
    audio[..., 6:] = 2.0
    packed, shapes = pack_streams((video, audio))

    video_mask = torch.ones_like(video)
    video_mask[:, :, :1] = 0
    audio_mask = torch.ones_like(audio)
    audio_mask[..., :6] = 0
    exact_mask = pack_streams((video_mask, audio_mask))[0]

    returned = packed.clone()
    _returned_video, returned_audio = unpack_streams(returned, shapes)
    returned_audio[..., 5] = 1.75
    returned_audio[..., 6] = 2.20
    sampled_relation = (
        returned_audio[..., 6].clone().to(torch.float32)
        - returned_audio[..., 5].clone().to(torch.float32)
    )

    binding = FlowBinding()
    guider = SimpleNamespace(model_options={"transformer_options": {"h3_flow_stage": "high"}})

    class Executor:
        class_obj = guider

        def __call__(self, *args, **kwargs):
            return returned

    sampler = SimpleNamespace(sampler_function=lambda: None, extra_options={})
    adapted = _ProgressiveExactMaskExecutor(
        Executor(),
        binding=binding,
        progressive=SimpleNamespace(exact_prefix_mode="partitioned_exact_prefix"),
        latent_image=packed,
        denoise_mask=exact_mask,
        sampler=sampler,
        latent_shapes=list(shapes),
        audio_exact_restore_suffix_bridge=True,
    )
    result = adapted(None)
    _result_video, result_audio = unpack_streams(result, shapes)

    assert torch.equal(result_audio[..., :6], audio[..., :6])
    torch.testing.assert_close(
        result_audio[..., 6].to(torch.float32) - result_audio[..., 5].to(torch.float32),
        sampled_relation,
        rtol=0.0,
        atol=1.0e-6,
    )
    assert torch.equal(result_audio[..., 7:], audio[..., 7:])

    bridge_events = [
        event for event in binding.metrics.events if event.kind == "audio_exact_restore_suffix_bridge"
    ]
    assert len(bridge_events) == 1
    bridge = bridge_events[0].fields
    assert bridge["source"] == "partitioned_exact_prefix_high"
    assert bridge["applied"] is True
    assert bridge["corrected_ticks"] == 1
    assert bridge["final_exact_restore_follows"] is True
    assert bridge["extra_h3_nfe"] == 0
    assert bridge["extra_sampler_lifetimes"] == 0
    assert bridge["extra_vae_calls"] == 0
    assert binding.metrics.counters["audio_exact_restore_suffix_bridge_runs"] == 1
