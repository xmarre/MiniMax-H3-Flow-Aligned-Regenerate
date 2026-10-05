"""End-to-end continuation scheduling through the actual Flow scheduler and Core forward.

Sampler tests call the real Flow partitioned transformer wrapper around a tiny
real MiniMax-H3 Core model, with Sol's dense reference arithmetic on CPU. Most
use a deterministic Euler oracle; native sampler cases also exercise ComfyUI's
noise scaling, inpaint wrapper and multistep/stochastic updates. The upscaler
is a bicubic fixture, so these tests establish state ownership, not visual quality.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.contracts import H3FlowTrajectory
from h3_flow_regenerate.geometry import pack_streams, resize_spatial_5d, unpack_streams
from h3_flow_regenerate.handoff import H3_LATENT_UPSCALER_API_VERSION, H3_LATENT_UPSCALER_KIND
from h3_flow_regenerate.partitioned_band import target_band_padding_max_abs, target_band_tail
from h3_flow_regenerate.partitioned_diagnostics import (
    PARTITIONED_SPATIAL_STAGE_CONTROL_KEY,
    PARTITIONED_SPATIAL_STAGE_PROGRESSIVE,
    PARTITIONED_SPATIAL_STAGE_SAME_GRID,
    PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
    PARTITIONED_SUFFIX_DC_BRIDGE_KEY,
    PARTITIONED_TARGET_BAND_TOKENS_KEY,
)
from h3_flow_regenerate.partitioned_scheduler import (
    PARTITIONED_SOL_REQUIRED_METADATA,
    SOL_RUNTIME_KEY,
    VDN_PARTITIONED_BOUNDARY_QUERY_API,
    VDN_PARTITIONED_BOUNDARY_QUERY_POLICY,
    run_partitioned_progressive,
)
from h3_flow_regenerate.partitioned_stage import PartitionedTargetBandGeometry
from h3_flow_regenerate.runtime import FLOW_BINDING_KEY, PROBE_MARKER, FlowBinding, flow_predict_wrapper, sampler_name

PROTECTED_T, TEMPORAL = 8, 14
TARGET_HW, SOURCE_HW = (40, 48), (20, 24)
AUDIO_T = 13


class _Upscaler:
    api_version = H3_LATENT_UPSCALER_API_VERSION
    kind = H3_LATENT_UPSCALER_KIND
    model_name = "bicubic-test-provider"
    device = "cpu"
    inference_device = "cpu"
    precision = "fp32"
    offload_after_upscale = False
    h3_patch_lattice_api = 2

    def __init__(self):
        self.inputs = []

    def upscale_clean_video(self, video, *, target_h, target_w):
        self.inputs.append(video.detach().clone())
        return resize_spatial_5d(video, target_h, target_w, mode="bicubic").to(video)

    upscale_clean_video_h3_patch_lattice = upscale_clean_video


def _vdn_owner():
    owner = SimpleNamespace()
    owner._vdn_forward = True
    owner._vdn_partitioned_boundary_query_api = VDN_PARTITIONED_BOUNDARY_QUERY_API
    owner._vdn_partitioned_boundary_query_policy = VDN_PARTITIONED_BOUNDARY_QUERY_POLICY
    owner._vdn_external_sequence_api = 4
    owner._vdn_partitioned_native_carrier_grids = ("source", "target")
    return owner


def _harness(
    monkeypatch,
    *,
    spatial_stage_control,
    extra_transformer_options=None,
    guidance_mode="off",
    native_sampler=None,
):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    pytest.importorskip("sol_h3.runtime")
    import comfy.ops
    from comfy.ldm.minimax.model import MiniMaxH3Model
    from comfy.patcher_extension import WrapperExecutor
    from sol_h3 import partitioned_request
    from sol_h3.contracts import Config
    from sol_h3.runtime import _FORWARD, _REQUEST, Request

    from h3_flow_regenerate import partitioned_transformer as transform

    monkeypatch.setattr(partitioned_request, "_validate_thd", lambda *_args: None)

    def cpu_key_bias(state, *, kv_rows, prefix_range, log_measure, device, semantic_digest):
        if prefix_range is None or float(log_measure) == 0.0:
            return None
        bias = torch.zeros(kv_rows, dtype=torch.float32, device=device)
        bias[prefix_range[0] : prefix_range[1]] = float(log_measure)
        return bias

    monkeypatch.setattr(partitioned_request, "_key_bias", cpu_key_bias)
    monkeypatch.setattr(
        partitioned_request,
        "_checked_weighted_dense",
        lambda q, k, v, key_bias, *, scale, **_kw: partitioned_request._weighted_dense(q, k, v, key_bias, scale=scale),
    )
    if spatial_stage_control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND:
        from sol_h3 import partitioned_history

        if "target" not in getattr(partitioned_history, "PARTITIONED_NATIVE_CARRIER_GRIDS", ()):
            pytest.skip("installed Sol-H3 predates target native carrier recognition")

    generator = torch.Generator().manual_seed(71)
    dm = MiniMaxH3Model(
        hidden_size=8,
        num_layers=2,
        token_refiner_num_layers=0,
        num_attention_heads=1,
        attention_head_dim=128,
        ffn_hidden_size=16,
        text_dim=8,
        time_embed_dim=2,
        adaln_curve_grid=4,
        rope_inv_freq_len=2,
        dtype=torch.float32,
        device="cpu",
        operations=comfy.ops.disable_weight_init,
    )
    dm.requires_grad_(False)
    with torch.no_grad():
        for param in dm.parameters():
            param.copy_(torch.randn(param.shape, generator=generator) * 0.02)
        dm.adaln_t_table.copy_(torch.randn(dm.adaln_t_table.shape, generator=generator))
        dm.rope.inv_freq.fill_(0.2)
    context = torch.randn(1, 3, 8, generator=generator)

    target_video = torch.randn(1, 24, TEMPORAL, *TARGET_HW, generator=generator)
    target_audio = torch.randn(1, 32, 2, AUDIO_T, generator=generator)
    latent_image, shapes = pack_streams((target_video, target_audio))
    shapes = list(shapes)
    noise = pack_streams(
        (
            torch.randn(target_video.shape, generator=generator),
            torch.randn(target_audio.shape, generator=generator),
        )
    )[0]
    video_mask = torch.ones_like(target_video)
    video_mask[:, :, :PROTECTED_T] = 0
    mask = pack_streams((video_mask, torch.ones_like(target_audio)))[0]

    transformer_options = {SOL_RUNTIME_KEY: dict(PARTITIONED_SOL_REQUIRED_METADATA)}
    if spatial_stage_control != PARTITIONED_SPATIAL_STAGE_PROGRESSIVE:
        transformer_options[PARTITIONED_SPATIAL_STAGE_CONTROL_KEY] = spatial_stage_control
    transformer_options.update(extra_transformer_options or {})
    base_model = SimpleNamespace(
        model_sampling=SimpleNamespace(noise_scale=1.0),
        latent_shapes=None,
        process_latent_in=lambda value: value,
        process_latent_out=lambda value: value,
    )
    guider = SimpleNamespace(
        model_options={"transformer_options": transformer_options},
        conds={"positive": [{}]},
        model_patcher=SimpleNamespace(
            object_patches={"diffusion_model.blocks.0.attn.forward": _vdn_owner()},
            model=base_model,
            model_options=None,
        ),
    )
    guider.model_patcher.model_options = guider.model_options
    guider.model_patcher.get_model_object = lambda name: getattr(base_model, name)
    from h3_flow_regenerate.guidance import GuidanceConfig

    binding = FlowBinding(
        trajectory=H3FlowTrajectory(),
        capture_enabled=True,
        guidance=GuidanceConfig(mode=guidance_mode),
    )
    guider.model_options[FLOW_BINDING_KEY] = binding
    # Core's model wrapper publishes the active sampler geometry here.
    guider.inner_model = SimpleNamespace(latent_shapes=None)
    calls = []

    def core(x, sigma, shapes, mask_packed):
        video, audio = unpack_streams(x, shapes)
        video_mask, audio_mask = unpack_streams(mask_packed, shapes)
        state = Request(Config(backend="sol"))
        request = _REQUEST.set(state)
        execution = _FORWARD.set((dm, state, 0, set(), []))
        try:
            with torch.no_grad():
                out = WrapperExecutor.new_class_executor(
                    dm._forward, dm, [transform.partitioned_diffusion_wrapper]
                ).execute(
                    [video, audio],
                    torch.tensor([float(sigma) * 1000.0]),
                    context,
                    transformer_options=guider.model_options["transformer_options"],
                    denoise_mask=video_mask,
                    audio_denoise_mask=audio_mask,
                )
        finally:
            _FORWARD.reset(execution)
            _REQUEST.reset(request)
        velocity = pack_streams((out[0] * video_mask, out[1] * audio_mask))[0]
        return x - float(sigma) * velocity

    def model(x, sigma, shapes, mask_packed):
        """Evaluate through Flow's predict wrapper, as ComfyUI's guider does."""

        class Predict:
            class_obj = guider

            def __call__(self, x, timestep, model_options, seed):
                return core(x, float(timestep.reshape(-1)[0]), shapes, mask_packed)

        return flow_predict_wrapper(Predict(), x, torch.tensor([float(sigma)]), guider.model_options, 5)

    def executor(noise, latent_image, sampler, sigmas, denoise_mask, callback, disable_pbar, seed, latent_shapes):
        shapes = list(latent_shapes)
        guider.inner_model.latent_shapes = shapes
        assert noise.shape == latent_image.shape == denoise_mask.shape
        record = {
            "stage": guider.model_options["transformer_options"].get("h3_flow_stage"),
            "shapes": shapes,
            "noise": noise.clone(),
            "latent": latent_image.clone(),
            "mask": denoise_mask.clone(),
            "sigmas": sigmas.clone(),
        }
        calls.append(record)

        if native_sampler is not None:
            from comfy.model_sampling import CONST

            base_model.model_sampling = CONST()
            base_model.model_sampling.noise_scale = 1.7
            base_model.model_sampling.sigma_max = torch.tensor(1.0)
            base_model.scale_latent_inpaint = lambda *, x, sigma, noise, latent_image, denoise_mask: (
                base_model.model_sampling.noise_scaling(sigma, noise, latent_image)
            )

            class Model:
                inner_model = base_model
                model_patcher = guider.model_patcher
                cfg = 1.0

                def __call__(self, x, sigma, **_kwargs):
                    record.setdefault("entry_state", x.detach().clone())
                    return model(x, float(sigma.reshape(-1)[0]), shapes, denoise_mask)

            result = sampler.sample(
                Model(),
                sigmas,
                {"seed": seed},
                callback,
                noise,
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                disable_pbar=True,
            )
            record["final_state"] = result * (1.0 - float(sigmas[-1]))
            if record["stage"] == "high":
                from h3_flow_regenerate.comfy_compat import _canonicalize_exact_masked_output

                # Match the production _ProgressiveExactMaskExecutor boundary.
                result, _ = _canonicalize_exact_masked_output(result, latent_image, denoise_mask)
            return result

        def inpaint(x, sigma):
            protected = latent_image * (1.0 - sigma) + noise * sigma
            return x * denoise_mask + protected * (1.0 - denoise_mask)

        if sampler_name(sampler) == PROBE_MARKER:
            sigma = float(sigmas[0])
            x = noise * sigma + latent_image * (1.0 - sigma)
            denoised = model(inpaint(x, sigma), sigma, shapes, denoise_mask)
            denoised = denoised * denoise_mask + latent_image * (1.0 - denoise_mask)
            record["entry_state"] = x.clone()
            return denoised
        sigma0 = float(sigmas[0])
        x = noise * sigma0 + latent_image * (1.0 - sigma0)
        record["entry_state"] = x.clone()
        for index in range(int(sigmas.numel()) - 1):
            sigma, following = float(sigmas[index]), float(sigmas[index + 1])
            x = inpaint(x, sigma)
            denoised = model(x, sigma, shapes, denoise_mask)
            denoised = denoised * denoise_mask + latent_image * (1.0 - denoise_mask)
            x = x + (following - sigma) * (x - denoised) / sigma
            if callback is not None:
                callback(index, denoised, x, int(sigmas.numel()) - 1)
        record["final_state"] = x.clone()
        last = float(sigmas[-1])
        if last == 0.0:
            # Production wraps the sampler with Flow's exact-mask output adapter,
            # which restores protected values byte-exactly at the terminal sigma.
            return torch.where(denoise_mask == 0, latent_image, x)
        return x / (1.0 - last)

    upscaler = _Upscaler()
    from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig

    config = ProgressiveTargetInputConfig(
        source_latent_h=SOURCE_HW[0],
        source_latent_w=SOURCE_HW[1],
        handoff_coordinate=0.35,
        transfer_mode="learned_3d",
        learned_upscaler=upscaler,
        exact_prefix_mode="fallback",
        frame_gauge_repair=True,
    )
    sampler = SimpleNamespace(sampler_function=SimpleNamespace(__name__="sample_euler"), extra_options={})
    if native_sampler is not None:
        import comfy.samplers

        sampler = comfy.samplers.ksampler(native_sampler)
    sigmas = torch.linspace(1.0, 0.0, 9)
    result = run_partitioned_progressive(
        executor,
        guider,
        binding,
        config,
        noise,
        latent_image,
        sampler,
        sigmas,
        mask,
        None,
        True,
        5,
        shapes,
        exact_denoise_mask=mask,
    )
    return SimpleNamespace(
        binding=binding,
        result=result,
        calls=calls,
        shapes=shapes,
        latent_image=latent_image,
        upscaler=upscaler,
        metrics=binding.metrics,
        guider=guider,
    )


def _events(metrics, kind):
    return [event.fields for event in metrics.events if event.kind == kind]


@pytest.mark.parametrize(
    "control",
    [PARTITIONED_SPATIAL_STAGE_PROGRESSIVE, PARTITIONED_SPATIAL_STAGE_SAME_GRID, PARTITIONED_SPATIAL_STAGE_TARGET_BAND],
)
def test_every_spatial_stage_control_keeps_exact_prefix_ownership(monkeypatch, control):
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 2} if control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND else None
    run = _harness(monkeypatch, spatial_stage_control=control, extra_transformer_options=extra)
    stages = [call["stage"] for call in run.calls]
    assert stages == ["low", "probe", "high"]
    video, audio = unpack_streams(run.result, run.shapes)
    exact_video, _exact_audio = unpack_streams(run.latent_image, run.shapes)
    assert video.shape == exact_video.shape
    assert torch.equal(video[:, :, :PROTECTED_T], exact_video[:, :, :PROTECTED_T])
    assert bool(torch.isfinite(video).all()) and bool(torch.isfinite(audio).all())
    assert run.guider.model_options["transformer_options"].get("h3_flow_partitioned_stage_v1") is None
    transfers = _events(run.metrics, "partitioned_transfer")
    assert len(transfers) == 1
    assert transfers[0]["authoritative_target_prefix_restored"] is True


def test_target_band_runs_band_and_tail_with_identity_and_learned_handoffs(monkeypatch):
    band_t = 2
    run = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={PARTITIONED_TARGET_BAND_TOKENS_KEY: band_t},
    )
    geometry = PartitionedTargetBandGeometry(PROTECTED_T, band_t, TEMPORAL, *SOURCE_HW, *TARGET_HW)
    head = geometry.head_t
    low, probe, high = run.calls

    # Low and probe run on the target-sized band layout.
    for call in (low, probe):
        assert call["shapes"][0] == (1, 24, TEMPORAL, *TARGET_HW)
        noise_video, _ = unpack_streams(call["noise"], call["shapes"])
        mask_video, _ = unpack_streams(call["mask"], call["shapes"])
        assert target_band_padding_max_abs(noise_video, geometry) == 0.0
        assert target_band_padding_max_abs(mask_video, geometry) == 0.0
        assert bool((target_band_tail(mask_video, geometry) == 1).all())
        assert bool((mask_video[:, :, PROTECTED_T:head] == 1).all())
        assert torch.count_nonzero(mask_video[:, :, :PROTECTED_T]) == 0
    assert high["shapes"][0] == (1, 24, TEMPORAL, *TARGET_HW)

    # Identity handoff: high resumes the band exactly where low/probe left it.
    low_final, _ = unpack_streams(low["final_state"], low["shapes"])
    high_entry, _ = unpack_streams(high["entry_state"], high["shapes"])
    torch.testing.assert_close(high_entry[:, :, PROTECTED_T:head], low_final[:, :, PROTECTED_T:head], rtol=0, atol=1e-6)

    # The learned provider received a uniform reduced-grid view whose tail is the
    # stored reduced-grid probe prediction.
    assert len(run.upscaler.inputs) == 1
    provider_input = run.upscaler.inputs[0]
    assert provider_input.shape == (1, 24, TEMPORAL, *SOURCE_HW)

    transfer = _events(run.metrics, "partitioned_transfer")[0]
    assert transfer["learned_transfer_performed"] is True
    assert transfer["actual_learned_checkpoint_provider_invoked"] is True
    assert transfer["target_band_identity_state"] is True
    assert transfer["target_band_tokens"] == band_t
    assert transfer["target_band_transfer_start_t"] == head
    assert transfer["suffix_dc_bridge_prefix_t"] == head
    assert transfer["suffix_dc_bridge_boundary"] == "target_band_head"
    assert transfer["frame_gauge_reason"] == "target_band_identity_boundary"

    # Guidance consumes the low trajectory on the uniform reduced grid, as in
    # progressive_low_to_high; the target-band state is recorded through its view.
    runs = run.binding.trajectory.runs
    assert len(runs) == 1
    assert (runs[0].geometry.latent_h, runs[0].geometry.latent_w) == SOURCE_HW
    assert runs[0].samples
    assert all(sample.video_x0.shape == (1, 24, TEMPORAL, *SOURCE_HW) for sample in runs[0].samples)

    low_state = _events(run.metrics, "partitioned_target_band_low_state")
    assert len(low_state) == 1
    assert low_state[0]["clean_padding_max_abs"] == 0.0
    assert low_state[0]["low_probe_video_rows"] == geometry.partitioned_rows
    plan = _events(run.metrics, "partitioned_stage_plan")[0]
    assert plan["native_carrier_grid"] == "target"
    assert plan["partitioned_video_rows"] == geometry.partitioned_rows
    transformer_calls = _events(run.metrics, "partitioned_exact_prefix_transformer")
    low_calls = [call for call in transformer_calls if call["stage"] in {"low", "probe"}]
    assert low_calls and all(call["native_carrier_grid"] == "target" for call in low_calls)
    assert all(call["prefix_t"] == head and call["protected_prefix_t"] == PROTECTED_T for call in low_calls)


@pytest.mark.parametrize("dc_enabled", [True, False])
def test_target_band_dc_selector_controls_only_the_first_tail_token(monkeypatch, dc_enabled):
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 1}
    if not dc_enabled:
        extra[PARTITIONED_SUFFIX_DC_BRIDGE_KEY] = False
    run = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=extra,
    )
    transfer = _events(run.metrics, "partitioned_transfer")[0]
    assert transfer["suffix_dc_bridge_requested"] is dc_enabled
    assert transfer["suffix_dc_bridge_enabled"] is dc_enabled
    assert transfer["suffix_dc_bridge_corrected_tokens"] == (1 if dc_enabled else 0)
    assert transfer["suffix_dc_bridge_policy"] == ("one_token_spatial_mean_v1" if dc_enabled else "disabled")
    if dc_enabled:
        assert transfer["suffix_dc_bridge_delta_rms"] > 0.0
    else:
        assert transfer["suffix_dc_bridge_delta_rms"] == 0.0


def test_target_band_wider_than_the_generated_suffix_fails_before_sampling(monkeypatch):
    with pytest.raises(RuntimeError, match="at least one generated token on the reduced grid"):
        _harness(
            monkeypatch,
            spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
            extra_transformer_options={PARTITIONED_TARGET_BAND_TOKENS_KEY: TEMPORAL - PROTECTED_T},
        )


@pytest.mark.parametrize("control", [PARTITIONED_SPATIAL_STAGE_PROGRESSIVE, PARTITIONED_SPATIAL_STAGE_TARGET_BAND])
def test_learned_continuations_bind_guidance_to_the_actual_handoff_pair(monkeypatch, control):
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 2} if control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND else None
    run = _harness(
        monkeypatch,
        spatial_stage_control=control,
        extra_transformer_options=extra,
        guidance_mode="direction",
    )
    references = _events(run.metrics, "partitioned_handoff_guidance_reference")
    assert len(references) == 1
    assert references[0]["target_shape"] == (1, 24, TEMPORAL, *TARGET_HW)
    assert references[0]["source_shape"] == (1, 24, TEMPORAL, *SOURCE_HW)
    guidance = _events(run.metrics, "guidance")
    assert guidance and all(event["handoff_reference_used"] for event in guidance)


@pytest.mark.parametrize("sampler", ["euler", "res_multistep", "euler_ancestral"])
def test_target_band_identity_through_native_comfy_samplers(monkeypatch, sampler):
    run = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={PARTITIONED_TARGET_BAND_TOKENS_KEY: 2, PARTITIONED_SUFFIX_DC_BRIDGE_KEY: False},
        native_sampler=sampler,
    )
    low, _probe, high = run.calls
    low_end, _ = unpack_streams(low["final_state"], low["shapes"])
    high_start, _ = unpack_streams(high["entry_state"], high["shapes"])
    torch.testing.assert_close(
        high_start[:, :, PROTECTED_T : PROTECTED_T + 2],
        low_end[:, :, PROTECTED_T : PROTECTED_T + 2],
        rtol=0,
        atol=1e-6,
    )
    final, _ = unpack_streams(run.result, run.shapes)
    original, _ = unpack_streams(run.latent_image, run.shapes)
    assert torch.equal(final[:, :, :PROTECTED_T], original[:, :, :PROTECTED_T])
    state = _events(run.metrics, "partitioned_target_band_low_state")[0]
    assert state["clean_padding_max_abs"] == 0.0
    if sampler == "euler_ancestral":
        assert state["raw_padding_max_abs"] > 0.0
    assert run.binding.active_capture is None
    assert run.binding.active_guidance_run is None


@pytest.mark.parametrize(
    "control",
    [PARTITIONED_SPATIAL_STAGE_PROGRESSIVE, PARTITIONED_SPATIAL_STAGE_SAME_GRID, PARTITIONED_SPATIAL_STAGE_TARGET_BAND],
)
def test_sol_history_recognizes_the_block_replacement_in_every_stage(monkeypatch, control):
    """An unbound closure cell would make Sol's history identity opaque for that stage."""
    from sol_h3 import interop
    from sol_h3.partitioned_history import _partitioned_flow_replacement_identity

    from h3_flow_regenerate import partitioned_transformer as transform

    original = transform.partitioned_diffusion_wrapper
    recognized = {}

    def recording_wrapper(executor, *args, **kwargs):
        class Recording:
            class_obj = executor.class_obj

            def __call__(self, *call_args, **call_kwargs):
                options = call_kwargs.get("transformer_options", call_args[3] if len(call_args) > 3 else None)
                patch = options["patches_replace"]["dit"][("double_block", 0)]
                identity = _partitioned_flow_replacement_identity(interop, patch, 0)
                recognized.setdefault(options.get("h3_flow_stage"), set()).add(identity is not None)
                return executor(*call_args, **call_kwargs)

        return original(Recording(), *args, **kwargs)

    monkeypatch.setattr(transform, "partitioned_diffusion_wrapper", recording_wrapper)
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 2} if control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND else None
    _harness(monkeypatch, spatial_stage_control=control, extra_transformer_options=extra)
    assert recognized == {"low": {True}, "probe": {True}, "high": {True}}
