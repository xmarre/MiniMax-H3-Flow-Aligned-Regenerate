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
from h3_flow_regenerate.partitioned_band import (
    TARGET_BAND_HANDOFF_POLICY,
    target_band_padding_max_abs,
    target_band_tail,
)
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
    owner._vdn_partitioned_domain_stream_api = 1
    return owner


def _harness(
    monkeypatch,
    *,
    spatial_stage_control,
    extra_transformer_options=None,
    guidance_mode="off",
    native_sampler=None,
    residual_mode="off",
    witness_directory=None,
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
    if witness_directory is not None:
        from h3_flow_regenerate.boundary_witness import WITNESS_DIRECTORY_OPTION

        guider.model_options[WITNESS_DIRECTORY_OPTION] = str(witness_directory)
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
        frame_gauge_residual_mode=residual_mode,
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


def test_target_band_dense_head_reaches_every_high_core_block_and_cleans_up(monkeypatch):
    from h3_flow_regenerate import partitioned_transformer as transform
    from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY

    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    vdn_runtime = pytest.importorskip("vdn_h3.partitioned_runtime")

    observed = []
    native_options = transform._partitioned_transformer_options

    def options(*args, **kwargs):
        result = native_options(*args, **kwargs)
        runtime = result[PARTITIONED_STAGE_KEY]
        stage = result["h3_flow_stage"]
        if stage == "high":
            # A local group after the protected boundary still contains the
            # last free band token; VDN must keep that entire group dense.
            group = SimpleNamespace(query_prefix_domain=False, query_frames=(9, 10, 11))
            dense, _, _ = vdn_runtime._partitioned_local_force_dense(
                group,
                runtime.softmax_diagnostic,
                prefix_t=runtime.plan.prefix_t,
                attention_head_t=runtime.attention_head_t,
            )
            observed.append((runtime.plan.prefix_t, runtime.attention_head_t, dense))
        else:
            assert runtime.attention_head_t is None
        return result

    monkeypatch.setattr(transform, "_partitioned_transformer_options", options)
    run = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={PARTITIONED_TARGET_BAND_TOKENS_KEY: 2},
    )
    assert len(observed) == 2 * run.metrics.counters["transformer_actual_nfe_high"]
    assert set(observed) == {(PROTECTED_T, PROTECTED_T + 2, True)}
    high_call = next(call for call in run.calls if call["stage"] == "high")
    video_mask, _ = unpack_streams(high_call["mask"], high_call["shapes"])
    assert torch.count_nonzero(video_mask[:, :, :PROTECTED_T]) == 0
    assert bool((video_mask[:, :, PROTECTED_T:] == 1).all())
    assert PARTITIONED_STAGE_KEY not in run.guider.model_options["transformer_options"]
    assert _events(run.metrics, "partitioned_exact_prefix_complete")[0]["final_prefix_exact"] is True


def test_band_destination_contract_reaches_core_and_refuses_missing_vdn_work(monkeypatch):
    """Core/Sol fixture verifies transport; its unpatched VDN branch must fail verification."""
    from h3_flow_regenerate import partitioned_transformer as transform
    from h3_flow_regenerate.partitioned_diagnostics import (
        PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
        PARTITIONED_VDN_TEMPORAL_CARRIER_KEY,
    )

    make_owner = _vdn_owner

    def capable_owner():
        owner = make_owner()
        owner._vdn_partitioned_temporal_carrier_api = 1
        owner._vdn_partitioned_temporal_carrier_policies = (PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,)
        owner._vdn_partitioned_temporal_carrier_short_conv_spec = "vdn_solve_short_conv_v1|test"
        return owner

    monkeypatch.setitem(globals(), "_vdn_owner", capable_owner)
    publish = transform._partitioned_transformer_options
    observed = []

    def capture(*args, **kwargs):
        result = publish(*args, **kwargs)
        runtime = kwargs["runtime"]
        contract = result[PARTITIONED_VDN_TEMPORAL_CARRIER_KEY]
        observed.append((runtime.target_band.head_t, runtime.plan.prefix_t, contract))
        return result

    monkeypatch.setattr(transform, "_partitioned_transformer_options", capture)
    with pytest.raises(RuntimeError, match="no verified cross-grid carrier work"):
        _harness(
            monkeypatch,
            spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
            extra_transformer_options={
                PARTITIONED_TARGET_BAND_TOKENS_KEY: 2,
                PARTITIONED_VDN_TEMPORAL_CARRIER_KEY: PARTITIONED_VDN_TEMPORAL_CARRIER_DESTINATION,
            },
        )
    assert observed
    assert all(head == PROTECTED_T + 2 and protected == PROTECTED_T for head, protected, _ in observed)
    assert len({contract["numerical_digest"] for _, _, contract in observed}) == 1


@pytest.mark.parametrize("capture_mode", ["measure", "node", "environment"])
def test_band_stage_evidence_preserves_output_and_keeps_provider_and_band_ownership(
    monkeypatch, tmp_path, capture_mode
):
    import runpy
    import sys
    from pathlib import Path

    import h3_flow_regenerate.partitioned_scheduler as scheduler

    monkeypatch.setattr(sys.modules[__name__], "PROTECTED_T", 12)
    monkeypatch.setattr(sys.modules[__name__], "TEMPORAL", 22)
    exported = []
    native_export = scheduler.export_residual_geometry_evidence

    def export(tensors, **kwargs):
        exported.append(({name: value.clone() for name, value in tensors.items()}, kwargs))
        return native_export(tensors, **kwargs)

    monkeypatch.setattr(scheduler, "export_residual_geometry_evidence", export)
    monkeypatch.setenv("H3_FLOW_BOUNDARY_WITNESS_DIR", str(tmp_path / "witness"))
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 4}
    control = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=extra,
        guidance_mode="direction",
        witness_directory="",
    )
    import folder_paths

    monkeypatch.setattr(folder_paths, "get_output_directory", lambda: str(tmp_path))
    measured = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=extra,
        guidance_mode="direction",
        residual_mode="measure" if capture_mode == "measure" else "off",
        witness_directory=str(tmp_path / "witness")
        if capture_mode == "node"
        else ("" if capture_mode == "measure" else None),
    )
    assert torch.equal(measured.result, control.result)
    assert [call["stage"] for call in measured.calls] == [call["stage"] for call in control.calls]
    assert measured.metrics.counters["transformer_actual_nfe"] == control.metrics.counters["transformer_actual_nfe"]
    assert len(exported) == 1
    tensors, args = exported[0]
    metadata = args["metadata"]
    assert metadata["target_band_tokens"] == 4
    assert metadata["target_band_transfer_start_t"] == 16
    assert metadata["full_video_snapshots"] is True
    assert metadata["first_high_actual"] is True
    assert metadata["capture_boundary_witness_requested"] is (capture_mode != "measure")
    assert metadata["provider_clean_provenance"] == "actual_learned_provider_before_target_band_splice"
    provider = tensors["provider_native_clean_full"]
    source = tensors["source_probe_clean_full"]
    assert torch.equal(source, measured.upscaler.inputs[0])
    expected_provider = resize_spatial_5d(source, *TARGET_HW, mode="bicubic").to(provider)
    assert torch.equal(provider, expected_provider)
    band_probe = tensors["low_probe_native_carrier_clean_full"]
    before_high = tensors["pre_high_exact_restored_full"]
    # The band keeps its own target-grid prediction; the transfer is diagnostic only.
    assert torch.equal(before_high[:, :, PROTECTED_T:16], band_probe[:, :, PROTECTED_T:16])
    assert not torch.equal(provider[:, :, PROTECTED_T:16], band_probe[:, :, PROTECTED_T:16])
    affine = _events(measured.metrics, "partitioned_target_band_same_frame_affine")
    if capture_mode == "measure":
        assert len(affine) == 1 and [frame["frame"] for frame in affine[0]["frames"]] == [12, 13, 14, 15]
        assert affine[0]["output_mutated"] is False
    else:
        assert not affine
    assert not _events(control.metrics, "partitioned_target_band_same_frame_affine")
    trajectories = _events(measured.metrics, "partitioned_target_band_tail_trajectory")
    expected_trajectories = [
        (stage, roi)
        for stage in ("source_low", "provider_native", "pre_high", "final_post_high")
        for roi in ("upper45", "full")
    ]
    assert [(event["stage"], event["roi"]) for event in trajectories] == (
        expected_trajectories if capture_mode == "measure" else []
    )
    if trajectories:
        assert all(event["boundary_t"] == 16 and event["output_mutated"] is False for event in trajectories)
        assert trajectories[0]["grid"] == "source"
        assert all(event["grid"] == "target" for event in trajectories[2:])
    assert not _events(control.metrics, "partitioned_target_band_tail_trajectory")
    assert torch.equal(before_high[:, :, 17:], provider[:, :, 17:])
    assert tensors["first_high_before_flow_full"].shape[2] == TEMPORAL
    assert not torch.equal(tensors["first_high_before_flow_full"], tensors["first_high_after_flow_full"])
    final, _ = unpack_streams(measured.result, measured.shapes)
    assert torch.equal(tensors["final_post_high_internal_clean_full"], final)
    assert measured.binding.high_boundary_trace is None
    receipt = _events(measured.metrics, "partitioned_boundary_window_evidence")[-1]
    assert receipt["capture_boundary_witness_requested"] is (capture_mode != "measure")
    assert receipt["requested_mode"] == ("measure" if capture_mode == "measure" else "off")
    replay = runpy.run_path(str(Path(__file__).parents[1] / "tools" / "decode_native_boundary_evidence.py"))
    manifest, window_values = replay["load_bundle"](tmp_path / receipt["bundle"])
    assert set(window_values) == replay["TENSOR_NAMES"]
    assert manifest["metadata"]["target_band_transfer_start_t"] == 16
    assert torch.equal(window_values["provider_native_clean"], provider[:, :, 10:17])


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


def _record_band_clean(monkeypatch):
    """Capture the band's clean prediction as the scheduler splits the low/probe result."""
    import h3_flow_regenerate.partitioned_scheduler as scheduler

    records = []
    native = scheduler._target_band_source_views

    def recording(*args, **kwargs):
        result = native(*args, **kwargs)
        records.append(result[2].detach().clone())
        return result

    monkeypatch.setattr(scheduler, "_target_band_source_views", recording)
    return records


def _independent_handoff_noise(shape):
    from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig, deterministic_video_noise

    return deterministic_video_noise(
        shape,
        seed=5 + ProgressiveTargetInputConfig(source_scale=0.5).seed_offset,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_target_band_renoises_its_native_band_with_the_tail_noise(monkeypatch):
    band_t = 2
    band_clean = _record_band_clean(monkeypatch)
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

    # The band enters the high stage as its own clean prediction re-noised with
    # the same independent noise as the tail; its raw low state is dropped.
    low_final, _ = unpack_streams(low["final_state"], low["shapes"])
    high_entry, _ = unpack_streams(high["entry_state"], high["shapes"])
    assert len(band_clean) == 1
    sigma = float(high["sigmas"][0])
    noise = _independent_handoff_noise(tuple(high_entry.shape))
    expected_band = (1.0 - sigma) * band_clean[0] + sigma * noise[:, :, PROTECTED_T:head]
    torch.testing.assert_close(high_entry[:, :, PROTECTED_T:head], expected_band, rtol=0, atol=1e-5)
    assert not torch.allclose(high_entry[:, :, PROTECTED_T:head], low_final[:, :, PROTECTED_T:head], atol=1e-3)
    # The tail beyond the DC-bridged first token is the transfer re-noised with the same noise field.
    provider = resize_spatial_5d(run.upscaler.inputs[0], *TARGET_HW, mode="bicubic")
    expected_tail = (1.0 - sigma) * provider[:, :, head + 1 :] + sigma * noise[:, :, head + 1 :]
    torch.testing.assert_close(high_entry[:, :, head + 1 :], expected_tail, rtol=0, atol=1e-5)

    # The learned provider received a uniform reduced-grid view whose tail is the
    # stored reduced-grid probe prediction.
    assert len(run.upscaler.inputs) == 1
    provider_input = run.upscaler.inputs[0]
    assert provider_input.shape == (1, 24, TEMPORAL, *SOURCE_HW)

    transfer = _events(run.metrics, "partitioned_transfer")[0]
    assert transfer["learned_transfer_performed"] is True
    assert transfer["actual_learned_checkpoint_provider_invoked"] is True
    assert transfer["target_band_handoff_policy"] == TARGET_BAND_HANDOFF_POLICY
    assert transfer["target_band_raw_state_carried"] is False
    assert transfer["clean_video_postprocess"]["result"] == "target_band_native_splice"
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

    overlap = _events(run.metrics, "partitioned_target_band_overlap")
    assert len(overlap) == 1
    assert overlap[0]["policy"] == TARGET_BAND_HANDOFF_POLICY
    assert len(overlap[0]["delta_rms_per_frame"]) == band_t
    assert overlap[0]["delta_rms"] > 0.0 and overlap[0]["native_rms"] > 0.0
    assert overlap[0]["output_mutated"] is False

    stages = ["source_low", "provider_native", "pre_high", "final_post_high"]
    assert not _events(run.metrics, "partitioned_target_band_tail_trajectory")
    seams = _events(run.metrics, "partitioned_target_band_tail_seam")
    assert [event["stage"] for event in seams] == stages
    for event in seams:
        assert {"before_seam_rms", "boundary_seam_rms", "after_seam_rms"} <= set(event)
    # The provider stage measures the raw transfer, not the spliced handoff clean.
    from h3_flow_regenerate.seam_diagnostics import measure_video_boundary

    raw_provider_seam = measure_video_boundary(provider, head)
    assert seams[1]["boundary_seam_rms"] == pytest.approx(raw_provider_seam["seam_rms"], rel=1e-6)
    # The prefix-boundary trajectory receipts consumed by the runtime gate are unchanged.
    assert {event["stage"] for event in _events(run.metrics, "partitioned_multiframe_trajectory")} >= {
        "source_low_native",
        "learned_native",
        "exact_restored_pre_high",
        "final_post_high",
    }

    low_state = _events(run.metrics, "partitioned_target_band_low_state")
    assert len(low_state) == 1
    assert low_state[0]["band_raw_state_carried"] is False
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
def test_target_band_handoff_through_native_comfy_samplers(monkeypatch, sampler):
    band_clean = _record_band_clean(monkeypatch)
    run = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={PARTITIONED_TARGET_BAND_TOKENS_KEY: 2, PARTITIONED_SUFFIX_DC_BRIDGE_KEY: False},
        native_sampler=sampler,
    )
    low, _probe, high = run.calls
    low_end, _ = unpack_streams(low["final_state"], low["shapes"])
    high_start, _ = unpack_streams(high["entry_state"], high["shapes"])
    # Every generated token, band included, enters as a re-noised clean operand.
    assert len(band_clean) == 1
    sigma = float(high["sigmas"][0])
    band = slice(PROTECTED_T, PROTECTED_T + 2)
    renoised = (1.0 - sigma) * band_clean[0] + sigma * _independent_handoff_noise(tuple(high_start.shape))[:, :, band]
    torch.testing.assert_close(high_start[:, :, band], renoised.to(high_start), rtol=0, atol=1e-5)
    assert not torch.allclose(high_start[:, :, band], low_end[:, :, band], atol=1e-3)
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
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    interop = pytest.importorskip("sol_h3.interop")
    history = pytest.importorskip("sol_h3.partitioned_history")

    from h3_flow_regenerate import partitioned_transformer as transform

    original = transform.partitioned_diffusion_wrapper
    recognized = {}

    def recording_wrapper(executor, *args, **kwargs):
        class Recording:
            class_obj = executor.class_obj

            def __call__(self, *call_args, **call_kwargs):
                options = call_kwargs.get("transformer_options", call_args[3] if len(call_args) > 3 else None)
                patch = options["patches_replace"]["dit"][("double_block", 0)]
                identity = history._partitioned_flow_replacement_identity(interop, patch, 0)
                recognized.setdefault(options.get("h3_flow_stage"), set()).add(identity is not None)
                return executor(*call_args, **call_kwargs)

        return original(Recording(), *args, **kwargs)

    monkeypatch.setattr(transform, "partitioned_diffusion_wrapper", recording_wrapper)
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 2} if control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND else None
    _harness(monkeypatch, spatial_stage_control=control, extra_transformer_options=extra)
    assert recognized == {"low": {True}, "probe": {True}, "high": {True}}


@pytest.mark.parametrize(
    ("control", "policy"),
    [
        (PARTITIONED_SPATIAL_STAGE_PROGRESSIVE, "source_residual_dense_drift_v2"),
        (PARTITIONED_SPATIAL_STAGE_TARGET_BAND, "independent"),
    ],
)
def test_target_band_tail_uses_independent_handoff_noise_with_frame_gauge_repair(monkeypatch, control, policy):
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 2} if control == PARTITIONED_SPATIAL_STAGE_TARGET_BAND else None
    run = _harness(monkeypatch, spatial_stage_control=control, extra_transformer_options=extra)
    noise = _events(run.metrics, "partitioned_handoff_noise")
    assert [event["policy"] for event in noise] == [policy]
    provenance = _events(run.metrics, "partitioned_handoff_residual_provenance")
    assert len(provenance) == (0 if policy == "independent" else 1)


@pytest.mark.parametrize("sampler", [None, "euler", "euler_ancestral"])
def test_target_band_raw_carry_replaces_only_the_band_entry_state(monkeypatch, sampler):
    from h3_flow_regenerate.partitioned_band import TARGET_BAND_RAW_CARRY_HANDOFF_POLICY
    from h3_flow_regenerate.partitioned_diagnostics import (
        PARTITIONED_TARGET_BAND_HANDOFF_STATE_CARRY,
        PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY,
    )

    band_t = 2
    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: band_t}
    baseline = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=extra,
        native_sampler=sampler,
    )
    carried = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={
            **extra,
            PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY: PARTITIONED_TARGET_BAND_HANDOFF_STATE_CARRY,
        },
        native_sampler=sampler,
    )
    band = slice(PROTECTED_T, PROTECTED_T + band_t)
    assert [call["stage"] for call in carried.calls] == ["low", "probe", "high"]
    # Low/probe are untouched by the handoff control.
    for base_call, carry_call in zip(baseline.calls[:2], carried.calls[:2], strict=True):
        assert torch.equal(base_call["entry_state"], carry_call["entry_state"])
    assert torch.equal(baseline.calls[0]["final_state"], carried.calls[0]["final_state"])
    low_final, _ = unpack_streams(carried.calls[0]["final_state"], carried.shapes)
    base_entry_video, base_entry_audio = unpack_streams(baseline.calls[2]["entry_state"], baseline.shapes)
    carry_entry_video, carry_entry_audio = unpack_streams(carried.calls[2]["entry_state"], carried.shapes)
    # The band resumes its actual low/probe state at the handoff sigma. The
    # scheduler-assembled state is byte-identical (receipt below); the sampler
    # re-forms it from its noise argument, which rounds at float32 precision.
    torch.testing.assert_close(carry_entry_video[:, :, band], low_final[:, :, band], rtol=0, atol=1e-5)
    assert (base_entry_video[:, :, band] - low_final[:, :, band]).abs().max() > 1e-2
    # Prefix, tail (including the DC-bridged first tail token) and audio are byte-identical.
    assert torch.equal(carry_entry_video[:, :, :PROTECTED_T], base_entry_video[:, :, :PROTECTED_T])
    assert torch.equal(carry_entry_video[:, :, band.stop :], base_entry_video[:, :, band.stop :])
    assert torch.equal(carry_entry_audio, base_entry_audio)
    assert not torch.equal(carry_entry_video[:, :, band], base_entry_video[:, :, band])
    # The high mask still protects only the original prefix.
    video_mask, _ = unpack_streams(carried.calls[2]["mask"], carried.shapes)
    assert torch.count_nonzero(video_mask[:, :, :PROTECTED_T]) == 0
    assert bool((video_mask[:, :, PROTECTED_T:] == 1).all())
    final, _ = unpack_streams(carried.result, carried.shapes)
    original, _ = unpack_streams(carried.latent_image, carried.shapes)
    assert torch.equal(final[:, :, :PROTECTED_T], original[:, :, :PROTECTED_T])
    transfer = _events(carried.metrics, "partitioned_transfer")[0]
    assert transfer["target_band_handoff_policy"] == TARGET_BAND_RAW_CARRY_HANDOFF_POLICY
    assert transfer["target_band_handoff_state"] == PARTITIONED_TARGET_BAND_HANDOFF_STATE_CARRY
    assert transfer["target_band_raw_state_carried"] is True
    assert transfer["target_band_raw_state_identity"] is True
    low_state = _events(carried.metrics, "partitioned_target_band_low_state")[0]
    assert low_state["band_raw_state_carried"] is True
    base_transfer = _events(baseline.metrics, "partitioned_transfer")[0]
    assert base_transfer["target_band_raw_state_carried"] is False
    assert base_transfer["target_band_raw_state_identity"] is None
    assert _events(carried.metrics, "partitioned_handoff_noise") == _events(
        baseline.metrics, "partitioned_handoff_noise"
    )


def test_target_band_explicit_default_selectors_match_absent_leaves(monkeypatch):
    from h3_flow_regenerate.partitioned_diagnostics import (
        PARTITIONED_TARGET_BAND_CONTEXT_KEY,
        PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
        PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY,
        PARTITIONED_TARGET_BAND_HANDOFF_STATE_RENOISE,
    )

    extra = {PARTITIONED_TARGET_BAND_TOKENS_KEY: 2}
    absent = _harness(
        monkeypatch, spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND, extra_transformer_options=extra
    )
    explicit = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={
            **extra,
            PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY: PARTITIONED_TARGET_BAND_HANDOFF_STATE_RENOISE,
            PARTITIONED_TARGET_BAND_CONTEXT_KEY: PARTITIONED_TARGET_BAND_CONTEXT_MIXED,
        },
    )
    assert torch.equal(absent.result, explicit.result)
    for left, right in zip(absent.calls, explicit.calls, strict=True):
        assert torch.equal(left["entry_state"], right["entry_state"])
        if "final_state" in left:
            assert torch.equal(left["final_state"], right["final_state"])


@pytest.mark.parametrize(
    "leaf",
    ["h3_flow_partitioned_target_band_handoff_state_v1", "h3_flow_partitioned_target_band_context_v1"],
)
def test_target_band_selectors_are_refused_outside_band_mode(monkeypatch, leaf):
    value = "carry_raw_band" if "handoff" in leaf else "domain_uniform_v1"
    with pytest.raises(RuntimeError, match="require spatial_stage_control='progressive_target_band'"):
        _harness(
            monkeypatch,
            spatial_stage_control=PARTITIONED_SPATIAL_STAGE_SAME_GRID,
            extra_transformer_options={leaf: value},
        )


def _domain_extra(**extra):
    from h3_flow_regenerate.partitioned_diagnostics import (
        PARTITIONED_TARGET_BAND_CONTEXT_DOMAIN_UNIFORM,
        PARTITIONED_TARGET_BAND_CONTEXT_KEY,
    )

    return {
        PARTITIONED_TARGET_BAND_TOKENS_KEY: 2,
        PARTITIONED_TARGET_BAND_CONTEXT_KEY: PARTITIONED_TARGET_BAND_CONTEXT_DOMAIN_UNIFORM,
        **extra,
    }


@pytest.fixture
def native_audio_duration(monkeypatch):
    import sys

    # 14 tokens span 47 frames; the native audio/video relation gives 78 latents.
    monkeypatch.setattr(sys.modules[__name__], "AUDIO_T", 78)


@pytest.mark.usefixtures("native_audio_duration")
def test_domain_uniform_low_and_probe_run_both_streams_and_leave_high_unchanged(monkeypatch):
    history = pytest.importorskip("sol_h3.partitioned_history")
    if getattr(history, "PARTITIONED_DOMAIN_STREAM_API", 0) != 1:
        pytest.skip("installed Sol-H3 predates domain-stream history recognition")
    run = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=_domain_extra(),
    )
    assert [call["stage"] for call in run.calls] == ["low", "probe", "high"]
    stages = _events(run.metrics, "partitioned_target_band_domain_stage")
    assert [event["stage"] for event in stages] == ["low", "probe"]
    assert all(event["verified"] and event["blocks_per_call"] == 2 for event in stages)
    transformer_calls = _events(run.metrics, "partitioned_target_band_domain_transformer")
    assert {event["stage"] for event in transformer_calls} == {"low", "probe"}
    plan = _events(run.metrics, "partitioned_target_band_domain_plan")
    assert len(plan) == 1 and plan[0]["high_stage_changed"] is False
    assert 0.0 < plan[0]["projected_noise_variance_min"] <= plan[0]["projected_noise_variance_max"] <= 1.0
    # High keeps its uniform target-grid partitioned path.
    high_events = [
        event for event in _events(run.metrics, "partitioned_exact_prefix_transformer") if event["stage"] == "high"
    ]
    assert high_events and all(event["attention_head_t"] == PROTECTED_T + 2 for event in high_events)
    final, _ = unpack_streams(run.result, run.shapes)
    original, _ = unpack_streams(run.latent_image, run.shapes)
    assert torch.equal(final[:, :, :PROTECTED_T], original[:, :, :PROTECTED_T])
    assert bool(torch.isfinite(final).all())
    assert "h3_flow_partitioned_stage_v1" not in run.guider.model_options["transformer_options"]


@pytest.mark.usefixtures("native_audio_duration")
def test_domain_uniform_changes_only_low_probe_numerics_and_combines_with_raw_carry(monkeypatch):
    from h3_flow_regenerate.partitioned_diagnostics import (
        PARTITIONED_TARGET_BAND_HANDOFF_STATE_CARRY,
        PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY,
    )

    history = pytest.importorskip("sol_h3.partitioned_history")
    if getattr(history, "PARTITIONED_DOMAIN_STREAM_API", 0) != 1:
        pytest.skip("installed Sol-H3 predates domain-stream history recognition")
    mixed = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options={PARTITIONED_TARGET_BAND_TOKENS_KEY: 2},
    )
    domain = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=_domain_extra(),
    )
    both = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=_domain_extra(
            **{PARTITIONED_TARGET_BAND_HANDOFF_STATE_KEY: PARTITIONED_TARGET_BAND_HANDOFF_STATE_CARRY}
        ),
    )
    # The sampler inputs of low are identical; only the transformer context differs.
    assert torch.equal(mixed.calls[0]["entry_state"], domain.calls[0]["entry_state"])
    assert not torch.equal(mixed.calls[0]["final_state"], domain.calls[0]["final_state"])
    assert torch.equal(domain.calls[0]["final_state"], both.calls[0]["final_state"])
    band = slice(PROTECTED_T, PROTECTED_T + 2)
    low_final, _ = unpack_streams(both.calls[0]["final_state"], both.shapes)
    domain_entry, domain_audio = unpack_streams(domain.calls[2]["entry_state"], domain.shapes)
    both_entry, both_audio = unpack_streams(both.calls[2]["entry_state"], both.shapes)
    torch.testing.assert_close(both_entry[:, :, band], low_final[:, :, band], rtol=0, atol=1e-5)
    assert torch.equal(both_entry[:, :, band.stop :], domain_entry[:, :, band.stop :])
    assert torch.equal(both_entry[:, :, :PROTECTED_T], domain_entry[:, :, :PROTECTED_T])
    assert torch.equal(both_audio, domain_audio)
    transfer = _events(both.metrics, "partitioned_transfer")[0]
    assert transfer["target_band_raw_state_carried"] is True


@pytest.mark.usefixtures("native_audio_duration")
@pytest.mark.parametrize(
    ("leaf", "value"),
    [
        ("h3_flow_partitioned_vdn_linear_diagnostic_v1", "suppress_cross_grid_temporal_taps"),
        ("h3_flow_partitioned_softmax_diagnostic_v1", "target_query_sink_measure"),
        ("h3_flow_partitioned_vdn_temporal_carrier_v1", "destination_grid_stencil_v1"),
    ],
)
def test_domain_uniform_refuses_mixed_grid_selectors_without_fallback(monkeypatch, leaf, value):
    with pytest.raises(RuntimeError, match="domain_uniform_v1' requires"):
        _harness(
            monkeypatch,
            spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
            extra_transformer_options=_domain_extra(**{leaf: value}),
        )


@pytest.mark.usefixtures("native_audio_duration")
def test_domain_uniform_requires_paired_vdn_capability_without_fallback(monkeypatch):
    make_owner = _vdn_owner

    def legacy_owner():
        owner = make_owner()
        del owner._vdn_partitioned_domain_stream_api
        return owner

    monkeypatch.setitem(globals(), "_vdn_owner", legacy_owner)
    with pytest.raises(RuntimeError, match=r"refusing target-grid fallback|domain-stream API"):
        _harness(
            monkeypatch,
            spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
            extra_transformer_options=_domain_extra(),
        )


@pytest.mark.usefixtures("native_audio_duration")
def test_sol_history_recognizes_domain_low_probe_and_uniform_high(monkeypatch):
    cli = pytest.importorskip("comfy.cli_args")
    cli.args.cpu = True
    interop = pytest.importorskip("sol_h3.interop")
    history = pytest.importorskip("sol_h3.partitioned_history")
    if getattr(history, "PARTITIONED_DOMAIN_STREAM_API", 0) != 1:
        pytest.skip("installed Sol-H3 predates domain-stream history recognition")
    history.install_partitioned_history_bridge()

    from h3_flow_regenerate import partitioned_transformer as transform

    original = transform.partitioned_diffusion_wrapper
    recognized = {}

    def recording_wrapper(executor, *args, **kwargs):
        class Recording:
            class_obj = executor.class_obj

            def __call__(self, *call_args, **call_kwargs):
                options = call_kwargs.get("transformer_options", call_args[3] if len(call_args) > 3 else None)
                patch = options["patches_replace"]["dit"][("double_block", 0)]
                identity = interop._flow_mixed_grid_replacement_identity(patch, 0)
                kind = None if identity is None else identity[0][0]
                recognized.setdefault(options.get("h3_flow_stage"), set()).add(kind)
                return executor(*call_args, **call_kwargs)

        return original(Recording(), *args, **kwargs)

    monkeypatch.setattr(transform, "partitioned_diffusion_wrapper", recording_wrapper)
    _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=_domain_extra(),
    )
    domain_kind = history.PARTITIONED_DOMAIN_UNIFORM_IDENTITY
    assert recognized == {
        "low": {domain_kind},
        "probe": {domain_kind},
        "high": {history.PARTITIONED_FLOW_IDENTITY},
    }


@pytest.mark.usefixtures("native_audio_duration")
def test_domain_uniform_audio_position_selector_is_inactive(monkeypatch):
    from h3_flow_regenerate.partitioned_diagnostics import PARTITIONED_AUDIO_POSITION_DOMAIN_KEY

    default = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=_domain_extra(),
    )
    selected = _harness(
        monkeypatch,
        spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
        extra_transformer_options=_domain_extra(**{PARTITIONED_AUDIO_POSITION_DOMAIN_KEY: "source_carrier"}),
    )
    assert torch.equal(default.result, selected.result)
    assert not _events(selected.metrics, "partitioned_audio_position_domain_verified")


def test_domain_invalid_audio_fails_before_sampler_lifetime(monkeypatch):
    from h3_flow_regenerate import partitioned_scheduler as scheduler

    def sampling_started(*_args, **_kwargs):
        pytest.fail("unsupported domain input reached the sampler lifetime")

    monkeypatch.setattr(scheduler, "_begin_capture", sampling_started)
    with pytest.raises(RuntimeError, match="native audio/video duration relation"):
        _harness(
            monkeypatch,
            spatial_stage_control=PARTITIONED_SPATIAL_STAGE_TARGET_BAND,
            extra_transformer_options=_domain_extra(),
        )
