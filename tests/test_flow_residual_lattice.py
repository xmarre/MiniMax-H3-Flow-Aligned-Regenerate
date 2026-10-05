import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import (
    H3_HANDOFF_NOISE_DENSE_DRIFT,
    H3_HANDOFF_NOISE_INDEPENDENT,
    H3_HANDOFF_NOISE_SOURCE_RESIDUAL,
    CleanVideoPostprocessResult,
    build_handoff_state,
    refine_h3_flow_residual,
    refine_h3_patch_lattice_residual,
)


def test_gaussian_only_transport_preserves_the_existing_refinement_and_scaled_covariance():
    initial = torch.randn(4096, 1, 1, 4, 4, generator=torch.Generator().manual_seed(1144))
    before = initial.clone()
    scale = 1.7
    expected, _ = refine_h3_patch_lattice_residual(initial, target_h=6, target_w=6, seed=511)
    actual, report = refine_h3_flow_residual(
        initial * scale, initial, target_h=6, target_w=6, seed=511, noise_scale=scale
    )
    assert torch.equal(actual, expected * scale)
    assert torch.equal(initial, before)
    samples = actual[:, 0, 0].flatten(1) / scale
    covariance = torch.cov(samples.T)
    assert samples.mean().abs() < 0.012
    assert (covariance.diag() - 1).abs().max() < 0.065
    off_diagonal = covariance - torch.diag(covariance.diag())
    assert off_diagonal.abs().max() < 0.065
    assert report["gaussian_component_variance_if_initial_standard"] == pytest.approx(scale**2)
    assert report["gaussian_projection_max_abs_error"] < 2e-5
    assert report["coarse_projection_scope"] == "initial_gaussian_component"
    assert "projection_max_abs_error" not in report
    assert "gaussian_marginal_if_source_standard" not in report


@pytest.mark.parametrize("source_hw,target_hw", [((44, 44), (62, 62)), ((46, 40), (66, 58)), ((36, 54), (50, 76))])
@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("phase", [0, 1])
def test_structured_drift_has_one_continuous_peak_instead_of_phase_separated_support(source_hw, target_hw, axis, phase):
    initial = torch.zeros(1, 1, 1, *source_hw)
    source_n, target_n = source_hw[axis], target_hw[axis]
    # Choose an interior coarse group with two target owners to expose the old
    # even/odd duplication. The structured impulse must have no parity holes.
    source_grid, target_grid = source_n // 2, target_n // 2
    owner = next(
        i
        for i in range(source_grid // 3, source_grid - 2)
        if math.ceil((i + 1) * target_grid / source_grid) - math.ceil(i * target_grid / source_grid) == 2
    )
    source_position = 2 * owner + phase
    drift = torch.zeros_like(initial)
    if axis == 0:
        drift[..., source_position, :] = 1
    else:
        drift[..., source_position] = 1
    baseline, _ = refine_h3_flow_residual(initial, initial, target_h=target_hw[0], target_w=target_hw[1], seed=19)
    transported, _ = refine_h3_flow_residual(drift, initial, target_h=target_hw[0], target_w=target_hw[1], seed=19)
    contribution = (transported - baseline)[0, 0, 0]
    profile = contribution[:, target_hw[1] // 2] if axis == 0 else contribution[target_hw[0] // 2]
    # Independently evaluate a unit triangular impulse at native physical dense
    # cell coordinates, rather than calling the production interpolation helper.
    source_step = 32 / math.sqrt(math.prod(source_hw))
    target_step = 32 / math.sqrt(math.prod(target_hw))
    source_origin = 16 * (1 - source_n / math.sqrt(math.prod(source_hw))) - source_step / 2
    target_origin = 16 * (1 - target_n / math.sqrt(math.prod(target_hw))) - target_step / 2
    source_positions = (target_origin + torch.arange(target_n) * target_step - source_origin) / source_step
    expected = (1 - (source_positions - source_position).abs()).clamp_min(0)
    torch.testing.assert_close(profile, expected, rtol=0, atol=1e-5)
    occupied = torch.nonzero(profile > 1e-5).flatten()
    assert occupied.numel() >= 2
    assert occupied[-1] - occupied[0] + 1 == occupied.numel()
    peak = int(profile.argmax())
    assert bool(torch.all(profile[1 : peak + 1] >= profile[:peak] - 1e-6))
    assert bool(torch.all(profile[peak + 1 :] <= profile[peak:-1] + 1e-6))
    old_base, _ = refine_h3_patch_lattice_residual(initial, target_h=target_hw[0], target_w=target_hw[1], seed=19)
    old, _ = refine_h3_patch_lattice_residual(drift, target_h=target_hw[0], target_w=target_hw[1], seed=19)
    old_contribution = (old - old_base)[0, 0, 0]
    old_profile = old_contribution[:, target_hw[1] // 2] if axis == 0 else old_contribution[target_hw[0] // 2]
    old_occupied = torch.nonzero(old_profile > 1e-5).flatten()
    assert old_occupied.numel() == 2
    assert old_occupied[-1] - old_occupied[0] == 2


def test_constant_drift_retains_dc_without_group_size_attenuation():
    initial = torch.randn(1, 3, 2, 8, 12, generator=torch.Generator().manual_seed(4))
    baseline, _ = refine_h3_flow_residual(initial, initial, target_h=12, target_w=16, seed=62)
    target, report = refine_h3_flow_residual(initial + 0.25, initial, target_h=12, target_w=16, seed=62)
    torch.testing.assert_close(target - baseline, torch.full_like(target, 0.25), rtol=0, atol=4e-7)
    assert report["source_drift_rms"] == pytest.approx(0.25)
    assert report["target_drift_rms"] == pytest.approx(0.25)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_same_grid_residual_is_byte_exact_and_does_not_alias_input(dtype):
    source = torch.randn(1, 3, 2, 8, 12, generator=torch.Generator().manual_seed(12)).to(dtype)
    initial = torch.randn_like(source)
    actual, report = refine_h3_flow_residual(source, initial, target_h=8, target_w=12, seed=7, noise_scale=1.7)
    assert torch.equal(actual, source)
    assert actual.dtype == source.dtype
    assert actual.data_ptr() != source.data_ptr()
    assert report["identity"] is True


class DenseProvider:
    api_version = 1
    kind = "minimax_h3_learned_latent_upscaler"
    model_name = "residual_fixture"
    device = "cpu"
    precision = "fp32"
    offload_after_upscale = False

    def __init__(self):
        self.calls = []

    def upscale_clean_video(self, video, *, target_h, target_w):
        self.calls.append(video.clone())
        return torch.full((*video.shape[:-2], target_h, target_w), 2.0, dtype=video.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_dense_handoff_preserves_audio_and_provider_clean_ownership(dtype):
    generator = torch.Generator().manual_seed(11)
    initial = torch.randn(1, 24, 3, 4, 6, generator=generator).to(dtype)
    clean = torch.randn(initial.shape, generator=generator).to(dtype)
    source_residual = 1.7 * initial + 0.25
    sigma = 0.75
    source_video = ((1 - sigma) * clean + sigma * source_residual).to(dtype)
    audio = torch.randn(1, 32, 2, 7, generator=generator).to(dtype)
    state, shapes = pack_streams((source_video, audio))
    x0, _ = pack_streams((clean, torch.zeros_like(audio)))
    provider, receipts, captured = DenseProvider(), {}, []

    def postprocess(value):
        captured.append(value.clone())
        result = value.clone()
        result[:, :, 1:] = 9
        return CleanVideoPostprocessResult(result, protected_prefix_t=1)

    actual, target_shapes = build_handoff_state(
        source_packed_state=state,
        source_x0_packed=x0,
        source_shapes=shapes,
        sigma=sigma,
        target_h=8,
        target_w=10,
        seed=52,
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        clean_video_postprocess=postprocess,
        noise_mode=H3_HANDOFF_NOISE_DENSE_DRIFT,
        initial_source_noise=initial,
        model_noise_scale=1.7,
        transfer_metrics=receipts,
    )
    target_video, target_audio = unpack_streams(actual, target_shapes)
    recovered = (source_video.float() - (1 - sigma) * clean.float()) / sigma
    refined, _ = refine_h3_flow_residual(recovered, initial, target_h=8, target_w=10, seed=52, noise_scale=1.7)
    target_clean = torch.full_like(refined, 2).to(dtype)
    target_clean[:, :, 1:] = 9
    expected = ((1 - sigma) * target_clean + sigma * refined.to(dtype)).to(dtype)
    torch.testing.assert_close(target_video, expected, rtol=0, atol=0)
    assert torch.equal(target_audio, audio)
    assert target_video.dtype == dtype
    assert len(provider.calls) == len(captured) == 1
    assert torch.equal(provider.calls[0], clean)
    assert torch.equal(captured[0], torch.full_like(captured[0], 2))
    report = receipts["handoff_noise"]
    assert report["policy"] == H3_HANDOFF_NOISE_DENSE_DRIFT
    assert report["extra_h3_nfe"] == 0


@pytest.mark.parametrize(
    "invalid", ["missing", "shape", "nonfinite_noise", "nonfinite_state", "scale_zero", "scale_nan"]
)
def test_invalid_dense_residual_operand_fails_before_provider_evaluation(invalid):
    video = torch.ones(1, 24, 2, 4, 6)
    initial = torch.ones_like(video)
    scale = 1.0
    if invalid == "missing":
        initial = None
    elif invalid == "shape":
        initial = initial[:, :, :1]
    elif invalid == "nonfinite_noise":
        initial[..., 0, 0] = float("nan")
    elif invalid == "nonfinite_state":
        video[..., 0, 0] = float("inf")
    elif invalid == "scale_zero":
        scale = 0.0
    else:
        scale = float("nan")
    audio = torch.zeros(1, 32, 2, 7)
    state, shapes = pack_streams((video, audio))
    clean, _ = pack_streams((torch.ones_like(video), audio))
    provider = DenseProvider()
    with pytest.raises(ValueError):
        build_handoff_state(
            source_packed_state=state,
            source_x0_packed=clean,
            source_shapes=shapes,
            sigma=0.5,
            target_h=8,
            target_w=10,
            seed=2,
            transfer_mode="learned_3d",
            learned_upscaler=provider,
            noise_mode=H3_HANDOFF_NOISE_DENSE_DRIFT,
            initial_source_noise=initial,
            model_noise_scale=scale,
        )
    assert provider.calls == []


def sample_res_multistep():
    pass


def sample_euler():
    pass


def sample_er_sde():
    pass


@pytest.mark.parametrize(
    "enabled,source,sampler_function,extra_options,expected_mode",
    [
        (True, "main", sample_res_multistep, {}, H3_HANDOFF_NOISE_DENSE_DRIFT),
        (True, "main", sample_euler, {}, H3_HANDOFF_NOISE_DENSE_DRIFT),
        (True, "main", sample_euler, {"s_churn": 0.5}, H3_HANDOFF_NOISE_SOURCE_RESIDUAL),
        (True, "main", sample_er_sde, {}, H3_HANDOFF_NOISE_SOURCE_RESIDUAL),
        (False, "main", sample_res_multistep, {}, H3_HANDOFF_NOISE_INDEPENDENT),
        (True, "shadow", sample_res_multistep, {}, H3_HANDOFF_NOISE_INDEPENDENT),
    ],
)
def test_actual_scheduler_routing_passes_original_noise_and_model_scale(
    enabled, source, sampler_function, extra_options, expected_mode
):
    # Execute the production selection, provenance and handoff call directly.
    # This checks routing without pretending to rerun the full H3 sampler.
    from h3_flow_regenerate import partitioned_scheduler as scheduler

    tree = ast.parse(Path(scheduler.__file__).read_text())
    contract = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "_dense_drift_sampler_contract"
    )
    assignment = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "handoff_noise_mode" for t in node.targets)
    )
    provenance = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "handoff_noise_mode"
        and any(isinstance(n, ast.Name) and n.id == "source_effective_residual" for n in ast.walk(node))
    )
    handoff = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "build_handoff_state"
    )
    video = torch.randn(1, 24, 3, 4, 6, generator=torch.Generator().manual_seed(13))
    noise = video.clone()
    audio = torch.zeros(1, 32, 2, 7)
    source_raw, shapes = pack_streams((video, audio))
    source_x0, _ = pack_streams((torch.zeros_like(video), audio))
    calls, events = [], []

    def capture(**kwargs):
        calls.append(kwargs)
        return build_handoff_state(**kwargs)

    env = dict(vars(scheduler))
    env.update(
        config=SimpleNamespace(frame_gauge_repair=enabled, seed_offset=7),
        av_handoff_source=scheduler.PARTITIONED_AV_HANDOFF_SOURCE_MAIN if source == "main" else "shadow",
        sampler=SimpleNamespace(sampler_function=sampler_function, extra_options=extra_options),
        source_raw=source_raw,
        source_x0=source_x0,
        source_shapes=shapes,
        source_video_noise=noise,
        sigma=0.75,
        base_model=SimpleNamespace(model_sampling=SimpleNamespace(noise_scale=1.7)),
        stage_plan=SimpleNamespace(prefix_t=1),
        binding=SimpleNamespace(metrics=SimpleNamespace(event=lambda *a, **kw: events.append((a, kw)))),
        effective_upscaler=DenseProvider(),
        target_h=8,
        target_w=10,
        seed=52,
        transfer_metrics={},
        clean_video_postprocess=None,
        spatial_stage_control="learned",
        build_handoff_state=capture,
    )
    program = ast.fix_missing_locations(ast.Module(body=[contract, assignment, provenance, handoff], type_ignores=[]))
    exec(compile(program, scheduler.__file__, "exec"), env)
    assert len(calls) == 1
    assert calls[0]["noise_mode"] == expected_mode
    assert calls[0]["initial_source_noise"] is (noise if expected_mode == H3_HANDOFF_NOISE_DENSE_DRIFT else None)
    assert calls[0]["model_noise_scale"] == (1.7 if expected_mode == H3_HANDOFF_NOISE_DENSE_DRIFT else 1.0)
    assert len(env["effective_upscaler"].calls) == 1
    assert len(events) == (0 if expected_mode == H3_HANDOFF_NOISE_INDEPENDENT else 1)
    if events:
        assert events[0][1]["policy"] == expected_mode
        assert events[0][1]["sampler"] == sampler_function.__name__
        assert events[0][1]["dense_drift_sampler_contract"] == (
            "deterministic_initial_noise_flow"
            if expected_mode == H3_HANDOFF_NOISE_DENSE_DRIFT
            else "stochastic_or_unverified_sampler_gaussian_refinement"
        )
    assert env["transfer_metrics"]["handoff_noise"]["policy"] == expected_mode
