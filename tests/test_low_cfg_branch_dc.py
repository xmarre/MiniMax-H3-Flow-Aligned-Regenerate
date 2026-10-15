import ast
import math
import os
from pathlib import Path

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.low_cfg_branch_dc import (
    capture_low_cfg_branch_dc,
    low_cfg_branch_dc_receipt,
    measure_cfg_branch_dc,
    protected_reference_means,
)

PREFIX_T, TEMPORAL, C = 12, 16, 24
SHAPES = [(1, C, TEMPORAL, 4, 6), (1, 32, 2, 8)]


def _packed(dc_prefix, dc_suffix, seed):
    generator = torch.Generator().manual_seed(seed)
    video = torch.randn(1, C, TEMPORAL, 4, 6, generator=generator)
    video = video - video.mean(dim=(-2, -1), keepdim=True)
    video[:, :, :PREFIX_T] += torch.as_tensor(dc_prefix, dtype=video.dtype).reshape(-1, 1, 1, 1)
    video[:, :, PREFIX_T:] += torch.as_tensor(dc_suffix, dtype=video.dtype).reshape(-1, 1, 1, 1)
    return pack_streams((video, torch.zeros(1, 32, 2, 8)))[0]


def _reference(dc=0.0):
    return protected_reference_means(unpack_streams(_packed(dc, 0.0, 7), SHAPES)[0], PREFIX_T)


def _row(cond, uncond=None, *, reference=None, scale=1.0, uncond_evaluated=None, factors=None):
    guided = uncond + scale * (cond - uncond) if uncond is not None else cond
    evaluated = uncond is not None if uncond_evaluated is None else uncond_evaluated
    args = {
        "input": _packed(0.0, 0.0, 9),
        "cond_denoised": cond,
        "uncond_denoised": uncond if uncond is not None else torch.zeros_like(cond),
        "uncond": [{}] if evaluated else None,
        "denoised": guided,
        "cond_scale": scale,
        "sigma": torch.tensor([0.95]),
    }
    return measure_cfg_branch_dc(
        args, SHAPES, PREFIX_T, _reference() if reference is None else reference, preview_factors=factors
    )


def _step(entry):
    return torch.tensor(entry["anchored_step"])


def test_branch_step_is_anchored_on_the_protected_reference_not_the_discarded_prefix_prediction():
    seam_free = _row(_packed(-1.0, 0.0, 2))
    assert seam_free["cond"]["anchored_step_rms"] < 1e-5
    assert seam_free["cond"]["predicted_prefix_step_rms"] == pytest.approx(1.0, rel=1e-4)

    seam = _row(_packed(-1.0, -1.0, 3))
    torch.testing.assert_close(_step(seam["cond"]), torch.full((C,), -1.0), atol=1e-5, rtol=0)
    assert seam["cond"]["predicted_prefix_step_rms"] < 1e-5


def test_skipped_unconditional_branch_is_not_reported_as_a_prediction():
    skipped = _row(_packed(0.0, -0.4, 2))
    assert skipped["uncond_evaluated"] is False and skipped["uncond"] is None
    assert skipped["uncond_provenance"] == "not_requested"

    factors = torch.full((C, 3), 0.1)
    guided = _row(_packed(0.0, -0.4, 2), _packed(0.0, 0.0, 3), scale=4.0, factors=factors)
    assert guided["uncond_evaluated"] is None and guided["uncond_candidate"] is True
    assert guided["uncond_provenance"] == "candidate_tensor_execution_unverified"
    assert guided["uncond"]["anchored_step_rms"] < 1e-5
    assert guided["guided"]["anchored_step_rms"] == pytest.approx(4 * 0.4, rel=1e-4)
    assert guided["cond"]["preview_luma_anchored_step"] == pytest.approx(-0.4 * C * 0.1, rel=1e-4)


def test_capture_returns_the_prediction_unchanged_and_restores_options():
    def other(args):
        return args["denoised"]

    options = {"sampler_post_cfg_function": [other]}
    rows = []
    reference_video = unpack_streams(_packed(0.0, 0.0, 7), SHAPES)[0]
    cond = _packed(0.0, 1.0, 2)
    args = {"denoised": cond, "cond_denoised": cond, "uncond_denoised": None, "uncond": None, "sigma": 1.0}
    with capture_low_cfg_branch_dc(options, SHAPES, PREFIX_T, reference_video, stage="low", rows=rows):
        hooks = options["sampler_post_cfg_function"]
        assert hooks[0] is other and len(hooks) == 2
        assert hooks[1](args) is cond
    assert options["sampler_post_cfg_function"] == [other]
    receipt = low_cfg_branch_dc_receipt(rows)
    assert receipt["status"] == "measured" and receipt["uncond_candidate_rows"] == 0
    assert receipt["summary"]["uncond"] is None

    bare, failed = {}, []
    with capture_low_cfg_branch_dc(bare, SHAPES, PREFIX_T, torch.zeros(1, C, 4, 4, 6), stage="probe", rows=failed):
        assert bare["sampler_post_cfg_function"][0](args) is cond
    assert "sampler_post_cfg_function" not in bare
    assert low_cfg_branch_dc_receipt(failed)["status"] == "error"
    assert low_cfg_branch_dc_receipt([])["status"] == "not_observed"


@pytest.fixture
def core():
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        pytest.skip("native CFG dispatch oracle runs in source-contract CI with COMFYUI_ROOT")
    root = Path(root)
    env = {"torch": torch, "math": math}

    def load(path, names):
        tree = ast.parse((root / path).read_text())
        nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
        assert {n.name for n in nodes} == set(names)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(root / path), "exec"), env)

    load("comfy/patcher_extension.py", ["copy_nested_dicts"])
    load("comfy/samplers.py", ["cfg_function", "sampling_function", "KSamplerX0Inpaint"])
    return env


class _Patcher:
    @staticmethod
    def scale_latent_inpaint(x, sigma, noise, latent_image, denoise_mask):
        return 0.999 * latent_image + 0.001 * noise


def _native_sample(core, predictions, *, cfg, disable_cfg1=False, observe=True):
    def calc_cond_batch(model, conds, x, timestep, model_options):
        return [
            prediction if cond is not None else torch.zeros_like(x)
            for prediction, cond in zip(predictions, conds, strict=True)
        ]

    core["calc_cond_batch"] = calc_cond_batch
    latent = _packed(0.0, 0.0, 7)
    video_mask = torch.ones(1, C, TEMPORAL, 4, 6)
    video_mask[:, :, :PREFIX_T] = 0
    mask = pack_streams((video_mask, torch.zeros(1, 32, 2, 8)))[0]
    options = {"disable_cfg1_optimization": True} if disable_cfg1 else {}
    rows = []

    class Inner:
        inner_model = _Patcher()

        def __call__(self, x, sigma, model_options, seed):
            return core["sampling_function"](None, x, sigma, [{}], [{}], cfg, model_options=model_options, seed=seed)

    wrapper = core["KSamplerX0Inpaint"](Inner(), torch.tensor([1.0, 0.0]))
    wrapper.latent_image, wrapper.noise = latent, torch.zeros_like(latent)
    reference_video = unpack_streams(latent, SHAPES)[0]
    if observe:
        with capture_low_cfg_branch_dc(options, SHAPES, PREFIX_T, reference_video, stage="low", rows=rows):
            sampler_options = core["copy_nested_dicts"](options)
            out = wrapper(_packed(0.0, 0.0, 4), torch.tensor([0.95]), mask, model_options=sampler_options)
        assert "sampler_post_cfg_function" not in options
    else:
        out = wrapper(_packed(0.0, 0.0, 4), torch.tensor([0.95]), mask, model_options=options)
    return out, rows, latent


def test_native_core_dispatch_anchors_branches_on_the_restored_output(core):
    cond, uncond = _packed(-1.0, -0.5, 2), _packed(-1.0, 0.0, 3)
    out, rows, latent = _native_sample(core, (cond, uncond), cfg=3.0)
    control, _, _ = _native_sample(core, (cond, uncond), cfg=3.0, observe=False)
    assert torch.equal(out, control)
    (row,) = rows
    assert row["status"] == "measured" and row["cond_scale"] == 3.0
    assert row["uncond_evaluated"] is None and row["uncond_candidate"] is True
    out_video, latent_video = unpack_streams(out, SHAPES)[0], unpack_streams(latent, SHAPES)[0]
    emitted = out_video[0, :, PREFIX_T].mean(dim=(-2, -1)) - latent_video[0, :, PREFIX_T - 1].mean(dim=(-2, -1))
    torch.testing.assert_close(_step(row["guided"]), emitted, atol=1e-5, rtol=0)
    torch.testing.assert_close(_step(row["guided"]), torch.full((C,), -1.5), atol=1e-5, rtol=0)
    assert row["guided"]["predicted_prefix_step_rms"] == pytest.approx(0.5, rel=1e-4)


@pytest.mark.parametrize("disable_cfg1", [False, True])
def test_native_core_cfg1_reports_the_unconditional_branch_only_when_evaluated(core, disable_cfg1):
    _out, rows, _latent = _native_sample(
        core, (_packed(0.0, -0.4, 2), _packed(0.0, 0.0, 3)), cfg=1.0, disable_cfg1=disable_cfg1
    )
    (row,) = rows
    assert row["uncond_evaluated"] is (None if disable_cfg1 else False)
    assert row["uncond_candidate"] is disable_cfg1
    assert (row["uncond"] is not None) is disable_cfg1


def test_negative_empty_and_out_of_timestep_window_are_not_evaluated():
    cond, uncond = _packed(0.0, -0.4, 2), _packed(0.0, 0.0, 3)
    base = {
        "input": _packed(0.0, 0.0, 9),
        "cond_denoised": cond,
        "uncond_denoised": torch.zeros_like(cond),
        "denoised": cond,
        "sigma": torch.tensor([0.95]),
        "cond_scale": 4.0,
    }
    for conditioning, reason in (
        ([], "empty_conditioning"),
        ([{"timestep_start": 0.90}], "out_of_timestep_range"),
        ([{"timestep_end": 0.99}], "out_of_timestep_range"),
    ):
        result = measure_cfg_branch_dc({**base, "uncond": conditioning}, SHAPES, PREFIX_T, _reference())
        assert result["uncond_evaluated"] is False
        assert result["uncond_candidate"] is False and result["uncond"] is None
        assert result["uncond_provenance"] == reason
    result = measure_cfg_branch_dc({**base, "uncond": [{}], "uncond_denoised": uncond}, SHAPES, PREFIX_T, _reference())
    assert result["uncond_candidate"] is True and result["uncond_evaluated"] is None


def test_observer_accounting_separates_cfg_dispatch_from_logical_and_spectrum():
    cond = _packed(0.0, -0.4, 2)
    ref = unpack_streams(_packed(0.0, 0.0, 7), SHAPES)[0]
    rows = []
    with capture_low_cfg_branch_dc({}, SHAPES, PREFIX_T, ref, stage="low", rows=rows) as _unused:
        pass
    # Build a real observed hook receipt without mutating any denoised tensor.
    opts = {}
    with capture_low_cfg_branch_dc(opts, SHAPES, PREFIX_T, ref, stage="low", rows=rows):
        opts["sampler_post_cfg_function"][0](
            {
                "input": cond,
                "cond_denoised": cond,
                "uncond": None,
                "denoised": cond,
                "sigma": 0.95,
                "cond_scale": 1.0,
            }
        )
    receipt = low_cfg_branch_dc_receipt(
        rows,
        counter_deltas={
            "sampler_logical_calls_low": 8,
            "transformer_actual_nfe_low": 5,
            "spectrum_forecast_calls_low": 3,
            "sampler_logical_calls_probe": 1,
            "transformer_actual_nfe_probe": 1,
            "spectrum_forecast_calls_probe": 0,
        },
    )
    assert receipt["observation_scope"] == "sampler_post_cfg_function_calls_not_logical_sampler_steps"
    assert receipt["stage_counts"]["low"] == {
        "cfg_hook_observations": 1,
        "measured": 1,
        "logical_sampler_calls": 8,
        "actual_transformer_nfe": 5,
        "spectrum_forecasts": 3,
        "logical_calls_without_cfg_observation": 7,
    }
    assert receipt["stage_counts"]["probe"]["cfg_hook_observations"] == 0
    assert receipt["stage_counts"]["probe"]["logical_sampler_calls"] == 1
