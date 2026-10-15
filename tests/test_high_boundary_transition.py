import ast
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_boundary_transition import (
    HIGH_BOUNDARY_TRANSITION_KEY,
    H3HighBoundaryTransition,
    HighBoundaryTransitionRequest,
    prepare_high_boundary_transition,
)
from h3_flow_regenerate.runtime import _merge_preserved_noise

PREFIX_T, TEMPORAL, H, W = 12, 17, 4, 6
SIGMA, NOISE_SCALE = 0.878, 1.0


def _noise_argument(state, sigma, clean):
    return (state - (1.0 - sigma) * clean) / (sigma * NOISE_SCALE)


def _case(tokens=2):
    generator = torch.Generator().manual_seed(3)
    exact = torch.randn(1, 24, PREFIX_T, H, W, generator=generator)
    pre_high = torch.randn(1, 24, TEMPORAL, H, W, generator=generator)
    pre_high[:, :, :PREFIX_T] = exact
    epsilon = torch.randn(pre_high.shape, generator=generator)
    caller_video = torch.zeros_like(pre_high)
    caller_video[:, :, :PREFIX_T] = exact
    caller_audio = torch.randn(1, 32, 2, 8, generator=generator)
    latent_image, shapes = pack_streams((caller_video, caller_audio))
    shapes = list(shapes)
    state_video = (1.0 - SIGMA) * pre_high + SIGMA * epsilon
    state_audio = torch.randn(caller_audio.shape, generator=generator)
    target_raw = pack_streams((state_video, state_audio))[0]
    video_mask = torch.ones_like(caller_video)
    video_mask[:, :, :PREFIX_T] = 0
    mask = pack_streams((video_mask, torch.ones_like(caller_audio)))[0]
    preserved = torch.randn(latent_image.shape, generator=generator)
    target_noise = _merge_preserved_noise(_noise_argument(target_raw, SIGMA, latent_image), preserved, mask)
    return SimpleNamespace(
        request=HighBoundaryTransitionRequest(tokens),
        latent_image=latent_image,
        target_raw=target_raw,
        target_noise=target_noise,
        mask=mask,
        preserved=preserved,
        clean=pre_high[:, :, PREFIX_T : PREFIX_T + tokens].clone(),
        pre_high=pre_high,
        epsilon=epsilon,
        shapes=shapes,
    )


def _prepare(case, **overrides):
    arguments = dict(
        latent_image=case.latent_image,
        target_raw=case.target_raw,
        target_noise=case.target_noise,
        denoise_mask=case.mask,
        preserved_noise=case.preserved,
        transition_clean=case.clean,
        prefix_t=PREFIX_T,
        sigma=SIGMA,
        noise_scale=NOISE_SCALE,
        shapes=case.shapes,
        process_latent_in=lambda value: value,
        noise_argument=_noise_argument,
        merge_preserved_noise=_merge_preserved_noise,
    )
    arguments.update(overrides)
    return prepare_high_boundary_transition(case.request, **arguments)


@pytest.mark.parametrize("tokens", [1, 2])
def test_transition_owns_clean_and_mask_without_changing_the_entry_state(tokens):
    case = _case(tokens)
    caller = case.latent_image.clone()
    mask = case.mask.clone()
    latent, noise, graded, receipt = _prepare(case)
    assert torch.equal(case.latent_image, caller) and torch.equal(case.mask, mask)
    latent_video, latent_audio = unpack_streams(latent, case.shapes)
    caller_video, caller_audio = unpack_streams(caller, case.shapes)
    stop = PREFIX_T + tokens
    assert torch.equal(latent_video[:, :, PREFIX_T:stop], case.clean)
    assert torch.equal(latent_video[:, :, :PREFIX_T], caller_video[:, :, :PREFIX_T])
    assert torch.equal(latent_video[:, :, stop:], caller_video[:, :, stop:])
    assert torch.equal(latent_audio, caller_audio)
    graded_video, graded_audio = unpack_streams(graded, case.shapes)
    assert torch.equal(graded_audio, unpack_streams(mask, case.shapes)[1])
    assert bool((graded_video[:, :, :PREFIX_T] == 0).all()) and bool((graded_video[:, :, stop:] == 1).all())
    for offset, weight in enumerate(case.request.weights):
        assert bool((graded_video[:, :, PREFIX_T + offset] == weight).all())
    # Core starts the sampler at sigma * noise + (1 - sigma) * latent_image.
    start = SIGMA * NOISE_SCALE * noise + (1.0 - SIGMA) * latent
    old_start = SIGMA * NOISE_SCALE * case.target_noise + (1.0 - SIGMA) * caller
    torch.testing.assert_close(unpack_streams(start, case.shapes)[0], unpack_streams(old_start, case.shapes)[0])
    torch.testing.assert_close(
        unpack_streams(noise, case.shapes)[0][:, :, PREFIX_T:stop], case.epsilon[:, :, PREFIX_T:stop]
    )
    assert receipt["applied"] is True and receipt["mask_weights"] == list(case.request.weights)
    assert receipt["effective_sigma"] == pytest.approx([SIGMA * w for w in case.request.weights])


def test_transition_refuses_non_binary_masks_and_non_identity_latent_in():
    case = _case()
    video, audio = unpack_streams(case.mask, case.shapes)
    soft = video.clone()
    soft[:, :, PREFIX_T] = 0.5
    with pytest.raises(RuntimeError, match="binary"):
        _prepare(case, denoise_mask=pack_streams((soft, audio))[0])
    with pytest.raises(RuntimeError, match="identity"):
        _prepare(case, process_latent_in=lambda value: value * 2)
    with pytest.raises(RuntimeError, match="geometry"):
        _prepare(case, transition_clean=case.clean[:, :, :1])


def test_node_is_off_by_default_and_publishes_only_when_enabled():
    spec = H3HighBoundaryTransition.INPUT_TYPES()["required"]
    assert spec["enabled"][1]["default"] is False
    model = SimpleNamespace(model_options={})
    model.clone = lambda: SimpleNamespace(model_options=dict(model.model_options))
    assert H3HighBoundaryTransition().patch(model, False, 2)[0] is model
    (patched,) = H3HighBoundaryTransition().patch(model, True, 2)
    assert patched.model_options[HIGH_BOUNDARY_TRANSITION_KEY] == HighBoundaryTransitionRequest(2)
    assert HIGH_BOUNDARY_TRANSITION_KEY not in model.model_options
    with pytest.raises(ValueError):
        HighBoundaryTransitionRequest(3)


@pytest.fixture
def native():
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        pytest.skip("native Core inpaint oracle runs in source-contract CI with COMFYUI_ROOT")
    root = Path(root)
    env = {"torch": torch}
    path = root / "comfy/samplers.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "KSamplerX0Inpaint"
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    path = root / "comfy/model_base.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3")
    names = {"_pool_masks_to_token_grid", "_token_grid_masks", "_denoise_mask_values", "scale_latent_inpaint"}
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name in names]
    env["utils"] = SimpleNamespace(unpack_latents=unpack_streams, pack_latents=pack_streams)
    env["comfy"] = SimpleNamespace(
        ldm=SimpleNamespace(minimax=SimpleNamespace(model=SimpleNamespace(VISUAL_COND_TIMESTEP=0.999)))
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    path = root / "comfy/ldm/minimax/model.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == "mask_row_values"
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    base = env["MiniMaxH3"]()
    base.audio_scale = lambda: 1.0
    base.diffusion_model = SimpleNamespace(patch_size=(1, 2, 2))
    return SimpleNamespace(inpaint=env["KSamplerX0Inpaint"], h3=base, mask_row_values=env["mask_row_values"])


@pytest.mark.parametrize("tokens", [1, 2])
def test_native_core_inpaint_labels_and_restore_follow_one_graded_mask(native, tokens):
    case = _case(tokens)
    latent, noise, graded, _ = _prepare(case)
    native.h3.latent_shapes = case.shapes
    # Core labels H3 rows from its 1/256 token-grid mask; the sampler mask must already be on that grid.
    assert torch.equal(native.h3._token_grid_masks(graded, case.shapes)[0], unpack_streams(graded, case.shapes)[0])
    condition = native.h3._denoise_mask_values(graded, case.shapes)["denoise_mask"]
    rows = native.mask_row_values(condition[0, 0].float(), TEMPORAL, H, W).reshape(TEMPORAL, -1)
    labels = (1.0 - rows * SIGMA).clamp(max=0.999)
    assert torch.allclose(labels[:PREFIX_T], torch.full_like(labels[:PREFIX_T], 0.999))
    for offset, weight in enumerate(case.request.weights):
        assert torch.allclose(labels[PREFIX_T + offset], torch.full_like(labels[0], 1.0 - weight * SIGMA))
    assert torch.allclose(labels[PREFIX_T + tokens :], torch.full_like(labels[PREFIX_T + tokens :], 1.0 - SIGMA))

    seen = {}
    out = torch.randn(latent.shape, generator=torch.Generator().manual_seed(9))

    class Model:
        inner_model = native.h3

        def __call__(self, x, sigma, model_options, seed):
            seen["x"] = x.clone()
            return out

    model = native.inpaint(Model(), torch.tensor([SIGMA, 0.0]))
    model.latent_image = latent
    model.noise = noise
    start = SIGMA * NOISE_SCALE * noise + (1.0 - SIGMA) * latent
    denoised = model(start, torch.tensor([SIGMA]), graded)
    x_video = unpack_streams(seen["x"], case.shapes)[0]
    out_video = unpack_streams(denoised, case.shapes)[0]
    raw_out = unpack_streams(out, case.shapes)[0]
    noise_video = unpack_streams(noise, case.shapes)[0]
    for offset, weight in enumerate(case.request.weights):
        t = PREFIX_T + offset
        clean, epsilon = case.clean[:, :, offset], noise_video[:, :, t]
        # The row enters the model at its labelled sigma (weight * sigma), up to Core's 0.001 visual-condition noise.
        expected = (1.0 - weight * SIGMA) * clean + weight * SIGMA * epsilon
        expected = expected + (1.0 - weight) * 0.001 * (epsilon - clean)
        torch.testing.assert_close(x_video[:, :, t], expected, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(out_video[:, :, t], weight * raw_out[:, :, t] + (1.0 - weight) * clean)
    stop = PREFIX_T + tokens
    torch.testing.assert_close(out_video[:, :, stop:], raw_out[:, :, stop:])
