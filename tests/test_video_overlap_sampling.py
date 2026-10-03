"""Exercise prefix closure through pinned Core inpaint and H3 mask functions."""

import ast
import copy
import itertools
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.comfy_compat import _canonicalize_exact_masked_output
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.video_guided_overlap import apply_video_guided_overlap_mask
from h3_flow_regenerate.video_overlap_closure import video_overlap_closure


def _case():
    video = torch.ones(1, 24, 17, 4, 6)
    audio = torch.ones(1, 32, 2, 70)
    latent, shapes = pack_streams((video, audio))
    video_mask, audio_mask = torch.ones_like(video), torch.ones_like(audio)
    video_mask[:, :, :12] = 0
    audio_mask[..., :64] = 0
    return latent, list(shapes), pack_streams((video_mask, audio_mask))[0]


@pytest.fixture
def native():
    root = os.environ.get("COMFYUI_ROOT")
    if not root:
        pytest.skip("native overlap sampling oracle runs in source-contract CI with COMFYUI_ROOT")
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
    path = root / "comfy/model_sampling.py"
    tree = ast.parse(path.read_text())
    nodes = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in {"reshape_sigma", "CONST"}
    ]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    path = root / "comfy/ldm/minimax/model.py"
    node = next(
        n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "MiniMaxH3Model"
    )
    node.bases = []
    node.body = [n for n in node.body if isinstance(n, ast.FunctionDef) and n.name == "forward"]

    class Executor:
        @staticmethod
        def new_class_executor(function, _owner, _wrappers):
            return SimpleNamespace(execute=function)

    env["comfy"].model_prefetch = SimpleNamespace(malloc_graph_enabled=lambda _device: False)
    env["comfy"].patcher_extension = SimpleNamespace(
        WrapperExecutor=Executor,
        get_all_wrappers=lambda *a: [],
        WrappersMP=SimpleNamespace(DIFFUSION_MODEL="diffusion_model"),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return SimpleNamespace(
        inpaint=env["KSamplerX0Inpaint"], h3=env["MiniMaxH3"], forward=env["MiniMaxH3Model"], sampling=env["CONST"]
    )


def _sample(native, *, tokens, close):
    latent, shapes, exact = _case()
    runtime, _ = apply_video_guided_overlap_mask(exact, shapes, tokens=tokens)
    original = runtime.clone()
    noise = torch.zeros_like(latent)
    sigmas = torch.tensor([0.88, 0.84, 0.8, 0.73, 0.63, 0.44, 0.0])
    base = native.h3()
    base.latent_shapes = shapes
    base.audio_scale = lambda: 1.0
    base.diffusion_model = SimpleNamespace(patch_size=(1, 2, 2))
    sampling = native.sampling()
    h3 = native.forward()
    static_conditions = base._denoise_mask_values(runtime, shapes)
    reference_video, _ = unpack_streams(base.scale_latent_inpaint(sigmas[0:1], noise, latent), shapes)
    metrics = H3FlowMetrics()
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    publications = []

    def predict(x, sigma, _concat, _cross, _control, _options, **kwargs):
        video, audio = unpack_streams(x, shapes)
        model_mask = kwargs["denoise_mask"]
        # Core's H3 labels and velocity conversion consume this same condition.
        expected = base._denoise_mask_values(publications[-1], shapes)["denoise_mask"]
        assert torch.equal(model_mask, expected)
        drift = (video[:, :, :12] - reference_video[:, :, :12]).mean()
        clean = torch.ones_like(video)
        # A temporal denoiser may alter the released prefix and propagate the
        # seen context into the first generated token. Final prefix equality
        # alone does not test that dependency.
        clean[:, :, :12] += 1
        clean[:, :, 12] += drift

        def network(streams, timestep, context, options, **conditions):
            del timestep, context, options
            mask = conditions["denoise_mask"]
            # The network velocity uses the per-row sigma represented by its
            # mask condition. Core's real forward performs velocity conversion.
            v_video = torch.where(mask > 0, (streams[0] - clean) / (sigma * mask).clamp(min=1e-6), 0)
            v_audio = (streams[1] - 1) / sigma
            return [v_video, v_audio]

        h3._forward = network
        velocity = h3.forward([video, audio], sigma * 1000, None, **kwargs)
        return sampling.calculate_denoised(sigma, pack_streams(velocity)[0], x)

    class Guider:
        inner_model = base

        def __call__(self, x, sigma, *, model_options, seed):
            del seed
            kwargs = dict(static_conditions)
            wrappers = model_options["transformer_options"].get("wrappers", {}).get("apply_model", {})
            chain = [wrapper for group in wrappers.values() for wrapper in group]

            def invoke(index, *args, **kw):
                if index == len(chain):
                    return predict(*args, **kw)
                return chain[index](lambda *a, **k: invoke(index + 1, *a, **k), *args, **kw)

            return invoke(0, x, sigma, None, None, None, model_options["transformer_options"], **kwargs)

    model = native.inpaint(Guider(), sigmas)
    model.latent_image = latent
    model.noise = noise

    def run(options):
        state = sampling.noise_scaling(sigmas[0], noise, latent)
        for sigma, next_sigma in itertools.pairwise(sigmas):
            # Observe the mask actually returned by Core's public hook, without
            # replacing any native inpaint input/output arithmetic.
            if close:
                hook = options["denoise_mask_function"]

                def observe(*args, hook=hook, **kwargs):
                    value = hook(*args, **kwargs)
                    publications.append(value.clone())
                    return value

                step_options = {**options, "denoise_mask_function": observe}
            else:
                publications.append(runtime.clone())
                step_options = options
            denoised = model(state, sigma[None], runtime, model_options=step_options)
            state = state + (state - denoised) / sigma * (next_sigma - sigma)
        return state

    if close:
        with video_overlap_closure(
            guider, runtime, shapes, prefix_t=12, tokens=tokens, sigmas=sigmas, metrics=metrics
        ) as owner:
            # Core clones model options after building its static conditions.
            copied = copy.deepcopy(guider.model_options)
            result = run(copied)
            assert owner.mask_calls == 6
            assert owner.model_calls == 6
        assert owner.closed and owner.current_mask is None and owner.base_ramp is None
        assert guider.model_options == {"transformer_options": {}}
    else:
        result = run(guider.model_options)
    assert torch.equal(runtime, original)
    return result, latent, shapes, exact, publications, metrics


@pytest.mark.parametrize("tokens", [4, 6, 8, 12, 10**30])
def test_native_sampling_closes_context_before_final_prefix_restoration(native, tokens):
    old, latent, shapes, exact, _, _ = _sample(native, tokens=tokens, close=False)
    new, _, _, _, publications, metrics = _sample(native, tokens=tokens, close=True)
    old, _ = _canonicalize_exact_masked_output(old, latent, exact)
    new, _ = _canonicalize_exact_masked_output(new, latent, exact)
    old_video, _ = unpack_streams(old, shapes)
    new_video, _ = unpack_streams(new, shapes)
    source_video, _ = unpack_streams(latent, shapes)
    assert (old_video[:, :, 12] - source_video[:, :, 12]).abs().max().item() > 0.01
    torch.testing.assert_close(new_video[:, :, 12], source_video[:, :, 12], atol=1e-6, rtol=0)
    for publication in publications[-2:]:
        video, audio = unpack_streams(publication, shapes)
        exact_video, exact_audio = unpack_streams(exact, shapes)
        assert torch.equal(video[:, :, :12], exact_video[:, :, :12])
        assert torch.equal(video[:, :, 12:], exact_video[:, :, 12:])
        assert torch.equal(audio, exact_audio)
    receipt = metrics.events[-1].fields
    assert receipt["applied_tokens"] == min(tokens, 12)
    assert receipt["close_sigma"] == pytest.approx(0.63)
    assert receipt["failed"] is False
    assert receipt["final_prefix_mask_max"] == 0
    assert receipt["exact_model_context_before_completion"] is True


@pytest.mark.parametrize("sigmas", [[0.8, 0.0], [0.8, 0.4, 0.0]])
def test_short_high_schedules_keep_the_context_exact_and_report_zero_release(sigmas):
    _, shapes, exact = _case()
    runtime, _ = apply_video_guided_overlap_mask(exact, shapes, tokens=8)
    guider = SimpleNamespace(model_options={"transformer_options": {}})
    metrics = H3FlowMetrics()
    with video_overlap_closure(
        guider, runtime, shapes, prefix_t=12, tokens=8, sigmas=torch.tensor(sigmas), metrics=metrics
    ) as owner:
        actual = owner.mask(torch.tensor([sigmas[0]]), runtime)
        video, _ = unpack_streams(actual, shapes)
        assert torch.count_nonzero(video[:, :, :12]) == 0
        prior = unpack_streams(runtime, shapes)[0][:1, :1]
        owner.apply_model(lambda *a, **k: None, None, torch.tensor([sigmas[0]]), denoise_mask=prior)
    assert metrics.events[-1].fields["applied_tokens"] == 0
    assert metrics.events[-1].fields["release_enabled"] is False


def test_overlap_closure_cleans_only_its_hooks_and_releases_buffers_after_failure():
    _, shapes, exact = _case()
    runtime, _ = apply_video_guided_overlap_mask(exact, shapes, tokens=8)
    foreign = [lambda *a, **k: None]
    apply = {"foreign": foreign}
    guider = SimpleNamespace(model_options={"transformer_options": {"wrappers": {"apply_model": apply}}})
    metrics = H3FlowMetrics()
    with (
        pytest.raises(RuntimeError, match="inference failed"),
        video_overlap_closure(
            guider, runtime, shapes, prefix_t=12, tokens=8, sigmas=torch.tensor([0.8, 0.6, 0.4, 0.0]), metrics=metrics
        ) as owner,
    ):
        owner.mask(torch.tensor([0.8]), runtime)
        with (
            pytest.raises(RuntimeError, match="nested"),
            video_overlap_closure(
                guider, runtime, shapes, prefix_t=12, tokens=8, sigmas=torch.tensor([0.8, 0.0]), metrics=metrics
            ),
        ):
            pass
        raise RuntimeError("inference failed")
    assert apply == {"foreign": foreign}
    assert apply["foreign"] is foreign
    assert "denoise_mask_function" not in guider.model_options
    assert owner.closed and owner.current_mask is None
    with pytest.raises(RuntimeError, match="outside its sampler lifetime"):
        owner.mask(torch.tensor([0.8]), runtime)


def test_disabled_overlap_does_not_install_hooks_or_emit_receipts():
    _, shapes, exact = _case()
    options = {"denoise_mask_function": object()}
    guider = SimpleNamespace(model_options=options)
    metrics = H3FlowMetrics()
    with video_overlap_closure(
        guider, exact, shapes, prefix_t=12, tokens=0, sigmas=torch.tensor([0.8, 0.0]), metrics=metrics
    ) as owner:
        assert owner is None
    assert guider.model_options is options
    assert not metrics.events
