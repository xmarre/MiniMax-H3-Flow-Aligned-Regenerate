from __future__ import annotations

import ast
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.runtime import (
    _exact_probe_function,
    _merge_preserved_noise,
    _noise_argument,
    _raw_sampler_state,
)


@pytest.fixture(scope="module")
def native_sampling():
    root = Path(os.environ.get("COMFYUI_ROOT", Path(__file__).resolve().parents[2] / "comfy"))
    path = root / "comfy/model_sampling.py"
    if not path.is_file():
        if os.environ.get("COMFYUI_ROOT"):
            raise FileNotFoundError(path)
        pytest.skip("native source oracle runs in source-contract CI with COMFYUI_ROOT")

    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    wanted = {"reshape_sigma", "CONST"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in wanted]
    found = {node.name for node in nodes}
    if found != wanted:
        raise AssertionError(f"pinned ComfyUI sampling oracle changed: found {sorted(found)}")

    module = ModuleType("native_h3_state_sampling_oracle")
    module.__dict__["torch"] = torch
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("noise_scale", [0.5, 1.0, 2.0])
@pytest.mark.parametrize("sigma", [1.0e-5, 0.37, 0.99999])
def test_runtime_noise_reconstruction_closes_over_pinned_comfy_const(native_sampling, noise_scale, sigma):
    native = native_sampling.CONST()
    native.noise_scale = noise_scale
    noise = torch.linspace(-1.3, 1.7, steps=96, dtype=torch.float64).reshape(1, 1, 96)
    latent = torch.linspace(0.8, -0.6, steps=96, dtype=torch.float64).reshape_as(noise)
    state = native.noise_scaling(torch.tensor(sigma, dtype=torch.float64), noise, latent)

    base = SimpleNamespace(model_sampling=SimpleNamespace(noise_scale=noise_scale))
    recovered = _noise_argument(base, state, sigma, latent)
    torch.testing.assert_close(recovered, noise, rtol=1e-10, atol=1e-10)


def test_split_sampler_return_reconstructs_exact_raw_state_with_pinned_inverse(native_sampling):
    sigma = 0.8780487775802612
    native = native_sampling.CONST()
    raw_state = torch.linspace(-0.9, 1.1, steps=120, dtype=torch.float64).reshape(1, 1, 120)
    returned = native.inverse_noise_scaling(torch.tensor(sigma, dtype=torch.float64), raw_state)
    base = SimpleNamespace(process_latent_in=lambda value: value, latent_shapes=None)

    recovered = _raw_sampler_state(base, returned, [(1, 1, 120)], sigma)
    torch.testing.assert_close(recovered, raw_state, rtol=1e-12, atol=1e-12)


def test_exact_probe_return_closes_to_clean_under_pinned_inverse(native_sampling):
    sigma = 0.8780487775802612
    clean = torch.linspace(-0.4, 0.7, steps=120, dtype=torch.float64).reshape(1, 1, 120)
    state = torch.linspace(0.6, -0.2, steps=120, dtype=torch.float64).reshape_as(clean)

    def model(_x, _sigma, **_kwargs):
        return clean

    returned = _exact_probe_function(model, state, torch.tensor([sigma], dtype=torch.float64))
    native = native_sampling.CONST()
    accepted = native.inverse_noise_scaling(torch.tensor(sigma, dtype=torch.float64), returned)
    torch.testing.assert_close(accepted, clean, rtol=1e-12, atol=1e-12)


def test_protected_noise_merge_keeps_caller_owned_noise_exactly():
    generated = torch.arange(12, dtype=torch.float32).reshape(1, 1, 12)
    caller = -generated - 1.0
    mask = torch.ones_like(generated)
    mask[..., :5] = 0.0

    merged = _merge_preserved_noise(generated, caller, mask)
    assert torch.equal(merged[..., :5], caller[..., :5])
    assert torch.equal(merged[..., 5:], generated[..., 5:])
