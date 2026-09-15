from __future__ import annotations

import inspect

import torch

from h3_flow_regenerate import same_state_high_replay as replay


def _base_model(noise_scale: float):
    sampling = type("Sampling", (), {"noise_scale": noise_scale})()
    return type("BaseModel", (), {"model_sampling": sampling})()


def test_flow_inverse_is_not_a_bit_exact_noise_recovery_contract():
    torch.manual_seed(0)
    sigma = 0.4
    noise_scale = 1.25
    noise = torch.randn(1, 4096, dtype=torch.float32)
    latent = torch.randn_like(noise)
    state = sigma * (noise_scale * noise) + (1.0 - sigma) * latent

    recovered = replay._runtime._noise_argument(_base_model(noise_scale), state, sigma, latent)

    assert not torch.equal(recovered, noise)
    assert torch.allclose(recovered, noise, atol=1e-6, rtol=1e-6)


def test_forward_initialization_reconstructs_captured_state_bit_exactly():
    torch.manual_seed(1)
    sigma = torch.tensor(0.4, dtype=torch.float32)
    noise_scale = 1.25
    noise = torch.randn(1, 4096, dtype=torch.float32)
    latent = torch.randn_like(noise)
    captured_state = sigma * (noise_scale * noise) + (1.0 - sigma) * latent

    reconstructed = replay._forward_sampler_initial_state(
        _base_model(noise_scale),
        noise,
        latent,
        sigma,
        captured_state,
    )

    assert torch.equal(reconstructed, captured_state)


def test_forward_initialization_detects_changed_noise_argument():
    torch.manual_seed(2)
    sigma = torch.tensor(0.4, dtype=torch.float32)
    noise_scale = 1.25
    noise = torch.randn(1, 1024, dtype=torch.float32)
    latent = torch.randn_like(noise)
    captured_state = sigma * (noise_scale * noise) + (1.0 - sigma) * latent
    changed_noise = noise.clone()
    changed_noise[0, 17] += 0.01

    reconstructed = replay._forward_sampler_initial_state(
        _base_model(noise_scale),
        changed_noise,
        latent,
        sigma,
        captured_state,
    )

    assert not torch.equal(reconstructed, captured_state)


def test_replay_uses_captured_noise_and_not_algebraic_inverse():
    source = inspect.getsource(replay._replay_wrapper)

    assert '_tensor_for_replay(state, "high_noise_argument")' in source
    assert "replay_noise = captured_noise" in source
    assert "_forward_sampler_initial_state(" in source
    assert "_runtime._noise_argument(" not in source
