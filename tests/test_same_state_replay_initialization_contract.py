from __future__ import annotations

import inspect
from types import SimpleNamespace

import torch

from h3_flow_regenerate import same_state_high_replay as replay


def _base_model(noise_scale: float):
    sampling = type("Sampling", (), {"noise_scale": noise_scale})()
    return type("BaseModel", (), {"model_sampling": sampling})()


def _snapshot(value: torch.Tensor):
    return replay._diag._Snapshot(
        tensor=value.clone(),
        original_device=str(value.device),
        original_stride=tuple(int(item) for item in value.stride()),
    )


def _replay_state(video: torch.Tensor, audio: torch.Tensor):
    tensors = {
        "first_high_sampler_input_video": video,
        "first_high_sampler_input_audio": audio,
    }
    return replay._ReplayState(
        manifest_path="bundle.json",
        manifest={
            "tensor_hashes": {name: replay._tensor_sha256(value) for name, value in tensors.items()},
            "tensor_contracts": {
                name: {
                    "shape": [int(dim) for dim in value.shape],
                    "stride": [int(item) for item in value.stride()],
                    "dtype": str(value.dtype),
                    "device": str(value.device),
                }
                for name, value in tensors.items()
            },
        },
        tensors={},
        exact_checkpoint_identity={},
    )


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


def test_actual_sampler_entry_gate_accepts_exact_job1_state():
    video = torch.arange(1 * 24 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 24, 2, 3, 4)
    audio = torch.arange(1 * 32 * 2 * 5, dtype=torch.float32).reshape(1, 32, 2, 5)
    state = _replay_state(video, audio)
    record = SimpleNamespace(
        snapshots={
            "first_high_sampler_input_video": _snapshot(video),
            "first_high_sampler_input_audio": _snapshot(audio),
        }
    )

    replay._validate_replay_sampler_entry(state, record)


def test_actual_sampler_entry_gate_rejects_changed_state():
    video = torch.arange(1 * 24 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 24, 2, 3, 4)
    audio = torch.arange(1 * 32 * 2 * 5, dtype=torch.float32).reshape(1, 32, 2, 5)
    state = _replay_state(video, audio)
    changed_video = video.clone()
    changed_video[0, 0, 0, 0, 0] += 1.0
    record = SimpleNamespace(
        snapshots={
            "first_high_sampler_input_video": _snapshot(changed_video),
            "first_high_sampler_input_audio": _snapshot(audio),
        }
    )

    try:
        replay._validate_replay_sampler_entry(state, record)
    except RuntimeError as exc:
        assert "SAMPLER_SAMPLE" in str(exc)
        assert "sha256" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("changed actual sampler-entry state was accepted")


def test_actual_sampler_entry_gate_rejects_contract_change():
    video = torch.arange(1 * 24 * 2 * 3 * 4, dtype=torch.float32).reshape(1, 24, 2, 3, 4)
    audio = torch.arange(1 * 32 * 2 * 5, dtype=torch.float32).reshape(1, 32, 2, 5)
    state = _replay_state(video, audio)
    state.manifest["tensor_contracts"]["first_high_sampler_input_audio"]["device"] = "cuda:7"
    record = SimpleNamespace(
        snapshots={
            "first_high_sampler_input_video": _snapshot(video),
            "first_high_sampler_input_audio": _snapshot(audio),
        }
    )

    try:
        replay._validate_replay_sampler_entry(state, record)
    except RuntimeError as exc:
        assert "device differs" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("changed sampler-entry tensor contract was accepted")


def test_replay_uses_captured_noise_and_actual_sampler_boundary_gate():
    source = inspect.getsource(replay._replay_wrapper)
    patch_source = inspect.getsource(replay.patch_same_state_replay)
    sampler_source = inspect.getsource(replay._replay_sampler_entry_wrapper)

    assert '_tensor_for_replay(state, "high_noise_argument")' in source
    assert "_runtime._noise_argument(" not in source
    assert "_forward_sampler_initial_state(" not in source
    assert "WrappersMP.SAMPLER_SAMPLE" in patch_source
    assert "_REPLAY_SAMPLER_KEY" in patch_source
    assert "_diag._SAMPLER_KEY" in patch_source
    assert "before=False" in patch_source
    assert "_validate_replay_sampler_entry" in sampler_source
