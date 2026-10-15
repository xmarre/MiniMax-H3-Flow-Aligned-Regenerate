"""Native high temporal-conditioning invariants without an H3 GPU/VAE audit."""

import pytest
import torch

from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.high_native_temporal_continuity import (
    high_native_temporal_clean_latent,
    high_native_temporal_mask,
    verify_high_native_temporal_entry,
)

SHAPES = [(1, 24, 17, 4, 6), (1, 32, 2, 8)]


def operands():
    video = torch.zeros(SHAPES[0])
    audio = torch.randn(SHAPES[1])
    clean_video = torch.randn(SHAPES[0])
    video_mask = torch.ones(SHAPES[0])
    video_mask[:, :, :12] = 0
    audio_mask = torch.rand(SHAPES[1])
    return (
        pack_streams((video, audio))[0],
        clean_video,
        pack_streams((video_mask, audio_mask))[0],
    )


def test_high_clean_owns_only_first_two_suffix_tokens_and_preserves_audio():
    latent, clean, _ = operands()
    original = latent.clone()
    owned = high_native_temporal_clean_latent(latent, clean, SHAPES, 12)
    old_video, old_audio = unpack_streams(original, SHAPES)
    new_video, new_audio = unpack_streams(owned, SHAPES)
    assert torch.equal(latent, original)
    assert torch.equal(old_video[:, :, :12], new_video[:, :, :12])
    assert torch.equal(new_video[:, :, 12:14], clean[:, :, 12:14])
    assert torch.equal(old_video[:, :, 14:], new_video[:, :, 14:])
    assert torch.equal(old_audio, new_audio)


def test_exact_graded_mask_preserves_audio_prefix_later_suffix_and_labels():
    _, _, mask = operands()
    original = mask.clone()
    result = high_native_temporal_mask(mask, SHAPES, 12)
    video, audio = unpack_streams(result, SHAPES)
    old_video, old_audio = unpack_streams(original, SHAPES)
    assert torch.equal(original, mask)
    assert torch.equal(video[:, :, :12], old_video[:, :, :12])
    assert torch.equal(video[:, :, 14:], old_video[:, :, 14:])
    assert torch.equal(audio, old_audio)
    assert bool((video[:, :, 12] == 0.5).all())
    assert bool((video[:, :, 13] == 0.75).all())
    assert torch.equal(torch.ceil(video * 256) / 256, video)
    # Core H3 denoise labels follow the sigma-scaled, per-token noise strength.
    for sigma in (0.87804878, 0.5, 0.125):
        labels = 1 - video[:, :, 12:14] * sigma
        assert torch.allclose(labels[:, :, 0], torch.full_like(labels[:, :, 0], 1 - sigma * 0.5))
        assert torch.allclose(labels[:, :, 1], torch.full_like(labels[:, :, 1], 1 - sigma * 0.75))


def test_high_entry_noisy_state_is_reconstructed_against_matched_clean():
    latent, clean, _ = operands()
    ref = high_native_temporal_clean_latent(latent, clean, SHAPES, 12)
    original_video, _ = unpack_streams(latent, SHAPES)
    new_video, _ = unpack_streams(ref, SHAPES)
    sigma = 0.87804878
    target = torch.randn_like(new_video)
    original_noise = (target - (1 - sigma) * original_video) / sigma
    rebuilt_noise = (target - (1 - sigma) * new_video) / sigma
    assert torch.allclose((1 - sigma) * new_video + sigma * rebuilt_noise, target, atol=1e-6)
    # The change is confined to the two transition tokens in the reconstructed noise.
    assert torch.equal(original_noise[:, :, :12], rebuilt_noise[:, :, :12])
    assert torch.equal(original_noise[:, :, 14:], rebuilt_noise[:, :, 14:])


def test_nonbinary_or_nonprotected_input_fails_closed():
    _, _, mask = operands()
    video, audio = unpack_streams(mask, SHAPES)
    v = video.clone()
    v[:, :, 12] = 0.4
    with pytest.raises(ValueError, match="generated VIDEO mask"):
        high_native_temporal_mask(pack_streams((v, audio))[0], SHAPES, 12)
    v = video.clone()
    v[:, :, 11] = 0.2
    with pytest.raises(ValueError, match="exact zero"):
        high_native_temporal_mask(pack_streams((v, audio))[0], SHAPES, 12)


def test_actual_entry_verifier_rejects_bad_noise():
    latent, clean, _ = operands()
    ref = high_native_temporal_clean_latent(latent, clean, SHAPES, 12)
    sigma = 0.87804878
    noise = torch.randn_like(ref)
    raw = (1 - sigma) * ref + sigma * noise
    assert verify_high_native_temporal_entry(raw, noise, ref, SHAPES, 12, sigma, 1.0) < 1e-5
    with pytest.raises(RuntimeError, match="did not reconstruct"):
        verify_high_native_temporal_entry(raw, torch.zeros_like(noise), ref, SHAPES, 12, sigma, 1.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA/CPU mixed-device regression requires a CUDA runner")
def test_actual_entry_verifier_accepts_cuda_state_and_cpu_clean_without_full_latent_copy():
    """Regression for the user's 2026-10-15 Continuum chunk-2 crash."""
    latent, clean, _ = operands()
    # Flow's OUTER_SAMPLE may retain the clean reference on CPU, even
    # though the transfer prediction and sampler noise already live on CUDA.
    owned = high_native_temporal_clean_latent(latent, clean, SHAPES, 12)
    assert owned.device.type == "cpu"
    sigma = 0.87804878
    noise = torch.randn_like(owned, device="cuda")
    raw = (1 - sigma) * owned.cuda() + sigma * noise
    assert verify_high_native_temporal_entry(raw, noise, owned, SHAPES, 12, sigma, 1.0) < 1e-5
    # The reverse mismatch is also valid; only graded rows are transferred.
    assert verify_high_native_temporal_entry(raw, noise.cpu(), owned, SHAPES, 12, sigma, 1.0) < 1e-5


def test_actual_entry_verifier_converts_clean_reference_dtype_on_cpu():
    """Regression on every CPU-only CI runner for mismatched clean dtype."""
    latent, clean, _ = operands()
    ref = high_native_temporal_clean_latent(latent, clean, SHAPES, 12)
    sigma = 0.87804878
    eps = torch.randn_like(ref)
    target_raw = (1 - sigma) * ref + sigma * eps
    assert verify_high_native_temporal_entry(target_raw, eps, ref.double(), SHAPES, 12, sigma, 1.0) < 1e-5
