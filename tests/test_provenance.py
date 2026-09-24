import torch

from h3_flow_regenerate.provenance import effective_seed, tensor_provenance


def test_effective_seed_matches_flow_noise_seed_domain():
    assert effective_seed(-1) == (1 << 63) - 1
    assert effective_seed((1 << 63) + 7) == 7


def test_tensor_provenance_is_bounded_deterministic_and_output_neutral():
    value = torch.arange(256, dtype=torch.float32).reshape(1, 1, 1, 16, 16)
    before = value.clone()
    first = tensor_provenance(value, samples=16)
    second = tensor_provenance(value, samples=16)
    assert torch.equal(value, before)
    assert first == second
    assert first["shape"] == [1, 1, 1, 16, 16]
    assert first["dtype"] == "torch.float32"
    assert first["numel"] == 256
    assert first["sample_count"] == 16
    assert len(first["sample_sha256"]) == 64
    assert first["sample_finite"] is True


def test_tensor_provenance_never_transfers_more_than_requested_samples():
    value = torch.randn(4, 8, 32)
    report = tensor_provenance(value, samples=7)
    assert report["sample_count"] == 7
