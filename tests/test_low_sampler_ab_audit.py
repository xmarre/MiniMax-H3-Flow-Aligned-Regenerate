"""Native VAE offline source A/B stage audit provenance and replay controls."""

import hashlib
import json
from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate import low_sampler_ab_audit as audit


def _save(folder, table, name, tensor):
    raw = tensor.detach().contiguous().numpy().tobytes()
    (folder / f"{name}.bin").write_bytes(raw)
    table[name] = {
        "file": f"{name}.bin",
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "nbytes": len(raw),
        "byte_order": "native_torch_contiguous",
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@pytest.fixture
def source_pair(tmp_path):
    a = torch.randn((1, 24, 22, 4, 4), generator=torch.Generator().manual_seed(998)) / 8
    b = a.clone()
    b[:, :, :7] += 0.1
    b[:, :, 7:] += 0.05
    a2 = a.clone()
    target = torch.randn((1, 24, 7, 6, 6), generator=torch.Generator().manual_seed(65)) / 8
    tensor_list = (a, b, a2, target, a.clone(), b.clone(), a.clone())
    manifest = {
        "schema": 1,
        "kind": audit.KIND,
        "tensor_bytes": {},
        "metadata": {
            "policy": "h3_frozen_low_source_aba_v1",
            "model_domain": "model_internal_clean",
            "source_hw": [4, 4],
            "target_hw": [6, 6],
            "prefix_t": 7,
            "temporal": 22,
            "low_input_pairing": {"input_pair_eligible": True},
            "reproduction_verified": True,
        },
    }
    for name, tensor in zip(audit.NAMES, tensor_list, strict=True):
        _save(tmp_path, manifest["tensor_bytes"], name, tensor)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    model = SimpleNamespace(
        first_stage_model=SimpleNamespace(tokens_chunk_size=5, token_overlap=2, frame_pre_padding=3, clip_length=17)
    )
    return tmp_path, model, manifest


def _no_decode_implementation(monkeypatch):
    def decode(_vae, tensor, _process_out, _origin, labels):
        # Decode ownership is tested separately; this stub isolates file
        # integrity and matched-time ROI measurement contracts.
        pixels = tensor[0, 0, :1].mean().item()
        result = torch.linspace(0.1, 0.5, steps=128).reshape(1, 1, 128, 1)
        result = result.expand(len(labels), 128, 128, 3).clone()
        result += pixels * 0.05
        return result.clamp(0, 1), 0.0

    monkeypatch.setattr(audit, "_native_pixels", decode)


def test_saved_low_ab_pair_decodes_matched_times_and_preserves_operands(source_pair, monkeypatch):
    folder, model, _manifest = source_pair
    before = {p.name: p.read_bytes() for p in folder.iterdir()}
    _no_decode_implementation(monkeypatch)
    report = audit.audit_saved_low_aba(
        model,
        str(folder),
        175,
        lambda tensor: tensor,
        feature_tracking_enabled=False,
    )
    assert report["policy"] == audit.POLICY
    assert report["source_generation_suffix_is_altered"] is True
    assert report["a_replay_verified"] is True
    assert report["decode"]["full_source_vae_calls"] == 2
    assert report["decode"]["extra_h3_nfe"] == 0
    assert report["frame_labels"][0] == 170
    assert report["frame_labels"][-1] == 195
    assert "upper45_full" in report["same_frame_A_to_B"]["regions"]
    assert "upper45_full" not in report["baseline_A"]["static_background_rois"]["regions"]
    assert before == {p.name: p.read_bytes() for p in folder.iterdir()}


@pytest.mark.parametrize("fault", ["hash", "metadata", "reproduction", "manifest_sha"])
def test_invalid_pair_fails_before_any_vae_decode(source_pair, monkeypatch, fault):
    folder, model, manifest = source_pair
    calls = []
    monkeypatch.setattr(audit, "_native_pixels", lambda *_args: calls.append(1))
    if fault == "hash":
        target = folder / "source_B_full.bin"
        data = bytearray(target.read_bytes())
        data[0] ^= 1
        target.write_bytes(data)
    elif fault == "metadata":
        manifest["metadata"]["model_domain"] = "vae_encoded"
    elif fault == "reproduction":
        manifest["metadata"]["reproduction_verified"] = False
    else:
        pass
    if fault in ("metadata", "reproduction"):
        (folder / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        audit.audit_saved_low_aba(
            model,
            str(folder),
            175,
            lambda tensor: tensor,
            feature_tracking_enabled=False,
            expected_manifest_sha256="0" * 64 if fault == "manifest_sha" else None,
        )
    assert calls == []
