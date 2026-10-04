import hashlib
import json
import runpy
from pathlib import Path

import pytest
import torch

REPLAY = runpy.run_path(str(Path(__file__).parents[1] / "tools/decode_native_boundary_evidence.py"))


def write_operand(directory, manifest, name, value):
    raw = value.contiguous().numpy().tobytes()
    filename = f"{name}.bin"
    (directory / filename).write_bytes(raw)
    manifest["tensor_bytes"][name] = {
        "file": filename,
        "shape": list(value.shape),
        "dtype": "torch.float32",
        "byte_order": "native_torch_contiguous",
        "nbytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


@pytest.fixture
def bundle(tmp_path):
    generator = torch.Generator().manual_seed(115)
    shape = (1, 24, 7, 4, 6)
    prefix = torch.randn(1, 24, 2, 4, 6, generator=generator)
    manifest = {
        "schema": 1,
        "kind": "h3_flow_native_boundary_decoder_window_evidence",
        "metadata": {
            "policy": "native_boundary_decoder_window_evidence_v1",
            "first_high_actual": True,
            "provider_clean_provenance": "actual_clean_postprocess",
            "decoder_comparison_prefix": "replace_with_authoritative_prefix_bytes",
            "process_latent_out_required_before_vae": True,
            "window": {
                "window_tokens": 7,
                "token_overlap": 2,
                "chunk_stride_tokens": 5,
                "window_start_t": 10,
                "window_stop_t": 17,
                "prefix_t": 12,
                "temporal": 22,
                "decoder_chunk_output_start_frame": 34,
                "decoded_trim_frames": 39,
                "first_retained_local_frame": 5,
            },
        },
        "tensor_bytes": {},
    }
    for name in REPLAY["TENSOR_NAMES"]:
        value = torch.randn(shape, generator=generator)
        if name == "authoritative_prefix":
            value = prefix
        elif name in {"pre_high_exact_restored", "final_post_high_internal_clean"}:
            value[:, :, :2] = prefix
        elif name == "initial_high_video_mask":
            value.fill_(1)
            value[:, :, :2] = 0
        write_operand(tmp_path, manifest, name, value)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path, manifest


def test_replay_uses_only_clean_stages_and_restores_saved_prediction_prefixes(bundle):
    directory, _ = bundle
    _manifest, values = REPLAY["load_bundle"](directory)
    original = {name: value.clone() for name, value in values.items()}
    stages = REPLAY["clean_stages"](values)
    assert set(stages) == {
        "provider_native",
        "provider_exact_prefix",
        "pre_high_dc",
        "first_high_before_flow",
        "first_high_after_flow",
        "final",
    }
    assert torch.equal(stages["provider_native"], original["provider_native_clean"])
    for name, value in stages.items():
        if name != "provider_native":
            assert torch.equal(value[:, :, :2], original["authoritative_prefix"])
    for name, value in values.items():
        assert torch.equal(value, original[name])
    for source in ("first_high_before_flow", "first_high_after_flow"):
        assert torch.equal(stages[source][:, :, 2:], original[source][:, :, 2:])


def test_replay_rejects_byte_corruption_before_decoder_import(bundle):
    directory, _ = bundle
    path = directory / "first_high_before_flow.bin"
    raw = bytearray(path.read_bytes())
    raw[-1] ^= 1
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="bytes do not match"):
        REPLAY["load_bundle"](directory)


@pytest.mark.parametrize(
    "mutation,reason",
    [
        ("nonfinite", "non-finite"),
        ("prefix", "protected clean prefix"),
        ("prefix_zero_sign", "protected clean prefix"),
        ("mask", "initial high video mask"),
        ("path", "path leaves"),
        ("window", "window geometry drifted"),
        ("temporal_bound", "window geometry drifted"),
        ("trim", "window geometry drifted"),
        ("missing", "all eight"),
        ("forecast", "unsupported native boundary"),
        ("provenance", "unsupported native boundary"),
        ("dtype", "geometry or dtype"),
    ],
)
def test_replay_rejects_semantically_invalid_operands_even_with_updated_hashes(bundle, mutation, reason):
    directory, manifest = bundle
    if mutation in {"nonfinite", "prefix", "mask"}:
        name = {
            "nonfinite": "first_high_after_flow",
            "prefix": "final_post_high_internal_clean",
            "mask": "initial_high_video_mask",
        }[mutation]
        entry = manifest["tensor_bytes"][name]
        value = torch.frombuffer(bytearray((directory / entry["file"]).read_bytes()), dtype=torch.float32)
        value[0] = float("nan") if mutation == "nonfinite" else 1e3
        write_operand(directory, manifest, name, value.reshape(entry["shape"]))
    elif mutation == "prefix_zero_sign":
        # Signed zero compares equal numerically but violates byte ownership.
        for name in ("authoritative_prefix", "pre_high_exact_restored", "final_post_high_internal_clean"):
            entry = manifest["tensor_bytes"][name]
            value = torch.frombuffer(bytearray((directory / entry["file"]).read_bytes()), dtype=torch.float32)
            value[0] = -0.0 if name == "final_post_high_internal_clean" else 0.0
            write_operand(directory, manifest, name, value.reshape(entry["shape"]))
    elif mutation == "path":
        manifest["tensor_bytes"]["provider_native_clean"]["file"] = "../outside.bin"
    elif mutation == "window":
        manifest["metadata"]["window"]["window_start_t"] = 9
    elif mutation == "temporal_bound":
        manifest["metadata"]["window"]["temporal"] = 16
    elif mutation == "trim":
        manifest["metadata"]["window"]["decoded_trim_frames"] = 38
    elif mutation == "missing":
        del manifest["tensor_bytes"]["first_high_sampler_input"]
    elif mutation == "forecast":
        manifest["metadata"]["first_high_actual"] = False
    elif mutation == "provenance":
        manifest["metadata"]["provider_clean_provenance"] = "independently_replayed_provider"
    elif mutation == "dtype":
        manifest["tensor_bytes"]["provider_native_clean"]["dtype"] = "torch.float16"
    (directory / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=reason):
        REPLAY["load_bundle"](directory)
