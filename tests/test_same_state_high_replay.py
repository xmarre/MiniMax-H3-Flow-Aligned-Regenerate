from __future__ import annotations

import copy
from pathlib import Path

import torch

from h3_flow_regenerate import same_state_high_replay as replay


def test_provenance_identity_ignores_only_oc_and_replay_instrumentation():
    source = {
        "active_wrapper_order": {
            "outer_sample": [
                {"key": "h3_flow_regenerate.exec_contract.invocation.v1", "callable": {"module": "diag"}},
                {"key": "h3_flow_regenerate.outer.v1", "callable": {"module": "flow"}},
                {"key": "h3_flow_regenerate.exec_contract.stage.v1", "callable": {"module": "diag"}},
                {"key": "sol_h3_runtime_v1", "callable": {"module": "sol"}},
            ]
        },
        "active_injections": {
            "vdn_lora_7ffdeadc0de": {
                "file": {"sha256": "stable-source"},
                "owner": "<object at 0x7ffdeadc0de>",
            }
        },
        "observer_wrapper_keys": {"outer": ["diagnostic-only"]},
    }
    capture = copy.deepcopy(source)
    capture["active_wrapper_order"]["outer_sample"].append(
        {"key": "h3_flow_regenerate.same_state_replay.capture.v1", "callable": {"module": "capture"}}
    )
    replay_job = copy.deepcopy(source)
    replay_job["active_wrapper_order"]["outer_sample"].append(
        {"key": "h3_flow_regenerate.same_state_replay.execute.v1", "callable": {"module": "replay"}}
    )

    assert replay._provenance_digest(capture) == replay._provenance_digest(replay_job)

    changed = copy.deepcopy(replay_job)
    changed["active_injections"]["vdn_lora_7ffdeadc0de"]["file"]["sha256"] = "changed-source"
    assert replay._provenance_digest(capture) != replay._provenance_digest(changed)


def test_runtime_policy_identity_tracks_execution_policy_not_counters():
    baseline = {
        "h3_refinement": {
            "api": 1,
            "active": True,
            "min_actual_prefix_steps": 1,
            "sigma_reference": 1.0,
            "source": "h3_flow_progressive_handoff",
        },
        "minimax_h3_untwist_rope": {
            "enabled": True,
            "beta": 2.0,
            "start_percent": 0.0,
            "end_percent": 0.95,
            "low_scale_start": 1.0,
            "low_scale_end": 1.05,
            "high_scale_start": 0.95,
            "high_scale_end": 1.0,
            "reference_scope": "image_and_video",
            "progress": 0.0,
            "progress_source": "local_sample_sigmas",
            "transient_counter": 99,
        },
        "sol_h3_runtime_v1": {
            "api": 1,
            "backend": "sol",
            "exact": True,
            "approximate": True,
            "dense_evaluations": 1,
            "dense_layers": 2,
            "kernel_contract": "rect-sm120-v3",
            "fingerprint": "abc",
            "actual_evaluations": 17,
            "sparse_calls": 1234,
        },
        "spectrum_h3_external_patch_runtime": [
            {"provider": "comfyui-diffaid-patches", "instance_id": "diffaid-h3-1", "normalized_sigma": 0.878}
        ],
        "sample_sigmas": {"shape": [4], "dtype": "torch.float32", "sample_digest": "sigmas"},
    }
    counter_only = copy.deepcopy(baseline)
    counter_only["sol_h3_runtime_v1"]["actual_evaluations"] = 1
    counter_only["sol_h3_runtime_v1"]["sparse_calls"] = 2

    assert replay._runtime_policy_identity(baseline) == replay._runtime_policy_identity(counter_only)

    changed = copy.deepcopy(baseline)
    changed["minimax_h3_untwist_rope"]["progress"] = 5.0 / 7.0
    assert replay._runtime_policy_identity(baseline) != replay._runtime_policy_identity(changed)


def test_bundle_roundtrip_is_hash_checked_and_tensor_only(tmp_path: Path):
    tensors = {
        "high_noise_argument": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "high_sigmas": torch.tensor([0.8, 0.6, 0.0], dtype=torch.float32),
    }
    material = replay._ReplayMaterial(
        manifest={
            "schema_version": replay.SCHEMA_VERSION,
            "kind": "minimax_h3_same_state_high_replay",
            "capture_id": "capture-test",
            "tensor_hashes": {key: replay._tensor_sha256(value) for key, value in tensors.items()},
        },
        tensors=tensors,
    )
    manifest_path, tensor_path = replay._save_material(material, tmp_path, "replay-test")
    resolved, manifest, loaded = replay._load_bundle(str(manifest_path))

    assert resolved == manifest_path
    assert manifest["tensors_file"] == tensor_path.name
    assert set(loaded) == set(tensors)
    for key in tensors:
        assert torch.equal(loaded[key], tensors[key])

    tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")
    try:
        replay._load_bundle(str(manifest_path))
    except RuntimeError as exc:
        assert "tensor file hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("corrupt replay payload was accepted")


def test_rebuild_trajectory_restores_declarative_guidance_samples():
    video_x0 = torch.arange(1 * 24 * 2 * 4 * 4, dtype=torch.float32).reshape(1, 24, 2, 4, 4)
    manifest = {
        "trajectory": {
            "schema_version": 1,
            "session_id": "standalone",
            "chunk_id": "standalone",
            "sampler": "sample_euler",
            "scheduler": "schedule",
            "geometry": {
                "batch": 1,
                "latent_t": 2,
                "latent_h": 4,
                "latent_w": 4,
                "pixel_h": 64,
                "pixel_w": 64,
                "padded_h": 4,
                "padded_w": 4,
            },
            "audio_shape": [1, 32, 2, 9],
            "layout_signature": "layout",
            "conditioning_signature": "conditioning",
        },
        "guidance_samples": [
            {
                "tensor_key": "guidance_sample_000_video_x0",
                "coordinate": 0.5,
                "video_sigma": 0.9,
                "audio_sigma": 0.8,
                "outer_step": 2,
                "call_index": 3,
                "phase": "single",
                "provenance": "actual",
            }
        ],
    }
    trajectory, run_id = replay._rebuild_trajectory(
        manifest,
        {"guidance_sample_000_video_x0": video_x0},
    )

    assert trajectory is not None
    assert run_id is not None
    run = trajectory.select(run_id=run_id)
    assert len(run.samples) == 1
    sample = run.samples[0]
    assert sample.coordinate == 0.5
    assert sample.outer_step == 2
    assert sample.provenance == "actual"
    assert torch.equal(sample.video_x0, video_x0)


def test_replay_source_has_no_progressive_or_upscaler_execution_path():
    import inspect

    source = inspect.getsource(replay._replay_wrapper)
    assert "build_handoff_state" not in source
    assert "_run_progressive" not in source
    assert "handoff_learned_upscale" not in source
