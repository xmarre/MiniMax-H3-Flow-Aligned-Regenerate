from __future__ import annotations

import copy
import json
from pathlib import Path

import torch

from h3_flow_regenerate import same_state_high_replay as replay


def test_provenance_identity_ignores_only_diagnostic_execution_state():
    source = {
        "capture_phase": "outer_sample_runtime",
        "collected_ns": 100,
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
        "model_options_alias_graph": [{"alias": "alias_1", "paths": ["$.capture"]}],
        replay._RUNTIME_OBSERVATION_KEY: {
            "stage_lifecycles": [{"stage": "low"}],
            "companion_calls": {"high_first": {"before": {"counter": 1}}},
        },
    }
    capture = copy.deepcopy(source)
    capture["active_wrapper_order"]["outer_sample"].append(
        {"key": "h3_flow_regenerate.same_state_replay.capture.v1", "callable": {"module": "capture"}}
    )
    replay_job = copy.deepcopy(source)
    replay_job["collected_ns"] = 999
    replay_job["model_options_alias_graph"] = [{"alias": "alias_9", "paths": ["$.replay"]}]
    replay_job[replay._RUNTIME_OBSERVATION_KEY] = {
        "stage_lifecycles": [{"stage": "high"}],
        "companion_calls": {"high_first": {"before": {"counter": 99}}},
    }
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
        "spectrum_h3_actual": True,
        "spectrum_h3_solver_phase": "single",
        "spectrum_h3_outer_step_id": 0,
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

    spectrum_changed = copy.deepcopy(baseline)
    spectrum_changed["spectrum_h3_actual"] = False
    assert replay._runtime_policy_identity(baseline) != replay._runtime_policy_identity(spectrum_changed)


def _minimal_manifest(tensors: dict[str, torch.Tensor]) -> dict:
    provenance = {"schema_version": 3, "gate_complete": True, "model": {"runtime_fingerprint": "model"}}
    runtime_policy = {"h3_refinement": {"active": True}}
    companion_policy = {"before": {"vdn": [], "sol": {}, "spectrum": {}}, "after": {}}
    return {
        "schema_version": replay.SCHEMA_VERSION,
        "kind": "minimax_h3_same_state_high_replay",
        "capture_id": "capture-test",
        "tensor_hashes": {key: replay._tensor_sha256(value) for key, value in tensors.items()},
        "tensor_contracts": {
            key: {
                "shape": [int(dim) for dim in value.shape],
                "stride": [int(item) for item in value.stride()],
                "dtype": str(value.dtype),
                "device": "cpu",
            }
            for key, value in tensors.items()
        },
        "provenance_identity": provenance,
        "provenance_digest": replay._sha_json(provenance),
        "first_high_runtime_policy": runtime_policy,
        "first_high_runtime_policy_digest": replay._sha_json(runtime_policy),
        "first_high_companion_policy": companion_policy,
        "first_high_companion_policy_digest": replay._sha_json(companion_policy),
        "exact_checkpoint_identity": {
            "resolved_path": "/model.safetensors",
            "size": 1,
            "mtime_ns": 1,
            "sha256": "a" * 64,
        },
        "contract": {
            "stochastic_state_transport": False,
            "weighted_mixed_grid_active": False,
            "untwist_active": False,
        },
    }


def test_cpu_clone_preserves_noncontiguous_stride():
    value = torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1)
    assert not value.is_contiguous()
    cloned = replay._cpu_clone(value)
    assert cloned.stride() == value.stride()
    assert torch.equal(cloned, value)


def test_bundle_roundtrip_is_hash_checked_tensor_only_and_exact_keyed(tmp_path: Path):
    tensors = {
        "entry": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "strided": torch.arange(24, dtype=torch.float32).reshape(4, 6).transpose(0, 1),
    }
    material = replay._ReplayMaterial(manifest=_minimal_manifest(tensors), tensors=tensors)
    manifest_path, tensor_path = replay._save_material(material, tmp_path, "replay-test")
    resolved, manifest, loaded = replay._load_bundle(str(manifest_path))

    assert resolved == manifest_path
    assert manifest["tensors_file"] == tensor_path.name
    assert set(loaded) == set(tensors)
    for key in tensors:
        assert torch.equal(loaded[key], tensors[key])
        assert loaded[key].stride() == tensors[key].stride()

    payload = torch.load(tensor_path, map_location="cpu", weights_only=True)
    payload["unlisted"] = torch.ones(1)
    torch.save(payload, tensor_path)
    edited = json.loads(manifest_path.read_text(encoding="utf-8"))
    edited["tensors_file_sha256"] = replay._sha256_file(tensor_path)
    manifest_path.write_text(json.dumps(edited), encoding="utf-8")
    try:
        replay._load_bundle(str(manifest_path))
    except RuntimeError as exc:
        assert "payload keys differ" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unlisted replay tensor was accepted")


def test_bundle_rejects_corrupt_tensor_file_before_deserialization(tmp_path: Path):
    tensors = {"entry": torch.arange(4, dtype=torch.float32)}
    material = replay._ReplayMaterial(manifest=_minimal_manifest(tensors), tensors=tensors)
    manifest_path, tensor_path = replay._save_material(material, tmp_path, "replay-test")
    tensor_path.write_bytes(tensor_path.read_bytes() + b"corrupt")

    try:
        replay._load_bundle(str(manifest_path))
    except RuntimeError as exc:
        assert "tensor file hash mismatch" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("corrupt replay payload was accepted")


def test_checkpoint_identity_hashes_exact_file_and_rejects_changed_stat(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"exact-model-bytes")
    stat = checkpoint.stat()
    provenance = {
        "model": {
            "checkpoint_stat": {
                "resolved_path": str(checkpoint),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        }
    }
    identity = replay._checkpoint_identity(provenance)
    assert identity["sha256"] == replay._sha256_file(checkpoint)

    provenance["model"]["checkpoint_stat"]["size"] += 1
    try:
        replay._checkpoint_identity(provenance)
    except RuntimeError as exc:
        assert "size changed" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("changed checkpoint stat was accepted")


def test_checkpoint_identity_guard_rejects_file_change_after_preflight(tmp_path: Path):
    checkpoint = tmp_path / "model.safetensors"
    checkpoint.write_bytes(b"exact-model-bytes")
    stat = checkpoint.stat()
    provenance = {
        "model": {
            "checkpoint_stat": {
                "resolved_path": str(checkpoint),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        }
    }
    identity = replay._checkpoint_identity(provenance)
    replay._assert_checkpoint_identity_unchanged(identity)

    checkpoint.write_bytes(b"changed-model-bytes-longer")
    try:
        replay._assert_checkpoint_identity_unchanged(identity)
    except RuntimeError as exc:
        assert "changed after preflight" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("checkpoint changed after preflight was accepted")


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


def test_rebuild_trajectory_rejects_unknown_schema():
    manifest = {"trajectory": {"schema_version": 99}, "guidance_samples": []}
    try:
        replay._rebuild_trajectory(manifest, {})
    except RuntimeError as exc:
        assert "trajectory schema is unsupported" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unknown trajectory schema was accepted")


def test_checkpoint_identity_is_hashed_only_during_node_preflight():
    import inspect

    assert "_checkpoint_identity(" in inspect.getsource(replay.patch_same_state_capture)
    assert "_checkpoint_identity(" in inspect.getsource(replay.patch_same_state_replay)
    assert "_checkpoint_identity(" not in inspect.getsource(replay._capture_wrapper)
    assert "_checkpoint_identity(" not in inspect.getsource(replay._replay_wrapper)
    assert "_checkpoint_identity(" not in inspect.getsource(replay._material_from_record)


def test_replay_source_has_no_progressive_or_upscaler_execution_path():
    import inspect

    source = inspect.getsource(replay._replay_wrapper)
    assert "build_handoff_state" not in source
    assert "_run_progressive" not in source
    assert "learned_upscaler" not in source
