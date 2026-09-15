from __future__ import annotations

from pathlib import Path

from h3_flow_regenerate import same_state_high_replay as replay


def _source_entry(path: Path, *, module: str, sha256: str | None = None) -> dict:
    resolved = path.resolve()
    return {
        "module": module,
        "file": {
            "path": str(resolved),
            "resolved_path": str(resolved),
            "sha256": sha256 or replay._sha256_file(resolved),
        },
    }


def _identity(sol_sources: list[dict], *, runtime_fingerprint: str = "same-runtime") -> dict:
    return {
        "schema_version": 3,
        "gate_complete": True,
        "model": {"runtime_fingerprint": runtime_fingerprint},
        "loaded_companion_sources": {"sol_h3": sol_sources},
    }


def test_cross_process_companion_provenance_accepts_exact_55_to_10_lazy_import_subset(tmp_path: Path):
    source_root = tmp_path / "ComfyUI-Sol-H3" / "sol_h3"
    source_root.mkdir(parents=True)
    paths = []
    for index in range(55):
        path = source_root / f"module_{index:02d}.py"
        path.write_text(f"VALUE = {index}\n", encoding="utf-8")
        paths.append(path)

    capture_sources = [
        _source_entry(path, module=f"sol_h3.module_{index:02d}") for index, path in enumerate(paths)
    ]
    replay_sources = capture_sources[:10]
    capture = _identity(capture_sources)
    cold = _identity(replay_sources)

    assert replay._cross_process_provenance_diff_paths(capture, cold) == []

    strict = replay._provenance_diff_paths(capture, cold)
    assert "$.loaded_companion_sources.sol_h3.length (55 != 10)" in strict


def test_cross_process_companion_provenance_hashes_capture_only_sources_on_disk(tmp_path: Path):
    source_root = tmp_path / "ComfyUI-Sol-H3" / "sol_h3"
    source_root.mkdir(parents=True)
    paths = []
    for index in range(12):
        path = source_root / f"module_{index:02d}.py"
        path.write_text(f"VALUE = {index}\n", encoding="utf-8")
        paths.append(path)

    capture_sources = [
        _source_entry(path, module=f"sol_h3.module_{index:02d}") for index, path in enumerate(paths)
    ]
    capture = _identity(capture_sources)
    cold = _identity(capture_sources[:3])

    paths[9].write_text("VALUE = 'changed'\n", encoding="utf-8")

    differences = replay._cross_process_provenance_diff_paths(capture, cold)
    assert any("capture_source[" in item and ".on_disk_sha256" in item for item in differences)


def test_cross_process_companion_provenance_rejects_shared_reported_sha_mismatch(tmp_path: Path):
    path = tmp_path / "ComfyUI-Sol-H3" / "sol_h3" / "runtime.py"
    path.parent.mkdir(parents=True)
    path.write_text("VALUE = 1\n", encoding="utf-8")

    capture_entry = _source_entry(path, module="sol_h3.runtime")
    replay_entry = _source_entry(path, module="sol_h3.runtime", sha256="0" * 64)

    differences = replay._cross_process_provenance_diff_paths(
        _identity([capture_entry]),
        _identity([replay_entry]),
    )
    assert any("shared_source[" in item and item.endswith(".sha256") for item in differences)


def test_cross_process_companion_provenance_rejects_replay_only_source(tmp_path: Path):
    source_root = tmp_path / "ComfyUI-Sol-H3" / "sol_h3"
    source_root.mkdir(parents=True)
    capture_path = source_root / "runtime.py"
    replay_only_path = source_root / "lazy_only.py"
    capture_path.write_text("VALUE = 1\n", encoding="utf-8")
    replay_only_path.write_text("VALUE = 2\n", encoding="utf-8")

    capture_entry = _source_entry(capture_path, module="sol_h3.runtime")
    replay_entries = [
        capture_entry,
        _source_entry(replay_only_path, module="sol_h3.lazy_only"),
    ]

    differences = replay._cross_process_provenance_diff_paths(
        _identity([capture_entry]),
        _identity(replay_entries),
    )
    assert any("replay_only_source[" in item for item in differences)


def test_cross_process_companion_provenance_keeps_noncompanion_policy_fail_closed(tmp_path: Path):
    path = tmp_path / "ComfyUI-Sol-H3" / "sol_h3" / "runtime.py"
    path.parent.mkdir(parents=True)
    path.write_text("VALUE = 1\n", encoding="utf-8")
    entry = _source_entry(path, module="sol_h3.runtime")

    capture = _identity([entry], runtime_fingerprint="capture-runtime")
    cold = _identity([entry], runtime_fingerprint="different-runtime")

    differences = replay._cross_process_provenance_diff_paths(capture, cold)
    assert "$.model.runtime_fingerprint" in differences
