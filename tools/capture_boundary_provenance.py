"""Capture read-only runtime source, workflow and checkpoint provenance.

Run separately from timed inference; hashing large checkpoints incurs disk I/O.
Git is used only to inspect the installed Patcher trees, never to change them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import struct
import subprocess
import time
from pathlib import Path


def hash_file(path):
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
        raise RuntimeError(f"file changed during provenance hashing: {path}")
    return digest.hexdigest()


def git(root, *args):
    result = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"cannot inspect repository {root}: {result.stderr.strip()}")
    return result.stdout.strip()


def repository(root):
    root = root.resolve()
    entries = {}
    # Hash installed tracked bytes: HEAD alone does not identify Patcher overlays
    # or a modified working tree. Deleted files are represented explicitly.
    names = git(root, "ls-files", "-z").split("\0")
    for name in filter(None, names):
        path = root / name
        entries[name] = hash_file(path) if path.is_file() else None
    raw = json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    return {
        "path": str(root),
        "head": git(root, "rev-parse", "HEAD"),
        "head_tree": git(root, "rev-parse", "HEAD^{tree}"),
        "branch": git(root, "rev-parse", "--abbrev-ref", "HEAD"),
        "status": git(root, "status", "--porcelain=v1", "--untracked-files=normal"),
        "tracked_content_sha256": hashlib.sha256(raw).hexdigest(),
        "tracked_files": entries,
    }


def checkpoint(path):
    path = path.resolve()
    record = {"path": str(path), "bytes": path.stat().st_size, "sha256": hash_file(path)}
    if path.suffix == ".safetensors":
        with path.open("rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            if size > 64 * 1024 * 1024:
                raise RuntimeError("checkpoint header exceeds provenance parser budget")
            header = json.loads(handle.read(size))
        record["metadata"] = header.pop("__metadata__", {})
        record["tensor_shapes_dtypes"] = {
            key: {"shape": value["shape"], "dtype": value["dtype"]} for key, value in header.items()
        }
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--patcher-state", type=Path, help="Exported Patcher overlay/dependency state")
    parser.add_argument("--checkpoint", action="append", type=Path, default=[])
    parser.add_argument("--repo", action="append", type=Path, default=[])
    parser.add_argument("--witness-receipt", action="append", type=Path, default=[])
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    started = time.perf_counter()
    root = args.comfy_root.resolve()
    repos = [root, *(path.resolve() for path in args.repo)]
    repos += [path.resolve() for path in (root / "custom_nodes").iterdir() if (path / ".git").exists()]
    result = {
        "api": 1,
        "python": platform.python_version(),
        "repositories": [repository(path) for path in sorted(set(repos))],
        "workflow": {
            "path": str(args.workflow.resolve()),
            "sha256": hash_file(args.workflow),
            "content": json.loads(args.workflow.read_text()),
        },
        "checkpoints": [checkpoint(path) for path in args.checkpoint],
        "witness_receipts": [json.loads(path.read_text()) for path in args.witness_receipt],
        "environment": {
            key: "<redacted>" if any(word in key for word in ("TOKEN", "KEY", "SECRET", "PASSWORD")) else value
            for key, value in os.environ.items()
            if key.startswith(("H3_FLOW_", "SOL_H3_", "SPECTRUM_H3_", "VDN_H3_"))
        },
        "patcher_state": json.loads(args.patcher_state.read_text()) if args.patcher_state else None,
        "installed_provenance_complete": False,
        "completion_requires": [
            "verify all effective overlays/dependency order",
            "verify loaded module paths/hashes",
            "verify every model/VDN/upscaler/VAE/LoRA/DoRA checkpoint and effective strength",
            "record runtime precision/TF32, seed/noise and reference identities",
        ],
    }
    try:
        import torch

        result["torch"] = torch.__version__
        result["collector_process_precision"] = {
            "matmul": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        }
    except ImportError:
        result["torch"] = None
    result["provenance_hashing_host_wall_s"] = time.perf_counter() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
