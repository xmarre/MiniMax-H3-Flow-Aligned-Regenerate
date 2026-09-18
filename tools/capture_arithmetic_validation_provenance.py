#!/usr/bin/env python3
"""Capture installed source provenance for the arithmetic-validation campaign."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "h3_flow_regenerate" / "arithmetic_validation_provenance.py"
SPEC = importlib.util.spec_from_file_location(
    "_h3_arithmetic_validation_provenance",
    MODULE_PATH,
)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"cannot load source-provenance module from {MODULE_PATH}")
_PROVENANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(_PROVENANCE)

REPOSITORIES = _PROVENANCE.REPOSITORIES
SourceProvenanceError = _PROVENANCE.SourceProvenanceError
capture_source_provenance = _PROVENANCE.capture_source_provenance


def _named_path(value: str, *, option: str) -> tuple[str, Path]:
    name, separator, raw_path = value.partition("=")
    if not separator or name not in REPOSITORIES or not raw_path:
        choices = ",".join(REPOSITORIES)
        raise argparse.ArgumentTypeError(f"{option} must use NAME=PATH with NAME in {choices}")
    return name, Path(raw_path)


def _loaded_file(value: str) -> tuple[str, str, Path]:
    name_and_module, separator, raw_path = value.partition("=")
    if not separator or not raw_path:
        raise argparse.ArgumentTypeError("--loaded-file must use REPO:MODULE=PATH")
    name, separator, module = name_and_module.partition(":")
    if not separator or name not in REPOSITORIES or not module:
        choices = ",".join(REPOSITORIES)
        raise argparse.ArgumentTypeError(f"--loaded-file repository must be one of {choices}")
    return name, module, Path(raw_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        action="append",
        required=True,
        help="Repository root as NAME=PATH; repeat for flow, sol, vdn and continuum.",
    )
    parser.add_argument(
        "--loaded-file",
        action="append",
        required=True,
        help=("Actual loaded module __file__ as REPO:MODULE=PATH. Repeat at least once per repository."),
    )
    parser.add_argument(
        "--overlay-order",
        required=True,
        help="Comma-separated runtime overlay order containing flow,sol,vdn,continuum exactly once.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    repositories = {}
    for raw in args.repo:
        name, path = _named_path(raw, option="--repo")
        if name in repositories:
            raise SystemExit(f"duplicate --repo entry for {name}")
        repositories[name] = path

    loaded = {name: [] for name in REPOSITORIES}
    for raw in args.loaded_file:
        name, module, path = _loaded_file(raw)
        loaded[name].append((module, path))

    overlay = [name.strip() for name in args.overlay_order.split(",") if name.strip()]
    try:
        provenance = capture_source_provenance(
            repositories=repositories,
            loaded_files=loaded,
            overlay_order=overlay,
        )
    except SourceProvenanceError as exc:
        raise SystemExit(f"source provenance capture: FAIL: {exc}") from exc

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
