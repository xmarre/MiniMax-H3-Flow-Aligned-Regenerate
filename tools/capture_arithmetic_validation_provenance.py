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

SourceProvenanceError = _PROVENANCE.SourceProvenanceError
capture_source_provenance = _PROVENANCE.capture_source_provenance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runtime-receipt",
        required=True,
        type=Path,
        help=(
            "JSON emitted by the MiniMax H3 Arithmetic Validation Runtime Source Receipt "
            "node in the running ComfyUI process."
        ),
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    try:
        runtime_receipt = json.loads(args.runtime_receipt.read_text(encoding="utf-8"))
        provenance = capture_source_provenance(runtime_receipt=runtime_receipt)
    except (OSError, json.JSONDecodeError, SourceProvenanceError) as exc:
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
