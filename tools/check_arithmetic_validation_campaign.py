#!/usr/bin/env python3
"""Validate a complete staged arithmetic-validation hardware campaign."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3_flow_regenerate.arithmetic_validation_campaign import (  # noqa: E402
    CampaignEvidenceError,
    validate_campaign_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()

    path = args.manifest.resolve()
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read campaign manifest {path}: {exc}") from exc

    try:
        report = validate_campaign_manifest(manifest, root=path.parent)
    except CampaignEvidenceError as exc:
        raise SystemExit(f"arithmetic-validation campaign gate: FAIL: {exc}") from exc

    print(json.dumps({"status": "pass", **report.as_dict()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
