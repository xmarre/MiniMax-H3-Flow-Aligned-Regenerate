#!/usr/bin/env python3
"""Build or verify matched exact-prefix validation manifests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3_flow_regenerate.validation_fixture import (  # noqa: E402
    build_fixture_manifest,
    validate_candidate_pair,
    validate_fixture_manifest,
)


def _read(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build")
    build.add_argument("--spec", required=True)
    build.add_argument("--output", required=True)

    check = sub.add_parser("check")
    check.add_argument("--fixture", required=True)

    pair = sub.add_parser("check-pair")
    pair.add_argument("--fixture", required=True)
    pair.add_argument("--control", required=True)
    pair.add_argument("--candidate", required=True)

    args = parser.parse_args()
    if args.command == "build":
        manifest = build_fixture_manifest(_read(args.spec))
        Path(args.output).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(manifest["fixture_sha256"])
    elif args.command == "check":
        print(validate_fixture_manifest(_read(args.fixture)))
    else:
        validate_candidate_pair(_read(args.fixture), _read(args.control), _read(args.candidate))
        print("matched exact-prefix candidate pair: OK")


if __name__ == "__main__":
    main()
