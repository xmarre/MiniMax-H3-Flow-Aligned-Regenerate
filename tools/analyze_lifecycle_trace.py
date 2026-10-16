#!/usr/bin/env python3
"""Summarize an H3 lifecycle trace (``H3_LIFECYCLE_TRACE``) as a bottleneck report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3_flow_regenerate.lifecycle_report import analyze_trace, format_report  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="h3_lifecycle_*.json written by the trace")
    parser.add_argument("--json", type=Path, help="also write the machine-readable summary here")
    parser.add_argument("--top", type=int, default=15, help="rows shown in node and gap tables")
    args = parser.parse_args(argv)
    try:
        trace = json.loads(args.trace.read_text(encoding="utf-8"))
        summary = analyze_trace(trace)
    except (OSError, ValueError) as exc:
        print(f"cannot analyze {args.trace}: {exc}", file=sys.stderr)
        return 2
    if args.json is not None:
        args.json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(format_report(summary, top=args.top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
