#!/usr/bin/env python3
"""Validate one partitioned exact-prefix production run from saved evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from h3_flow_regenerate.partitioned_runtime_gate import (  # noqa: E402
    RuntimeGateError,
    validate_partitioned_runtime_evidence,
)


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read metrics JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit(f"metrics JSON root must be an object: {path}")
    return value


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise SystemExit(f"cannot read ComfyUI log {path}: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Flow metrics plus the matching ComfyUI log for the partitioned "
            "exact-prefix SM120 production gate."
        )
    )
    parser.add_argument("--metrics", required=True, type=Path, help="Flow H3 metrics JSON")
    parser.add_argument("--log", required=True, type=Path, help="ComfyUI process log")
    parser.add_argument("--expected-vdn-api", type=int, default=4)
    parser.add_argument("--expected-logical", type=int)
    parser.add_argument("--expected-actual", type=int)
    parser.add_argument("--expected-forecast", type=int)
    parser.add_argument(
        "--expected-audio-position-domain",
        choices=("legacy_target", "source_carrier"),
        help="Validate the selected target-audio position policy and its candidate receipts.",
    )
    parser.add_argument(
        "--expected-frame-gauge-mode",
        choices=("off", "on-accepted", "on-rejected", "on-identity"),
        help="Require the latest partitioned frame-gauge arm/result and its zero-extra-work receipt.",
    )
    parser.add_argument(
        "--auto-strength-report",
        action="append",
        type=Path,
        default=[],
        help="Resolved DoRA auto_strength_report_json file. Repeat for every applicable loader.",
    )
    parser.add_argument(
        "--require-auto-strength-off",
        action="store_true",
        help="Fail unless every supplied DoRA loader report proves resolved auto-strength OFF.",
    )
    parser.add_argument(
        "--expected-auto-strength-digest",
        action="append",
        default=[],
        help="Expected canonical DoRA report digest from the matched control arm. Repeat as needed.",
    )
    parser.add_argument(
        "--allow-no-spectrum",
        action="store_true",
        help="Do not require at least one Spectrum forecast call.",
    )
    parser.add_argument(
        "--allow-no-audio-overlap",
        action="store_true",
        help="Do not require the four-tick partitioned guided-audio marker.",
    )
    parser.add_argument(
        "--allow-no-vdn-linear",
        action="store_true",
        help="Do not require the VDN variable-grid linear-complement active marker.",
    )
    args = parser.parse_args()

    metrics = _read_json(args.metrics)
    log_text = _read_text(args.log)
    auto_strength_reports = [_read_json(path) for path in args.auto_strength_report]
    try:
        report = validate_partitioned_runtime_evidence(
            metrics,
            log_text,
            expected_vdn_api=args.expected_vdn_api,
            expected_logical=args.expected_logical,
            expected_actual=args.expected_actual,
            expected_forecast=args.expected_forecast,
            require_spectrum=not args.allow_no_spectrum,
            require_audio_overlap=not args.allow_no_audio_overlap,
            require_vdn_linear=not args.allow_no_vdn_linear,
            expected_audio_position_domain=args.expected_audio_position_domain,
            expected_frame_gauge_mode=args.expected_frame_gauge_mode,
            auto_strength_reports=auto_strength_reports,
            require_auto_strength_off=args.require_auto_strength_off,
            expected_auto_strength_digests=(
                args.expected_auto_strength_digest if args.expected_auto_strength_digest else None
            ),
        )
    except RuntimeGateError as exc:
        raise SystemExit(f"partitioned exact-prefix runtime gate: FAIL: {exc}") from exc

    print(json.dumps({"status": "pass", **report.as_dict()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
