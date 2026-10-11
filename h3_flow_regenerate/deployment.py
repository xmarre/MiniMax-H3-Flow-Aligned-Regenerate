"""Source receipts for verifying a restarted ComfyUI custom-node installation."""

import hashlib
import json
from pathlib import Path


def report_diagnostic_sources():
    package = Path(__file__).resolve().parent
    names = (
        "low_sampler_trace.py",
        "runtime.py",
        "partitioned_scheduler.py",
        "low_sampler_ab_audit.py",
        "native_execution_trace.py",
        "partitioned_transformer.py",
        "partitioned_node.py",
    )
    sources = {}
    for name in names:
        try:
            sources[name] = {"sha256": hashlib.sha256((package / name).read_bytes()).hexdigest()}
        except OSError as error:
            sources[name] = {"unobserved": type(error).__name__}
    receipt = {
        "policy": "h3_flow_low_trace_sources_v1",
        "package_path": str(package),
        "sources": sources,
    }
    print("[H3 Flow] diagnostic_sources=" + json.dumps(receipt, sort_keys=True))
    return receipt
