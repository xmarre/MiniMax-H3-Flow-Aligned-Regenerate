from pathlib import Path

path = Path("h3_flow_regenerate/first_high_sol_local_diagnostic.py")
text = path.read_text(encoding="utf-8")

text = text.replace(
    "_MIN_FREE_BYTES = 2 * 1024**3\n",
    '''_DEFAULT_DIAGNOSTIC_BUDGET_BYTES = 2 * 1024**3
_CUDA_BUDGET_ENV = "H3_FIRST_HIGH_E_CUDA_BUDGET_GIB"
_CPU_BUDGET_ENV = "H3_FIRST_HIGH_E_CPU_BUDGET_GIB"
''',
    1,
)

start = text.index("def _memory_preflight(device: torch.device) -> dict[str, Any]:")
end = text.index("\ndef _snapshot_report(", start)
replacement = r'''def _budget_bytes(name: str) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return _DEFAULT_DIAGNOSTIC_BUDGET_BYTES
    try:
        gib = float(raw)
    except ValueError as exc:
        raise RuntimeError(f"first-high Sol-local E {name} must be a finite GiB value") from exc
    if not math.isfinite(gib) or not 0.25 <= gib <= 64.0:
        raise RuntimeError(f"first-high Sol-local E {name} must be in [0.25, 64] GiB")
    return int(gib * 1024**3)


def _planned_witness_cpu_bytes() -> int:
    heads = 56
    dim = 128
    total = 0
    for group in _EXPECTED_WITNESS_GROUPS:
        q_rows = _EXPECTED_LOCAL_Q_ROWS[group]
        kv_rows = _EXPECTED_WINDOW_KV_ROWS[group]
        q_blocks = (q_rows + 63) // 64
        k_blocks = (kv_rows + 63) // 64
        route_groups = (k_blocks + 63) // 64
        qkv = (q_rows + 2 * kv_rows) * heads * dim * 2
        saved_outputs = 3 * q_rows * heads * dim * 2
        summaries = 2 * k_blocks * heads * dim * 2
        threshold = q_blocks * heads * 4
        traces = 2 * q_blocks * heads * route_groups * 2 * 4
        route_fields = 2 * q_blocks * heads * k_blocks * 4
        lse = q_rows * heads * 4
        total += qkv + saved_outputs + summaries + threshold + traces + route_fields + lse
    return total


def _projected_cuda_sidecar_bytes() -> int:
    heads = 56
    dim = 128
    group = max(_EXPECTED_WITNESS_GROUPS, key=lambda item: _EXPECTED_LOCAL_Q_ROWS[item] + 2 * _EXPECTED_WINDOW_KV_ROWS[item])
    q_rows = _EXPECTED_LOCAL_Q_ROWS[group]
    kv_rows = _EXPECTED_WINDOW_KV_ROWS[group]
    logical_qkv = (q_rows + 2 * kv_rows) * heads * dim * 2
    # Conservative bound: logical live QKV domain plus six Q-sized BF16 results
    # and 256 MiB for summaries, traces, FP32 tile/reference scratch and runtime
    # workspace. QKV is caller-owned rather than cloned, so this intentionally
    # overstates E's incremental allocation.
    q_output = q_rows * heads * dim * 2
    return logical_qkv + 6 * q_output + 256 * 1024**2


def _cpu_available_bytes() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().available)
    except (ImportError, AttributeError):
        if hasattr(os, "sysconf"):
            try:
                return int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
            except (OSError, ValueError):
                pass
    raise RuntimeError("first-high Sol-local E cannot prove host-memory headroom")


def _process_rss_bytes() -> int | None:
    try:
        import psutil

        return int(psutil.Process().memory_info().rss)
    except (ImportError, AttributeError, OSError):
        return None


def _process_maxrss_bytes() -> int | None:
    try:
        import resource
        import sys

        value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        return value if sys.platform == "darwin" else value * 1024
    except (ImportError, AttributeError, OSError, ValueError):
        return None


def _memory_snapshot(device: torch.device) -> dict[str, Any]:
    try:
        free_cuda, total_cuda = torch.cuda.mem_get_info(device)
    except TypeError:
        with torch.cuda.device(device):
            free_cuda, total_cuda = torch.cuda.mem_get_info()
    return {
        "cuda_allocator_current_bytes": int(torch.cuda.memory_allocated(device)),
        "cuda_allocator_process_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_free_bytes": int(free_cuda),
        "cuda_total_bytes": int(total_cuda),
        "cpu_rss_bytes": _process_rss_bytes(),
        "cpu_process_peak_rss_bytes": _process_maxrss_bytes(),
    }


def _memory_preflight(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("first-high Sol-local E requires CUDA SM120")
    cuda_budget = _budget_bytes(_CUDA_BUDGET_ENV)
    cpu_budget = _budget_bytes(_CPU_BUDGET_ENV)
    planned_cpu = _planned_witness_cpu_bytes()
    projected_cuda = _projected_cuda_sidecar_bytes()
    if planned_cpu > cpu_budget:
        raise RuntimeError(
            "first-high Sol-local E projected CPU witness storage exceeds the configured diagnostic budget: "
            f"{planned_cpu} > {cpu_budget}"
        )
    if projected_cuda > cuda_budget:
        raise RuntimeError(
            "first-high Sol-local E projected CUDA sidecar bound exceeds the configured diagnostic budget: "
            f"{projected_cuda} > {cuda_budget}"
        )
    snapshot = _memory_snapshot(device)
    cpu_available = _cpu_available_bytes()
    if int(snapshot["cuda_free_bytes"]) < cuda_budget:
        raise RuntimeError(
            "first-high Sol-local E has insufficient CUDA headroom for the configured diagnostic budget: "
            f"{snapshot['cuda_free_bytes']} < {cuda_budget}"
        )
    if int(cpu_available) < cpu_budget:
        raise RuntimeError(
            "first-high Sol-local E has insufficient host headroom for the configured diagnostic budget: "
            f"{cpu_available} < {cpu_budget}"
        )
    return {
        "cuda_budget_env": _CUDA_BUDGET_ENV,
        "cpu_budget_env": _CPU_BUDGET_ENV,
        "cuda_budget_bytes": cuda_budget,
        "cpu_budget_bytes": cpu_budget,
        "projected_cuda_sidecar_upper_bound_bytes": projected_cuda,
        "planned_cpu_witness_storage_bytes": planned_cpu,
        "cpu_available_bytes": int(cpu_available),
        "before": snapshot,
    }


def _unique_cpu_tensor_storage_bytes(value: Any) -> int:
    seen: set[tuple[int, int]] = set()

    def visit(item: Any) -> None:
        if torch.is_tensor(item):
            if item.device.type != "cpu":
                return
            storage = item.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            seen.add(key)
            return
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                visit(child)

    visit(value)
    return sum(size for _, size in seen)


def _memory_completion(
    device: torch.device,
    preflight: dict[str, Any],
    evidence: _Sink,
) -> dict[str, Any]:
    after = _memory_snapshot(device)
    before = preflight["before"]
    cpu_evidence = _unique_cpu_tensor_storage_bytes(evidence.items)
    cpu_peak_before = before.get("cpu_process_peak_rss_bytes")
    cpu_peak_after = after.get("cpu_process_peak_rss_bytes")
    cuda_peak_before = int(before["cuda_allocator_process_peak_bytes"])
    cuda_peak_after = int(after["cuda_allocator_process_peak_bytes"])
    return {
        **preflight,
        "after": after,
        "actual_retained_cpu_evidence_storage_bytes": cpu_evidence,
        "retained_cpu_evidence_within_budget": cpu_evidence <= int(preflight["cpu_budget_bytes"]),
        "cuda_allocator_new_process_high_water_bytes": max(0, cuda_peak_after - cuda_peak_before),
        "cpu_new_process_high_water_bytes": (
            None
            if cpu_peak_before is None or cpu_peak_after is None
            else max(0, int(cpu_peak_after) - int(cpu_peak_before))
        ),
        "peak_semantics": (
            "process high-water values are read without resetting global allocator/RSS statistics; "
            "new-high-water deltas are exact only when E exceeds the pre-existing process peak"
        ),
    }
'''
text = text[:start] + replacement + text[end:]

old = '''            sol_counter_isolation = _validate_sol_counter_isolation(record)
            execution_valid = bool(
                topology == expected_topology
'''
new = '''            sol_counter_isolation = _validate_sol_counter_isolation(record)
            memory_report = _memory_completion(replay_latent.device, memory_preflight, evidence_sink)
            memory_valid = bool(memory_report["retained_cpu_evidence_within_budget"])
            execution_valid = bool(
                topology == expected_topology
                and memory_valid
'''
if old not in text:
    raise SystemExit("memory completion insertion anchor not found")
text = text.replace(old, new, 1)

text = text.replace(
    '                "memory_preflight": memory_preflight,\n',
    '                "memory": memory_report,\n',
    1,
)

path.write_text(text, encoding="utf-8")
