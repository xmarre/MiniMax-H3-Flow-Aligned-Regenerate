"""Default-off observation of the actual Core H3 comfy-kitchen token admission.

This uses a thread-local Python profiling callback to inspect the *installed*
sol_attn_core call after it returns. Neither Core nor its CUDA extension is
replaced. GPU-to-host copies synchronize after the already-enqueued core call.
The receipt excludes unwritten token-ID tails.
"""

from __future__ import annotations

import contextlib
import hashlib
import sys

import torch

POLICY = "h3_core_sparse_token_admission_v1"
MAX_CAPTURE_BYTES = 32 * 1024**2
SOURCE_SUFFIX = "/comfy_kitchen/backends/cuda/__init__.py"


def _slot(workspace, plan, name, dtype, shape):
    count = 1
    for length in shape:
        count *= length
    size = torch.empty((), dtype=dtype).element_size() * count
    offset = plan[name]
    if type(offset) is not int or offset < 0 or offset + size > workspace.numel():
        raise ValueError(f"invalid sparse workspace slot: {name}")
    return workspace.narrow(0, offset, size).view(dtype).reshape(shape)


def _sha(tensor):
    data = tensor.detach().contiguous().reshape(-1).cpu().numpy()
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def capture_token_admission(locals_):
    """Observe only completed kernel outputs; no replay or producer mutation."""
    workspace, plan = locals_["ws"], locals_["p"]
    t, h, aug = (locals_[name] for name in ("t", "h", "token_aug"))
    if (
        type(t) is not int
        or type(h) is not int
        or type(aug) is not int
        or t <= 0
        or h <= 0
        or aug < 0
        or not isinstance(plan, dict)
        or not torch.is_tensor(workspace)
        or workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or workspace.device.type != "cuda"
    ):
        raise ValueError("unqualified Core sparse workspace")
    receipt = {"t": t, "h": h, "token_aug": aug, "producer": "comfy_kitchen.sol_attn_chunked"}
    if aug == 0:
        return {**receipt, "active": False, "complete": True}
    nq = plan["NQ"]
    ng = (nq + 1) // 2
    if type(nq) is not int or not 0 < nq <= (t + 63) // 64 or aug > 256:
        raise ValueError("unqualified token grouping or budget")
    bytes_read = h * ng * (4 + 128 * 4 + aug * 4)
    if bytes_read > MAX_CAPTURE_BYTES:
        raise ValueError("token receipt exceeds 32 MiB budget")
    counts = _slot(workspace, plan, "tokCnt", torch.int32, (h, ng))
    histogram = _slot(workspace, plan, "tokHist", torch.int32, (h, ng, 128))
    if bool(((counts < 0) | (counts > t)).any()):
        raise ValueError("invalid token reservation counts")
    if bool((histogram < 0).any()):
        raise ValueError("invalid token histogram counts")
    indices = _slot(workspace, plan, "tokIdx", torch.int32, (h, ng, aug))
    valid = torch.arange(aug, device=workspace.device) < counts.clamp(max=aug)[..., None]
    # Gather only initialized entries. Do not read or hash uninitialized tails
    # from empty or partially occupied rows, even under memory-sanitizer runs.
    populated = valid.nonzero(as_tuple=True)
    canonical = torch.zeros((h, ng, aug), dtype=torch.int32, device=workspace.device)
    canonical[populated] = indices[populated]
    windows = histogram.to(torch.int64).sum(-1)
    overflow = counts > aug
    return {
        **receipt,
        "active": True,
        "complete": True,
        "groups": h * ng,
        "bytes_observed": bytes_read,
        "max_reservations": int(counts.max()),
        "overflow_groups": int(overflow.sum()),
        "at_budget_groups": int((counts == aug).sum()),
        "max_window_candidates": int(windows.max()),
        "max_top_bin": int(histogram[..., -1].max()),
        "reservations_sha256": _sha(counts),
        "histogram_sha256": _sha(histogram),
        "retained_ids_sha256": _sha(canonical),
    }


class CoreTokenAdmissionObserver:
    """One selected actual Core block invocation per replay arm."""

    def __init__(self):
        self.records = {}

    @contextlib.contextmanager
    def observe(self, arm):
        record = {"policy": POLICY, "calls": 0, "complete": False}
        self.records[arm] = record
        previous = sys.getprofile()
        if previous is not None:
            record["unobserved"] = "existing_python_profiler"
            yield
            return

        def callback(frame, event, arg):
            if event not in ("c_return", "c_exception"):
                return
            if getattr(arg, "__name__", None) != "sol_attn_core":
                return
            if frame.f_code.co_name != "sol_attn_chunked":
                return
            if not frame.f_code.co_filename.replace("\\", "/").endswith(SOURCE_SUFFIX):
                return
            record["calls"] += 1
            if event == "c_exception":
                record["unobserved"] = "core_cuda_exception"
                return
            if record["calls"] > 1:
                record["unobserved"] = "multiple_core_calls"
                return
            try:
                record.update(capture_token_admission(frame.f_locals))
            except (KeyError, ValueError, TypeError, RuntimeError) as exc:
                record["unobserved"] = type(exc).__name__ + ": " + str(exc)[:160]

        sys.setprofile(callback)
        try:
            yield
        finally:
            sys.setprofile(previous)
            if record["calls"] != 1:
                record["complete"] = False
                record.setdefault("unobserved", "missing_or_multiple_core_calls")
            elif "unobserved" in record:
                record["complete"] = False

    def summary(self):
        baseline = self.records.get("A")
        pairs = {}
        for arm in ("B_target_projection", "A_replay"):
            other = self.records.get(arm)
            valid = bool(
                baseline
                and other
                and baseline.get("complete")
                and other.get("complete")
                and (baseline["t"], baseline["h"], baseline["token_aug"])
                == (other["t"], other["h"], other["token_aug"])
            )
            pairs[arm] = {
                "comparable": valid,
                "reservations_equal": (
                    baseline.get("reservations_sha256") == other.get("reservations_sha256") if valid else None
                ),
                "histogram_equal": (
                    baseline.get("histogram_sha256") == other.get("histogram_sha256") if valid else None
                ),
                "retained_ids_equal": (
                    baseline.get("retained_ids_sha256") == other.get("retained_ids_sha256") if valid else None
                ),
            }
        return {
            "policy": POLICY,
            "selected_native_core_calls": self.records,
            "comparisons": pairs,
            "limitations": (
                "Reads final installed-core token workspace after C return. "
                "Does not prove equality of producer inputs or identify an internal CUDA launch. "
                "GPU synchronization and profiling can affect scheduling. "
                "Only one selected block/native prediction is observed per arm."
            ),
        }
