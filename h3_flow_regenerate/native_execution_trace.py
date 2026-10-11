"""Optional full-state fingerprints and read-only metadata inside native blocks."""

from __future__ import annotations

import contextlib
import contextvars
import hashlib
import inspect
import itertools
import sys
from collections.abc import Mapping
from pathlib import Path

import torch

OPTION = "h3_flow_low_native_trace_step_v1"
PAIR_OPTION = "h3_flow_low_native_full_pair_block_v1"
MODULE_OPTION = "h3_flow_low_native_module_detail_v1"
POLICY = "h3_low_native_execution_v1"


def validate_step(step):
    if type(step) is not int or not -1 <= step <= 511:
        raise ValueError("low_native_trace_step must be an integer from -1 to 511")
    return step


def validate_pair_block(block):
    if type(block) is not int or not -1 <= block <= 511:
        raise ValueError("low_native_trace_pair_block must be an integer from -1 to 511")
    return block


def validate_module_detail(detail):
    if type(detail) is not bool:
        raise ValueError("low_native_trace_module_detail must be boolean")
    return detail


def _metadata(value, depth=0, remaining=None):
    """Describe cache storage without reading its potentially unwritten contents."""
    remaining = [256] if remaining is None else remaining
    remaining[0] -= 1
    if value is None or type(value) in (bool, int, float):
        return value
    if isinstance(value, str):
        return value if len(value) <= 512 else {"unobserved": "long_string"}
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype), "device": str(value.device)}
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    if depth < 4 and isinstance(value, (tuple, list)) and len(value) <= 128:
        result = []
        for v in value:
            if remaining[0] <= 0:
                return {"items": result, "entries_omitted": len(value) - len(result)}
            result.append(_metadata(v, depth + 1, remaining))
        return result
    if depth < 4 and isinstance(value, dict) and len(value) <= 16:
        result = {}
        for k, v in value.items():
            if remaining[0] <= 0:
                return {"items": result, "entries_omitted": len(value) - len(result)}
            result[str(k)] = _metadata(v, depth + 1, remaining)
        return result
    return {"unobserved": type(value).__module__ + "." + type(value).__qualname__}


def _cache_metadata(buffers):
    caches = {}
    for name in ("_scan", "_delta", "_plans", "_kv", "_activations", "_partition_indices"):
        bank = getattr(buffers, name, None)
        if not isinstance(bank, Mapping):
            caches[name] = {"unobserved": "cache_unavailable"}
            continue
        # VDN bounds these banks; inspect at most the eight newest entries.
        entries = [(key, bank[key]) for key in itertools.islice(reversed(bank), 8)]
        caches[name] = {
            "count": len(bank),
            "entries_omitted": max(0, len(bank) - len(entries)),
            "entries": [{"key": _metadata(key), "storage": _metadata(value)} for key, value in entries],
        }
    return caches


def _vdn_owner(attention):
    forward = getattr(attention, "forward", None)
    if not getattr(forward, "_vdn_forward", False):
        return None
    # Published partitioned bridge preserves the released VDN forward, whose
    # source-backed closure explicitly owns state. Never execute the callable.
    forward = getattr(forward, "_vdn_partitioned_released_forward", forward)
    code, cells = getattr(forward, "__code__", None), getattr(forward, "__closure__", None)
    if code is None or cells is None:
        return None
    for name, cell in zip(code.co_freevars, cells, strict=True):
        if name == "state":
            return getattr(cell.cell_contents, "runtime", None)
    return None


class NativeExecutionTrace:
    def __init__(
        self,
        step,
        hasher,
        *,
        pair_block=-1,
        module_detail=False,
        arm_bytes=2 * 1024**3,
        tensor_bytes=512 * 1024**2,
        max_records=40,
    ):
        self.step = validate_step(step)
        self.pair_block = validate_pair_block(pair_block)
        self.module_detail = validate_module_detail(module_detail)
        if self.module_detail and (self.step < 0 or self.pair_block < 0):
            raise ValueError("low_native_trace_module_detail requires native trace step and pair block")
        self.hasher = hasher
        self.arm_bytes, self.tensor_bytes, self.max_records = arm_bytes, tensor_bytes, max_records
        self.arms = {}
        self.bytes = {}
        self.omitted = {}
        self.sources = {}

    def selected_outputs(self, layers):
        if self.pair_block < 0:
            return sorted({0, layers // 2, layers - 1} | set(range(layers // 2 + 1, layers - 1)))
        # Five complete packed-hidden hashes (input 0, output pair-1,
        # input and output pair, output last) fit under 2 GiB for the
        # 50-block/36,445-row BF16 H3 geometry.
        # Refuse unsupported pair identities rather than silently discarding
        # a sentinel or claiming coverage after a budget overrun.
        if not layers // 2 + 2 <= self.pair_block < layers - 1:
            raise ValueError(
                "low_native_trace_pair_block requires a block between the middle+2 and penultimate transformer blocks"
            )
        return [self.pair_block] if self.module_detail else sorted({self.pair_block - 1, self.pair_block, layers - 1})

    def callable_source(self, function):
        function = getattr(function, "__func__", function)
        if not inspect.isfunction(function):
            return {"type": type(function).__module__ + "." + type(function).__qualname__, "source": "unobserved"}
        code = function.__code__
        path = code.co_filename
        if path not in self.sources and len(self.sources) < 64:
            try:
                source = Path(path)
                if source.stat().st_size > 8 * 1024**2:
                    self.sources[path] = {"unobserved": "source_size_limit"}
                else:
                    self.sources[path] = {"sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
            except OSError as error:
                self.sources[path] = {"unobserved": type(error).__name__}
        return {"module": function.__module__, "name": function.__qualname__, "file": path, "line": code.co_firstlineno}

    def core_sparse_stats(self, tensor, layer, options, previous):
        """Read only the Core H3 block-sparse producer's matching pooled state.

        The producer's (block, packed rows, conditioning UUIDs) cache survives
        model calls. Existence of cached statistics is not proof that this
        particular invocation selected the producer instead of dense fallback.
        """
        if not inspect.isfunction(previous):
            return {"available": False, "reason": "no_function_replacement"}
        try:
            values = inspect.getclosurevars(previous).nonlocals
        except (TypeError, ValueError):
            return {"available": False, "reason": "opaque_replacement"}
        patch = values.get("patch")
        if type(patch).__name__ != "SparseAttnPatch" or type(patch).__module__ != "comfy_extras.nodes_sparse_attention":
            return {"available": False, "reason": "not_core_sparse_patch"}
        pooled = getattr(patch, "pooled", None)
        if not isinstance(pooled, dict):
            return {"available": False, "reason": "no_pooled_mapping"}
        try:
            key = (int(layer), int(tensor.shape[0]), tuple(options.get("uuids", ())))
            hit = pooled.get(key)
        except (TypeError, ValueError):
            return {"available": True, "key_observed": False, "reason": "unhashable_uuids"}
        receipt = {
            "available": True,
            "key_observed": True,
            "present": hit is not None,
            "total_cached_keys": len(pooled),
            "statistics": [],
            "primitive_selection": "unobserved",
        }
        if hit is None:
            return receipt
        if not isinstance(hit, (tuple, list)) or len(hit) != 2:
            receipt["statistics"] = [{"unobserved": "unexpected_pool_shape"}]
            return receipt
        for statistic in hit:
            if not torch.is_tensor(statistic):
                receipt["statistics"].append({"unobserved": "not_tensor"})
                continue
            nbytes = statistic.numel() * statistic.element_size()
            row = {"dtype": str(statistic.dtype), "shape": list(statistic.shape), "nbytes": nbytes}
            if nbytes <= 256 * 1024:
                try:
                    row["sha256"] = self.hasher(statistic)
                except (OSError, RuntimeError, ValueError, TypeError):
                    row["unobserved"] = "hash_failed"
            else:
                row["unobserved"] = "statistic_byte_budget"
            receipt["statistics"].append(row)
        return receipt

    def backend(self, tensor, block, options, previous, layer=None):
        attention = getattr(block, "attn", None)
        vdn = sys.modules.get("vdn_h3.runtime")
        accessor = getattr(vdn, "current_runtime_buffers", None)
        buffers = accessor() if callable(accessor) else None
        owner = _vdn_owner(attention)
        active = getattr(owner, "_active", None)
        owner_pool = active.get() if isinstance(active, contextvars.ContextVar) else None
        prefetch = getattr(buffers, "_prefetcher", None)
        receipt = {
            "vdn": {
                "module_loaded": vdn is not None,
                "active_lease": buffers is not None,
                "owner_id": id(owner) if owner is not None else None,
                "buffers_id": id(buffers) if buffers is not None else None,
                "owner_pool_matches": owner_pool is buffers if owner is not None else None,
                "primary_pool": buffers is getattr(owner, "_primary", None) if owner is not None else None,
                "retain": getattr(buffers, "retain", None),
                "caches": _cache_metadata(buffers),
                "prefetch_generation": getattr(prefetch, "_generation", None),
                "prefetch_target": _metadata(getattr(prefetch, "_index", None)),
                "prefetch_result_and_weights": "unobserved",
                "cache_tensor_contents": "unobserved",
            },
            "cuda_stream": int(torch.cuda.current_stream(tensor.device).cuda_stream)
            if tensor.device.type == "cuda"
            else None,
            "callables": {
                "block": self.callable_source(getattr(block, "forward", None)),
                "attention": self.callable_source(getattr(attention, "forward", None)),
                "previous_replacement": self.callable_source(previous),
                **{
                    key: self.callable_source(options.get(key))
                    for key in (
                        "optimized_attention_override",
                        "vdn_attention_preprocess_v1",
                        "vdn_partitioned_attention_provider_v1",
                    )
                },
            },
            "hooks": {},
            "physical_state_isolation_verified": False,
            "primitive_kernel_internals_and_weight_contents": "unobserved",
        }
        if self.module_detail and layer == self.pair_block:
            receipt["core_sparse_pool"] = self.core_sparse_stats(tensor, layer, options, previous)
        for name, module in (
            ("block", block),
            ("attention", attention),
            ("qkv_proj", getattr(attention, "qkv_proj", None)),
            ("out_proj", getattr(attention, "out_proj", None)),
        ):
            hooks = getattr(module, "_forward_hooks", {})
            receipt["hooks"][name] = {
                "count": len(hooks),
                "entries_omitted": max(0, len(hooks) - 4),
                "callables": [self.callable_source(h) for h in itertools.islice(hooks.values(), 4)],
            }
        sol = sys.modules.get("sol_h3.runtime")
        request_var = getattr(sol, "_REQUEST", None)
        request = request_var.get() if isinstance(request_var, contextvars.ContextVar) else None
        receipt["sol"] = {
            "module_loaded": sol is not None,
            "active_request": request is not None,
            "request_id": id(request) if request is not None else None,
            "kernel_id": id(getattr(request, "kernel", None)) if request is not None else None,
            "backend_transitions": getattr(request, "backend_transitions", None),
            "evaluations": getattr(request, "evaluations", None),
        }
        return receipt

    @contextlib.contextmanager
    def internal_block_hooks(self, trace, layer, layers, block, options, previous):
        """Append read-only probes after existing adapter hooks; restore on exit."""
        selected = (
            self.module_detail
            and self.pair_block == layer
            and trace.arm in {"A", "B_target_projection", "A_replay"}
            and trace.stage == "low"
            and trace.evaluation == self.step
            and trace.native_call == 1
        )
        if not selected:
            yield
            return
        modules = (
            ("attention_out_proj", getattr(getattr(block, "attn", None), "out_proj", None)),
            ("post_attention_norm2", getattr(block, "norm2", None)),
            ("mlp_output", getattr(block, "mlp", None)),
        )
        handles = []
        try:
            for boundary, module in modules:
                if not isinstance(module, torch.nn.Module):
                    continue

                def callback(_module, _args, result, *, name=boundary):
                    if torch.is_tensor(result):
                        self.observe(trace, layer, layers, name, result, block, options, previous)

                handles.append(module.register_forward_hook(callback))
            yield
        finally:
            for handle in reversed(handles):
                handle.remove()

    def observe(self, trace, layer, layers, boundary, tensor, block, options, previous):
        if self.step < 0 or trace.arm not in {"A", "B_target_projection", "A_replay"}:
            return
        if trace.stage != "low" or trace.evaluation != self.step or trace.native_call != 1:
            return
        outputs = self.selected_outputs(layers)
        if self.module_detail:
            full = layer == self.pair_block and boundary in {
                "input",
                "attention_out_proj",
                "post_attention_norm2",
                "mlp_output",
                "output",
            }
        else:
            full_input = boundary == "input" and (layer == 0 or (self.pair_block >= 0 and layer == self.pair_block))
            full_output = (
                boundary == "output"
                and layer in outputs
                and (self.pair_block >= 0 or layer in {0, layers // 2, layers - 1})
            )
            full = full_input or full_output
        # Default mode retains the full sentinels and sampled outputs.
        # Pair mode observes both entry and exit to the suspect block,
        # distinguishing an upstream transition from in-block divergence.
        sampled = (
            not self.module_detail
            and self.pair_block < 0
            and boundary == "output"
            and (layers // 2 < layer < layers - 1)
        )
        if not (full or sampled):
            return
        rows = self.arms.setdefault(trace.arm, [])
        if len(rows) >= self.max_records:
            self.omitted[trace.arm] = self.omitted.get(trace.arm, 0) + 1
            return
        row = {
            "key": [trace.stage, trace.evaluation, trace.native_call, layer, boundary],
            "layers": layers,
            "coverage": "full" if full else "token_rows_stride_32",
            "source_shape": list(tensor.shape),
        }
        rows.append(row)
        try:
            row["backend"] = self.backend(tensor, block, options, previous, layer=layer)
            observed = tensor if full else tensor[::32]
            size = observed.numel() * observed.element_size()
            row.update(shape=list(observed.shape), dtype=str(observed.dtype), nbytes=size)
            used = self.bytes.get(trace.arm, 0)
            if size > self.tensor_bytes or used + size > self.arm_bytes:
                row["unobserved"] = "tensor_or_arm_byte_budget"
                return
            self.bytes[trace.arm] = used + size
            row["sha256"] = self.hasher(observed)
        except (OSError, RuntimeError, ValueError, TypeError, AttributeError) as error:
            row["unobserved"] = type(error).__name__

    def summary(self):
        comparisons = {}
        baseline = self.arms.get("A", [])
        layers = baseline[0]["layers"] if baseline else 0
        expected = []
        if layers > 0:
            if self.module_detail:
                expected = [
                    ["low", self.step, 1, self.pair_block, boundary]
                    for boundary in ("input", "attention_out_proj", "post_attention_norm2", "mlp_output", "output")
                ]
            elif self.pair_block >= 0:
                expected = [
                    ["low", self.step, 1, 0, "input"],
                    ["low", self.step, 1, self.pair_block - 1, "output"],
                    ["low", self.step, 1, self.pair_block, "input"],
                    ["low", self.step, 1, self.pair_block, "output"],
                    ["low", self.step, 1, layers - 1, "output"],
                ]
            else:
                expected = [["low", self.step, 1, 0, "input"]]
                expected.extend(
                    ["low", self.step, 1, layer, "output"]
                    for layer in self.selected_outputs(layers)
                )
        for arm in ("B_target_projection", "A_replay"):
            other = self.arms.get(arm, [])
            aligned = (
                bool(expected)
                and [r["key"] for r in baseline] == expected
                and len(baseline) == len(other)
                and all(
                    a["key"] == b["key"]
                    and a["layers"] == b["layers"]
                    and a["coverage"] == b["coverage"]
                    and a["source_shape"] == b["source_shape"]
                    for a, b in zip(baseline, other, strict=True)
                )
            )
            complete = (
                aligned
                and not self.omitted.get("A", 0)
                and not self.omitted.get(arm, 0)
                and all("sha256" in row for row in baseline + other)
            )
            first = None
            if aligned:
                for a, b in zip(baseline, other, strict=True):
                    if "sha256" in a and "sha256" in b and any(a[k] != b[k] for k in ("shape", "dtype", "sha256")):
                        first = {
                            "key": a["key"],
                            "classification": "packed_hidden_"
                            + a["key"][-1]
                            + ("_sampled" if a["coverage"] != "full" else ""),
                            "coverage": a["coverage"],
                        }
                        break
            comparisons[arm] = {
                "alignment_complete": aligned,
                "selected_boundaries_complete": bool(complete),
                "selected_boundaries_exact": bool(complete and first is None),
                "first_observed_difference": first,
                "missing_selected_native_call": not baseline or not other,
            }
        return {
            "policy": POLICY,
            "enabled": self.step >= 0,
            "low_logical_prediction": self.step,
            "full_pair_block": self.pair_block,
            "internal_module_detail": self.module_detail,
            "native_call": 1,
            "arms": self.arms,
            "comparisons": comparisons,
            "inspection_bytes_charged_per_arm": self.bytes,
            "omitted_records_per_arm": self.omitted,
            "limits": {
                "bytes_per_arm": self.arm_bytes,
                "tensor_bytes": self.tensor_bytes,
                "records_per_arm": self.max_records,
                "source_files": 64,
                "metadata_nodes_per_value": 256,
                "cache_entries": 8,
                "hooks_per_module": 4,
                "intervening_output_token_row_stride": 32 if self.pair_block < 0 else None,
                "full_pair_selected": self.pair_block >= 0,
                "internal_module_detail_selected": self.module_detail,
            },
            "source_files_at_observation": self.sources,
            "physical_state_isolation_verified": False,
            "limitations": [
                (
                    "Module detail hashes suspect block input, attention out_proj output, "
                    "post-attention norm2 output, MLP output and suspect block output. "
                    "Temporary post-forward hooks may perturb timing; missing module calls fail completeness."
                    if self.module_detail
                    else "Pair mode hashes block 0 input, pair-1 output, pair input and output, "
                    "and final output in full; other block boundaries are unobserved."
                    if self.pair_block >= 0
                    else "Block zero input and first/middle/final outputs are hashed in full; "
                    "intervening outputs after the middle block use every 32nd token row."
                ),
                "Sampled equality does not establish entire-block equality or an exact first differing block.",
                "A first observed difference bounds an interval; "
                "intervening blocks and primitive internals are unobserved.",
                "Core sparse pooled statistics are read-only and per selected key; "
                "different values do not prove causality.",
                "Cache keys, pool IDs and hook inventories do not establish tensor/weight or execution-state equality.",
                "Host hashing synchronizes observed data and can perturb timing; "
                "disappearance of a failure is inconclusive.",
            ],
        }
