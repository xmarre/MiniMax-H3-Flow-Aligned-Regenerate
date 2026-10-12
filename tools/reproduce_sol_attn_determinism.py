#!/usr/bin/env python3
"""Fixed-input CUDA diagnostic for comfy-kitchen 0.2.37; no model is loaded.

Run from the ComfyUI Python environment. See docs/sol-attn-determinism.md.
The installed Python wrapper is cloned with private globals to observe its
actual extension calls; the original wrapper/module and production state are
never patched. Optional source stages run in a separate diagnostic library.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import math
import shutil
import subprocess
import sys
import types
import urllib.request
from pathlib import Path

SOURCE_COMMIT = "be003b7c23c5b01328657955b8bc5d3f073d868e"
SOURCE_PREFIX = "comfy_kitchen/backends/cuda/sage_attention/"
MANIFEST = Path(__file__).with_name("sol_attn_source_manifest.json")


def digest(tensor):
    cpu = tensor.detach().contiguous().cpu()
    # Singleton dimensions may retain non-unit strides even after contiguous().
    # The logical elements are contiguous; normalize the flattened stride before
    # reinterpreting BF16/integer bytes without converting their values.
    flat = cpu.reshape(-1).as_strided((cpu.numel(),), (1,))
    data = flat.view(dtype=torch.uint8).numpy()
    return hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def receipt(tensor):
    return {"shape": list(tensor.shape), "dtype": str(tensor.dtype), "sha256": digest(tensor)}


def histogram_threshold(hist, reference, budget):
    """CPU source model: threshold admits whole bins; the final bin is clamped.

    Counts are integers; float threshold rounding is left to the CUDA runner.
    This models admission semantics, and is not evidence of a runtime race.
    """
    if len(hist) != 128 or any(n < 0 for n in hist):
        raise ValueError("expected 128 nonnegative bin counts")
    acc = 0
    b = 127
    while b >= 0:
        acc += hist[b]
        if acc > budget:
            break
        b -= 1
    edge = (b + 1) * 0.25 if b + 1 <= 96 else 24 + (b + 1 - 96) * 2
    return reference - 8 + edge


def view_slot(ws, plan, name, dtype, shape):
    n = math.prod(shape)
    size = torch.empty((), dtype=dtype).element_size()
    start = plan[name]
    if start < 0 or start + n * size > ws.numel():
        raise ValueError(f"out-of-range workspace slot {name}")
    return ws[start : start + n * size].view(dtype).reshape(shape)


def canonical_indices(indices, counts, capacity):
    """Exclude undefined list tails; preserve row boundaries and exact IDs."""
    if bool(((counts < 0) | (counts > 2**30)).any()):
        raise ValueError("invalid reservation count")
    valid = torch.arange(capacity, device=indices.device) < counts.clamp(max=capacity)[..., None]
    positions = valid.nonzero(as_tuple=True)
    result = torch.zeros(indices.shape, dtype=torch.int64, device=indices.device)
    # Gather only written entries, so diagnostic kernels do not load uninitialized
    # tails. Signed views let PyTorch index the underlying unsigned ID storage.
    values = indices[positions].to(torch.int64)
    if indices.dtype == torch.int16:
        values &= 0xFFFF
    result[positions] = values
    return result


def sorted_token_indices(indices, counts, capacity):
    result = canonical_indices(indices, counts, capacity)
    valid = torch.arange(capacity, device=indices.device) < counts.clamp(max=capacity)[..., None]
    # Sort only the set that was admitted, excluding unused slots from ordering.
    ordered = result.masked_fill(~valid, 2**63 - 1).sort(-1).values
    return ordered.masked_fill(~valid, 0)


def workspace_observations(ws, p, t, h, aug, stage):
    """Only initialized logical elements, never padding or unused list tails."""
    nq, npad, tp = p["NQ"], p["NPAD"], p["Tp"]
    ng, words = (nq + 1) // 2, (nq + 31) // 32
    result = {}

    def add(name, dtype, shape, logical=None):
        x = view_slot(ws, p, name, dtype, shape)
        result[name] = x if logical is None else logical(x)

    if stage == "producer":
        add("qiP", torch.int8, (t, h, 128))
        add("qs", torch.float32, (t, h))
        # Ragged-block masks/zeros are initialized. Physical key axes are
        # permuted, so truncating to T would omit live quantized operands.
        add("kiP", torch.int8, (h, tp, 128))
        add("ksb", torch.float32, (h, tp, 2))
        add("vTi", torch.int8, (h, 128, tp))
        if aug:
            add("vRow", torch.int8, (h, tp, 128))
        add("statsV", torch.float32, (h, 128))
        add("qmean", torch.float32, (h, npad, 128), lambda x: x[:, :nq])
        # The first scratch slot contains post-RoPE K block sums.
        add("scratch", torch.float32, (h, npad, 128), lambda x: x[:, :nq])
        add("vcT", torch.bfloat16, (h, 128, npad))
        add("cen8", torch.int8, (h, nq, 128))
        add("cens", torch.float32, (h, nq))
    elif stage == "finish":
        add("kciP", torch.int8, (h, npad, 128))
        add("kcs", torch.float32, (h, npad))
        add("thr", torch.float32, (h, nq))
        add("vsc", torch.float32, (h, 128))
    elif stage == "route":
        counts = view_slot(ws, p, "cnt", torch.int32, (h, nq))
        ids = view_slot(ws, p, "idx", torch.int16, (h, nq, nq))
        result["cnt"] = counts
        result["idx"] = canonical_indices(ids, counts, nq)
        if aug:
            add("tokRef", torch.float32, (h, nq))
            add("tokBits", torch.uint32, (h, nq, words))
    elif stage in ("token", "token_group", "token_tiles", "token_hist", "token_admit", "token_sort") and aug:
        if stage == "token_group":
            for name, dtype, shape in (
                ("gCen8", torch.int8, (h, ng, 128)),
                ("gCens", torch.float32, (h, ng)),
                ("gRef", torch.float32, (h, ng)),
                ("gBits", torch.uint32, (h, ng, words)),
            ):
                add(name, dtype, shape)
            return result
        if stage == "token_tiles":
            groups = (ng + 63) // 64
            counts = view_slot(ws, p, "tokTileCnt", torch.int32, (h, groups))
            ids = view_slot(ws, p, "tokTiles", torch.int16, (h, groups, nq))
            return {"tokTileCnt": counts, "tokTiles": canonical_indices(ids, counts, nq)}
        if stage == "token_hist":
            add("tokHist", torch.uint32, (h, ng, 128))
            return result
        counts = view_slot(ws, p, "tokCnt", torch.int32, (h, ng))
        ids = view_slot(ws, p, "tokIdx", torch.int32, (h, ng, aug))
        result["tokCnt"] = counts
        result["tokIdx"] = sorted_token_indices(ids, counts, aug)
        if stage == "token_admit":
            split = max(1, min(8, (512 + (ng + 63) // 64 - 1) // ((ng + 63) // 64)))
            add("tokPartO", torch.bfloat16, (split, h, ng, 128))
            add("tokPartM", torch.float32, (split, h, ng))
            add("tokPartL", torch.float32, (split, h, ng))
            return result
        if stage == "token_sort":
            return result
        add("tokHist", torch.uint32, (h, ng, 128))
        add("gCen8", torch.int8, (h, ng, 128))
        add("gCens", torch.float32, (h, ng))
        add("gRef", torch.float32, (h, ng))
        add("gBits", torch.uint32, (h, ng, words))
    if stage in ("route", "token"):
        add("oPart", torch.bfloat16, (h, nq, 128))
        add("mPart", torch.float32, (h, nq))
        add("lPart", torch.float32, (h, nq))
    return result


def token_summary(tensors, t, aug):
    if not aug:
        return {"enabled": False}
    counts = tensors["tokCnt"].to(torch.int64)
    hist = tensors["tokHist"].to(torch.int64)
    # Histogram candidates are only those within/above the scoring window.
    # Count ALL unrouted live token candidates separately using the group bitmap.
    bits = tensors["gBits"].to(torch.int64)
    blocks = (t + 63) // 64
    lengths = torch.full((blocks,), 64, device=bits.device, dtype=torch.int64)
    lengths[-1] = t - (blocks - 1) * 64
    b = torch.arange(blocks, device=bits.device)
    live = ((bits[..., b // 32] >> (b % 32)) & 1) * lengths
    window = hist.sum(-1)
    return {
        "enabled": True,
        "groups": counts.numel(),
        "candidate_tokens_before_score_window": receipt(live.sum(-1)),
        "window_candidate_counts": receipt(window),
        "max_window_candidates": int(window.max()),
        "max_reservations": int(counts.max()),
        "overflow_groups": int((counts > aug).sum()),
        "at_budget_groups": int((counts == aug).sum()),
        "total_reservations": int(counts.sum()),
        "total_retained": int(counts.clamp(max=aug).sum()),
        "reservations_sha256": digest(counts),
        "retained_ids_sha256": digest(tensors["tokIdx"]),
        "top_bin_counts_sha256": digest(hist[..., -1]),
    }


class Comparisons:
    """Byte-exact GPU comparisons; SHA256 only for baseline and changed bytes.

    Reference snapshots remain on the tensor's device until the case ends.
    Reusing a baseline SHA256 is valid only after a byte-view equality check,
    which distinguishes signed zero and floating-point NaN payloads.
    """

    def __init__(self):
        self.baseline = {}
        self.baseline_sha = {}
        self.first_difference = None

    def observe(self, key, tensors, iteration):
        report = {}
        for name, x in tensors.items():
            current = x.detach().contiguous()
            tag = (key, name)
            if tag not in self.baseline:
                reference = current.clone()
                self.baseline[tag] = reference
                self.baseline_sha[tag] = digest(reference)
                same = True
            else:
                reference = self.baseline[tag]
                if current.shape != reference.shape or current.dtype != reference.dtype:
                    raise ValueError(f"observation shape/dtype changed: {tag}")
                # torch.equal(float) would treat +0 and -0 as identical and
                # NaNs as different regardless of payload; compare raw bytes.
                same = torch.equal(
                    current.reshape(-1).view(torch.uint8),
                    reference.reshape(-1).view(torch.uint8),
                )
            row = {
                "shape": list(current.shape),
                "dtype": str(current.dtype),
                "sha256": self.baseline_sha[tag] if same else digest(current),
                "bitwise_equal": same,
            }
            if not same:
                if self.first_difference is None:
                    self.first_difference = {"iteration": iteration, "stage": key, "tensor": name}
                # Keep large reductions on the GPU. BF16 and the integer index
                # ranges in this diagnostic are represented exactly in FP32.
                diff = current.to(torch.float32) - reference.to(torch.float32)
                row.update(rms=float(diff.square().mean().sqrt()), max_abs=float(diff.abs().max()))
                if name in ("tokIdx", "idx"):
                    location = (current != reference).any(-1).nonzero()[0].tolist()
                    row["first_changed_list"] = {
                        "location": location,
                        "before": reference[tuple(location)].tolist(),
                        "after": current[tuple(location)].tolist(),
                    }
            report[name] = row
        return report


class ExtensionObserver:
    """Private extension proxy, receiving tensors through a private identity wrap."""

    def __init__(self, backend, t, h, aug, comparison, iteration, cohort="cold"):
        self.backend, self.t, self.h, self.aug = backend, t, h, aug
        self.comparison, self.iteration = comparison, iteration
        self.cohort = cohort
        self.plan = backend._C.sol_attn_plan(1, t, h, token_aug=aug)
        self.records = {}
        self.pass_number = 0
        self.frozen = None
        self.operands = None

    def __getattr__(self, name):
        target = getattr(self.backend._C, name)

        def call(*args, **kwargs):
            if name == "sol_producer_begin":
                if self.pass_number:
                    self.observe_producer(args[0], f"producer_pass_{self.pass_number}")
                self.pass_number += 1
            if name == "sol_attn_core":
                ws, _, vsc = args[:3]
                self.observe_producer(ws, f"producer_pass_{self.pass_number}")
                self.frozen = ws.clone()
                self.vscale = vsc.clone()
            if name == "sol_producer_chunk" and args[9] == 0:
                # Effective cast norm weights, packed RoPE, and current bootstrap
                # stats are observed once per pass; all original QKV is separately
                # hashed by the driver and never generated again between calls.
                tensors = dict(zip(("fab", "qw", "kw", "kmean", "vscale"), args[2:7], strict=True))
                self.effective_stats = (args[5].clone(), args[6].clone())
                self.records[f"operands_pass_{self.pass_number}"] = self.comparison.observe(
                    f"{self.cohort}/operands_pass_{self.pass_number}", tensors, self.iteration
                )
            wrap = self.backend._wrap_for_dlpack
            converted = tuple(wrap(x) if isinstance(x, torch.Tensor) else x for x in args)
            converted_kw = {k: wrap(x) if isinstance(x, torch.Tensor) else x for k, x in kwargs.items()}
            value = target(*converted, **converted_kw)
            if name == "sol_attn_core":
                for stage in ("finish", "route", "token"):
                    observed = workspace_observations(args[0], self.plan, self.t, self.h, self.aug, stage)
                    # Final o/m/l are token-merged, so only tag them once.
                    if stage == "route":
                        observed = {k: v for k, v in observed.items() if k not in ("oPart", "mPart", "lPart")}
                    self.records[f"post_core_{stage}"] = self.comparison.observe(
                        f"{self.cohort}/post_core_{stage}", observed, self.iteration
                    )
                    if stage == "token":
                        self.records["token_summary"] = token_summary(observed, self.t, self.aug)
            return value

        return call

    def observe_producer(self, ws, key):
        self.records[key] = self.comparison.observe(
            f"{self.cohort}/{key}",
            workspace_observations(ws, self.plan, self.t, self.h, self.aug, "producer"),
            self.iteration,
        )

    def wrapper(self):
        original = self.backend.sol_attn_chunked
        namespace = dict(original.__globals__)
        namespace.update(_C=self, _wrap_for_dlpack=lambda x: x)
        cloned = types.FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
        cloned.__kwdefaults__ = original.__kwdefaults__
        return cloned


def fixture(t, h, case, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    if case in ("overflow", "flat"):
        u = torch.randn(h, 128, generator=g, device="cuda")
        u /= u.norm(dim=-1, keepdim=True)
        q = u.expand(t, h, 128).clone()
        signs = torch.where(torch.arange(t, device="cuda") % 2 == 0, 1.0, -1.0)
        k = q * signs[:, None, None] if case == "overflow" else q.clone()
        v = torch.randn(t, h, 128, generator=g, device="cuda")
        qkv = torch.cat([x.reshape(t, -1) for x in (q, k, v)], dim=-1).to(torch.bfloat16)
        c = 30.0 if case == "overflow" else 1.0
        norm = torch.full((128,), c / math.sqrt(128), dtype=torch.bfloat16, device="cuda")
        angle = torch.zeros(t, 48, device="cuda")
    else:
        qkv = torch.randn(t, 3 * h * 128, generator=g, device="cuda", dtype=torch.bfloat16) * 0.5
        norm = torch.ones(128, dtype=torch.bfloat16, device="cuda")
        angle = torch.randn(t, 48, generator=g, device="cuda")
    f = torch.stack((angle.cos(), -angle.sin(), angle.sin(), angle.cos()), -1)
    rope = f.reshape(1, t, 1, 48, 2, 2).contiguous()
    return qkv, rope, (norm.clone(), norm.clone())


def verified_source(directory, download=False):
    manifest = json.loads(MANIFEST.read_text())
    if manifest["commit"] != SOURCE_COMMIT:
        raise ValueError("source manifest commit mismatch")
    for name, expected in manifest["files"].items():
        path = directory / name
        if download and not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://raw.githubusercontent.com/Comfy-Org/comfy-kitchen/{SOURCE_COMMIT}/{name}"
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"source mismatch: {name}")
    return manifest


def build_probe(directory, source, arch=None):
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("--stages requires nvcc; omit it for installed-primitive evidence only")
    directory.mkdir(parents=True, exist_ok=True)
    verified_source(source, download=True)
    if arch is None:
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
    root = source / SOURCE_PREFIX
    units = [root / f"sol_attn_{name}.cu" for name in ("preprocess", "producer", "vtranspose", "route", "exact")]
    libpath = directory / "sol_probe.so"
    command = [
        nvcc,
        "-shared",
        "-Xcompiler=-fPIC",
        "-O3",
        "--use_fast_math",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        f"-arch={arch}",
        "-I",
        str(root),
        str(Path(__file__).with_name("sol_attn_stage_probe.cu")),
        str(Path(__file__).with_name("sol_token_stage_probe.cu")),
        *map(str, units),
        "-o",
        str(libpath),
    ]
    # Always rebuild: a stale binary must not masquerade as verified source.
    # NVIDIA's pip toolkit stores static libraries in lib; system toolkits
    # commonly use lib64. Supply existing directories without changing linking.
    for name in ("lib", "lib64"):
        libdir = Path(nvcc).resolve().parent.parent / name
        if libdir.is_dir():
            command.extend(("-L", str(libdir)))
    subprocess.run(command, check=True)
    lib = ctypes.CDLL(str(libpath.resolve()))
    lib.sol_probe_error.restype = ctypes.c_char_p
    lib.sol_probe_stage.argtypes = [
        ctypes.c_int,
        *([ctypes.c_void_p] * 5),
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_float,
        ctypes.c_float,
        *([ctypes.c_int] * 5),
        ctypes.c_uint64,
    ]
    lib.sol_probe_stage.restype = ctypes.c_int
    return lib, {
        "command": command,
        "compiler": subprocess.check_output([nvcc, "--version"], text=True),
        "sha256": hashlib.sha256(libpath.read_bytes()).hexdigest(),
        "arch": arch,
    }


def replay_core(backend, observer, config, repeats, probe=None):
    """Same complete producer workspace and vscale on every repeat.

    Installed-core repeats have no intra-kernel synchronization. Source-stage
    repeats explicitly observe finish/route/token/exact and require exact output
    parity against the installed core before permitting attribution.
    """
    frozen, vs, p = observer.frozen, observer.vscale, observer.plan
    t, h, aug = config["t"], config["h"], config["aug"]
    comparison, source_comparison, rows = Comparisons(), Comparisons(), []
    reuse = torch.empty_like(frozen)
    installed_reference = None
    for index in range(repeats):
        mode = "fresh" if index % 2 == 0 else "reused"
        ws = torch.empty_like(frozen) if mode == "fresh" else reuse
        ws.copy_(frozen)
        out = torch.empty(1, t, h, 128, device="cuda", dtype=torch.bfloat16)
        km, vamax = (torch.empty(h, 128, device="cuda") for _ in range(2))
        stream = torch.cuda.current_stream().cuda_stream
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        backend._C.sol_attn_core(
            *(backend._wrap_for_dlpack(x) for x in (ws, out, vs, km, vamax)),
            1,
            t,
            h,
            config["tau"],
            128**-0.5,
            *config["sb"],
            0,
            0,
            stream,
            token_aug=aug,
        )
        end.record()
        end.synchronize()
        row = {"iteration": index, "workspace": mode, "core_ms": start.elapsed_time(end)}
        row["installed_output"] = comparison.observe(
            "installed_output", {"out": out, "kmean": km, "vamax": vamax}, index
        )
        if installed_reference is None:
            installed_reference = out.clone()
        for stage in ("finish", "route", "token"):
            obs = workspace_observations(ws, p, t, h, aug, stage)
            if stage == "route":
                obs = {k: v for k, v in obs.items() if k not in ("oPart", "mPart", "lPart")}
            row[f"installed_post_{stage}"] = comparison.observe(f"installed_post_{stage}", obs, index)
            if stage == "token":
                row["token_summary"] = token_summary(obs, t, aug)
        if probe is not None:
            ws.copy_(frozen)
            phases = [(0, "finish"), (1, "route")]
            if aug:
                phases.extend((20 + n, f"token_{s}") for n, s in enumerate(("group", "tiles", "hist", "admit", "sort")))
                phases.append((25, "token"))
            phases.append((3, "exact"))
            for stage_number, stage in phases:
                code = probe.sol_probe_stage(
                    stage_number,
                    ws.data_ptr(),
                    out.data_ptr(),
                    vs.data_ptr(),
                    km.data_ptr(),
                    vamax.data_ptr(),
                    t,
                    h,
                    config["tau"],
                    128**-0.5,
                    *config["sb"],
                    0,
                    0,
                    aug,
                    stream,
                )
                if code:
                    raise RuntimeError(probe.sol_probe_error().decode())
                torch.cuda.synchronize()
                obs = (
                    {"out": out, "kmean": km, "vamax": vamax}
                    if stage == "exact"
                    else workspace_observations(ws, p, t, h, aug, stage)
                )
                if stage == "finish":
                    obs["kmean_next"] = km
                row[f"source_{stage}"] = source_comparison.observe(f"source_{stage}", obs, index)
            row["source_matches_installed_reference"] = digest(out) == digest(installed_reference)
        rows.append(row)
    return {
        "runs": rows,
        "first_difference": comparison.first_difference,
        "source_first_difference": source_comparison.first_difference,
        "source_earliest_divergent_stage": next(
            (
                stage
                for stage in (
                    "finish",
                    "route",
                    "token_group",
                    "token_tiles",
                    "token_hist",
                    "token_admit",
                    "token_sort",
                    "token",
                    "exact",
                )
                if any(not item["bitwise_equal"] for row in rows for item in row.get(f"source_{stage}", {}).values())
            ),
            None,
        ),
        "source_parity_verified": probe is not None and all(r["source_matches_installed_reference"] for r in rows),
        "workspace_bytes": frozen.numel(),
    }


def run_case(backend, case, t, h, aug, repeats, probe):
    qkv, rope, weights = fixture(t, h, case, seed=119188)
    inputs = {"qkv": receipt(qkv), "rope": receipt(rope), "qw": receipt(weights[0]), "kw": receipt(weights[1])}
    config = {"case": case, "t": t, "h": h, "aug": aug, "tau": 1.0, "sb": [0, 101] if t == 36445 else [0, 0]}
    comparison, rows = Comparisons(), []
    frozen_stats = None
    core_result = None
    for i in range(repeats):
        cohort = "cold" if i % 4 < 2 else "frozen_stats"
        observer = ExtensionObserver(backend, t, h, aug, comparison, i, cohort)
        # Each pair uses a new identity then that same identity: cold/hot packed
        # RoPE cache, with no mutation or speculative cache clearing.
        if i % 2 == 0:
            freqs = rope.clone()
        stats = (None, None) if cohort == "cold" else frozen_stats
        output, km, vs = observer.wrapper()(
            lambda: iter(qkv.split(4096)),
            t,
            h,
            freqs,
            weights,
            kmean=stats[0],
            vscale=stats[1],
            tau=config["tau"],
            topk_ratio=0.0,
            sink_blocks=config["sb"],
            sink_q=[0, 0],
            rope_eps=1e-5,
            token_aug=aug,
        )
        # Cold and explicitly frozen-stat paths are compared within their own
        # cohorts. Their output equality with EACH OTHER is not an invariant.
        rows.append(
            {
                "iteration": i,
                "cohort": cohort,
                "producer_passes": observer.pass_number,
                "boundaries": observer.records,
                "output": comparison.observe(cohort, {"out": output, "km": km, "vs": vs}, i),
            }
        )
        if frozen_stats is None:
            # These are the effective inputs of the actual producer, not next
            # step's returned statistics; obtain them from the observer below.
            frozen_stats = observer.effective_stats
            core_result = replay_core(backend, observer, config, repeats, probe)
    after = {"qkv": receipt(qkv), "rope": receipt(rope), "qw": receipt(weights[0]), "kw": receipt(weights[1])}
    if inputs != after:
        raise RuntimeError("immutable fixture was mutated")
    overflow_counts = [r["token_summary"]["overflow_groups"] for r in core_result["runs"]] if aug else []
    return {
        "config": config,
        "inputs": inputs,
        "wrapper_runs": rows,
        "wrapper_first_difference": comparison.first_difference,
        "frozen_core": core_result,
        "overflow_fixture_verified": min(overflow_counts) > 0 if case == "overflow" and aug else None,
        "unsaturated_fixture_verified": max(overflow_counts) == 0 if case == "flat" and aug else None,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--production-geometry", action="store_true")
    parser.add_argument(
        "--stages", action="store_true", help="compile exact pinned source stage boundaries (needs nvcc)"
    )
    args = parser.parse_args(argv)
    if args.repeats < 8:
        parser.error("use at least 8 repetitions (default 32)")
    global torch
    import torch

    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable; no primitive experiment was executed")
    version = importlib.metadata.version("comfy-kitchen")
    if version != "0.2.37":
        parser.error(f"requires comfy-kitchen 0.2.37, found {version}")
    from comfy_kitchen.backends import cuda as backend

    expected = json.loads(MANIFEST.read_text())["files"]["comfy_kitchen/backends/cuda/__init__.py"]
    if hashlib.sha256(Path(backend.__file__).read_bytes()).hexdigest() != expected:
        parser.error("installed CUDA wrapper differs from pinned v0.2.37 source")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    props = torch.cuda.get_device_properties(0)
    report = {
        "schema": 1,
        "source_commit": SOURCE_COMMIT,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "kitchen": version,
        "gpu": props.name,
        "capability": list(torch.cuda.get_device_capability()),
        "extension": receipt_file(Path(backend._C.__file__)),
        "repeats": args.repeats,
        "scope": "synthetic primitive; not captured owner operands or end-to-end validation",
        "cases": [],
    }
    probe = None
    try:
        if args.stages:
            directory = args.output.parent / "sol-probe-build"
            probe, build = build_probe(directory, directory / "source")
            report["source_probe"] = build
        cases = [(c, 4196 if c == "random" else 4096, 2) for c in ("flat", "overflow", "random")]
        if args.production_geometry:
            cases.extend((c, 36445, 56) for c in ("overflow", "random"))
        for case, t, h in cases:
            for aug in (256, 0):
                torch.cuda.reset_peak_memory_stats()
                result = run_case(backend, case, t, h, aug, args.repeats, probe)
                result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["cases"].append(result)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    f"{case} T={t} H={h} token_aug={aug}: "
                    f"first frozen-core difference={result['frozen_core']['first_difference']}",
                    flush=True,
                )
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    # A reproducible failure is a successful diagnostic result, but deliberately
    # a failing exit status for deterministic regression use.
    if any(
        c["overflow_fixture_verified"] is False or c["unsaturated_fixture_verified"] is False for c in report["cases"]
    ):
        return 2
    return (
        1
        if any(
            c["wrapper_first_difference"]
            or c["frozen_core"]["first_difference"]
            or c["frozen_core"]["source_first_difference"]
            for c in report["cases"]
        )
        else 0
    )


def receipt_file(path):
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


if __name__ == "__main__":
    sys.exit(main())
