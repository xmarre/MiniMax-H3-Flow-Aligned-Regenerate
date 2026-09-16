# ruff: noqa: I001
# Import order is intentional: the PR35 checkpoint diagnostic must install first
# so the learned-anchor validation wrapper can sit outside it and make those
# diagnostics observe the transported target-grid guidance representation. The
# execution-contract recorder is imported after both; its runtime-observation
# extension is imported immediately after the base recorder. The same-state
# replay, W, and E first-high diagnostics compose only additional wrappers around
# that recorder.
import importlib
import sys
from pathlib import Path

try:
    from .h3_flow_regenerate.decode_context import (
        NODE_CLASS_MAPPINGS as DECODE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.decode_context import (
        NODE_DISPLAY_NAME_MAPPINGS as DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.handoff_checkpoint_diagnostic import (
        NODE_CLASS_MAPPINGS as HANDOFF_DIAGNOSTIC_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.handoff_checkpoint_diagnostic import (
        NODE_DISPLAY_NAME_MAPPINGS as HANDOFF_DIAGNOSTIC_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.guidance_anchor_transport_validation import (
        NODE_CLASS_MAPPINGS as GUIDANCE_ANCHOR_VALIDATION_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.guidance_anchor_transport_validation import (
        NODE_DISPLAY_NAME_MAPPINGS as GUIDANCE_ANCHOR_VALIDATION_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.execution_contract_diagnostics import (
        NODE_CLASS_MAPPINGS as EXECUTION_CONTRACT_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.execution_contract_diagnostics import (
        NODE_DISPLAY_NAME_MAPPINGS as EXECUTION_CONTRACT_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.execution_contract_runtime_observation import (
        NODE_CLASS_MAPPINGS as EXECUTION_CONTRACT_RUNTIME_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.execution_contract_runtime_observation import (
        NODE_DISPLAY_NAME_MAPPINGS as EXECUTION_CONTRACT_RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.same_state_high_replay import (
        NODE_CLASS_MAPPINGS as SAME_STATE_REPLAY_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.same_state_high_replay import (
        NODE_DISPLAY_NAME_MAPPINGS as SAME_STATE_REPLAY_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate import first_high_operator_comparison as _FIRST_HIGH_OPERATOR_MODULE
    from .h3_flow_regenerate import first_high_sol_local_diagnostic as _FIRST_HIGH_SOL_LOCAL_MODULE
    from .h3_flow_regenerate.first_high_operator_comparison import (
        NODE_CLASS_MAPPINGS as FIRST_HIGH_OPERATOR_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.first_high_operator_comparison import (
        NODE_DISPLAY_NAME_MAPPINGS as FIRST_HIGH_OPERATOR_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.first_high_sol_local_diagnostic import (
        NODE_CLASS_MAPPINGS as FIRST_HIGH_SOL_LOCAL_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.first_high_sol_local_diagnostic import (
        NODE_DISPLAY_NAME_MAPPINGS as FIRST_HIGH_SOL_LOCAL_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .h3_flow_regenerate.target_sparse_node import (
        NODE_CLASS_MAPPINGS as TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.target_sparse_node import (
        NODE_DISPLAY_NAME_MAPPINGS as TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:  # Direct-file import used by packaging and test smoke checks.
    from h3_flow_regenerate.decode_context import (
        NODE_CLASS_MAPPINGS as DECODE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.decode_context import (
        NODE_DISPLAY_NAME_MAPPINGS as DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.handoff_checkpoint_diagnostic import (
        NODE_CLASS_MAPPINGS as HANDOFF_DIAGNOSTIC_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.handoff_checkpoint_diagnostic import (
        NODE_DISPLAY_NAME_MAPPINGS as HANDOFF_DIAGNOSTIC_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.guidance_anchor_transport_validation import (
        NODE_CLASS_MAPPINGS as GUIDANCE_ANCHOR_VALIDATION_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.guidance_anchor_transport_validation import (
        NODE_DISPLAY_NAME_MAPPINGS as GUIDANCE_ANCHOR_VALIDATION_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.execution_contract_diagnostics import (
        NODE_CLASS_MAPPINGS as EXECUTION_CONTRACT_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.execution_contract_diagnostics import (
        NODE_DISPLAY_NAME_MAPPINGS as EXECUTION_CONTRACT_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.execution_contract_runtime_observation import (
        NODE_CLASS_MAPPINGS as EXECUTION_CONTRACT_RUNTIME_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.execution_contract_runtime_observation import (
        NODE_DISPLAY_NAME_MAPPINGS as EXECUTION_CONTRACT_RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.same_state_high_replay import (
        NODE_CLASS_MAPPINGS as SAME_STATE_REPLAY_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.same_state_high_replay import (
        NODE_DISPLAY_NAME_MAPPINGS as SAME_STATE_REPLAY_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate import first_high_operator_comparison as _FIRST_HIGH_OPERATOR_MODULE
    from h3_flow_regenerate import first_high_sol_local_diagnostic as _FIRST_HIGH_SOL_LOCAL_MODULE
    from h3_flow_regenerate.first_high_operator_comparison import (
        NODE_CLASS_MAPPINGS as FIRST_HIGH_OPERATOR_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.first_high_operator_comparison import (
        NODE_DISPLAY_NAME_MAPPINGS as FIRST_HIGH_OPERATOR_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.first_high_sol_local_diagnostic import (
        NODE_CLASS_MAPPINGS as FIRST_HIGH_SOL_LOCAL_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.first_high_sol_local_diagnostic import (
        NODE_DISPLAY_NAME_MAPPINGS as FIRST_HIGH_SOL_LOCAL_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from h3_flow_regenerate.target_sparse_node import (
        NODE_CLASS_MAPPINGS as TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.target_sparse_node import (
        NODE_DISPLAY_NAME_MAPPINGS as TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    )


def _resolve_w_source_entry(entry):
    """Resolve reviewed source bytes without assuming custom-node top-level imports.

    ComfyUI loads directory custom nodes under a generated module name derived
    from the full filesystem path. Relative subpackages are therefore present in
    ``sys.modules`` but are not necessarily importable as bare ``h3_flow_regenerate``,
    ``sol_h3`` or ``vdn_h3`` packages. Prefer already-loaded exact/suffix matches,
    deduplicate aliases by resolved file, and only import normally when no loaded
    candidate exists. Ambiguous source identities fail closed.
    """
    module_name = entry.get("module")
    relative = entry.get("relative_path", ".")
    if not isinstance(module_name, str) or not module_name or not isinstance(relative, str):
        raise RuntimeError("first-high W source-delta entry has invalid path metadata")

    suffix = f".{module_name}"
    modules = [
        module
        for loaded_name, module in tuple(sys.modules.items())
        if module is not None and (loaded_name == module_name or loaded_name.endswith(suffix))
    ]
    if not modules:
        try:
            modules = [importlib.import_module(module_name)]
        except Exception as exc:
            raise RuntimeError(f"first-high W required source module is not loaded/importable: {module_name}") from exc

    resolved_files = {}
    for module in modules:
        raw_file = getattr(module, "__file__", None)
        if not isinstance(raw_file, str) or not raw_file:
            continue
        try:
            path = Path(raw_file).resolve(strict=True)
        except OSError:
            continue
        resolved_files[str(path)] = path

    if not resolved_files:
        raise RuntimeError(f"first-high W source module has no resolvable file: {module_name}")
    if len(resolved_files) != 1:
        raise RuntimeError(f"first-high W source module resolves ambiguously: {module_name}: {sorted(resolved_files)}")

    base = next(iter(resolved_files.values()))
    candidate = base if relative == "." else (base.parent / relative).resolve(strict=True)
    if not candidate.is_file():
        raise RuntimeError(f"first-high W source-delta path is not a file: {candidate}")
    return candidate


def _w_wrapper_specs():
    import comfy.patcher_extension

    return (
        (
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            _FIRST_HIGH_OPERATOR_MODULE._OUTER_KEY,
            _FIRST_HIGH_OPERATOR_MODULE._outer_wrapper,
        ),
        (
            comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
            _FIRST_HIGH_OPERATOR_MODULE._SAMPLER_KEY,
            _FIRST_HIGH_OPERATOR_MODULE._sampler_entry_wrapper,
        ),
    )


def _normalize_w_wrapper_order(current, guider):
    """Remove only the exact, proven W wrappers from R provenance comparison.

    Experiment R could not contain W's own OUTER_SAMPLE/SAMPLER_SAMPLE wrappers.
    The runtime manifest collected inside W therefore has two expected additions.
    Before comparing to R, prove those additions are the exact live W callables at
    the exact wrapper types/keys and that both active and patcher manifests record
    the same callable identities. Only those two entries are then removed. Any
    unrelated wrapper change remains visible to the ordinary R provenance diff.
    """
    patcher = getattr(guider, "model_patcher", None)
    runtime_wrappers = getattr(patcher, "wrappers", None)
    # Unit-level provenance helpers can be exercised without a real ModelPatcher.
    # Real W runtime always has ModelPatcher.wrappers; malformed runtime state is
    # rejected below rather than silently normalized.
    if runtime_wrappers is None:
        return current
    if not isinstance(runtime_wrappers, dict):
        raise RuntimeError("first-high W cannot inspect live ModelPatcher wrapper ownership")

    specs = _w_wrapper_specs()
    for wrapper_type, key, expected_callable in specs:
        locations = [
            existing_type
            for existing_type, keyed in runtime_wrappers.items()
            if isinstance(keyed, dict) and key in keyed
        ]
        if len(locations) != 1 or locations[0] != wrapper_type:
            raise RuntimeError(f"first-high W live wrapper key ownership changed for {key}: {locations!r}")
        keyed = runtime_wrappers.get(wrapper_type)
        values = keyed.get(key) if isinstance(keyed, dict) else None
        if not isinstance(values, (list, tuple)) or len(values) != 1 or values[0] is not expected_callable:
            raise RuntimeError(f"first-high W live wrapper callable identity changed for {key}")

    result = dict(current)
    for field_name in ("active_wrapper_order", "patcher_wrapper_order"):
        manifest = result.get(field_name)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"first-high W provenance lacks {field_name}")
        rebuilt_manifest = dict(manifest)
        for wrapper_type, key, expected_callable in specs:
            manifest_key = str(wrapper_type)
            entries = rebuilt_manifest.get(manifest_key)
            if not isinstance(entries, list):
                raise RuntimeError(f"first-high W provenance lacks wrapper list {field_name}.{manifest_key}")
            matches = [
                (index, entry)
                for index, entry in enumerate(entries)
                if isinstance(entry, dict) and entry.get("key") == key
            ]
            if len(matches) != 1:
                raise RuntimeError(f"first-high W provenance must contain exactly one {field_name} entry for {key}")
            index, entry = matches[0]
            expected_identity = _FIRST_HIGH_OPERATOR_MODULE._callable_equivalence_identity(expected_callable)
            observed_identity_json = _FIRST_HIGH_OPERATOR_MODULE._canonical_json(entry.get("callable"))
            expected_identity_json = _FIRST_HIGH_OPERATOR_MODULE._canonical_json(expected_identity)
            if observed_identity_json != expected_identity_json:
                raise RuntimeError(f"first-high W provenance wrapper callable identity changed for {field_name}:{key}")
            rebuilt_manifest[manifest_key] = [item for item_index, item in enumerate(entries) if item_index != index]
        result[field_name] = rebuilt_manifest
    return result


_FIRST_HIGH_OPERATOR_MODULE._resolve_source_entry = _resolve_w_source_entry
_ORIGINAL_W_VDN_PROVENANCE_NORMALIZER = _FIRST_HIGH_OPERATOR_MODULE._normalize_vdn_w_object_patches


def _normalize_w_provenance_delta(current, capture, guider, source_gate):
    normalized = _ORIGINAL_W_VDN_PROVENANCE_NORMALIZER(current, capture, guider, source_gate)
    return _normalize_w_wrapper_order(normalized, guider)


_FIRST_HIGH_OPERATOR_MODULE._normalize_vdn_w_object_patches = _normalize_w_provenance_delta


def _drop_current_only_closure_expansion(capture_value, current_value):
    """Prune only closure metadata introduced by an extra diagnostic wrapper layer.

    ``callable_identity`` intentionally expands closures only to a bounded depth.
    E wraps the already-reviewed W VDN wrapper, so Core's derived DIT replacement
    chain can expose one additional nested ``closure`` dictionary even though the
    live E -> W -> production object-patch chain has already been proven exactly
    against R. No non-closure key, capture-owned closure, list shape, or value is
    relaxed here.
    """
    if isinstance(capture_value, dict) and isinstance(current_value, dict):
        rebuilt = {}
        dropped = 0
        for raw_key, item in current_value.items():
            key = str(raw_key)
            if key not in capture_value:
                if key == "closure":
                    dropped += 1
                    continue
                rebuilt[key] = item
                continue
            child, child_dropped = _drop_current_only_closure_expansion(capture_value[key], item)
            rebuilt[key] = child
            dropped += child_dropped
        return rebuilt, dropped
    if isinstance(capture_value, list) and isinstance(current_value, list) and len(capture_value) == len(current_value):
        rebuilt = []
        dropped = 0
        for capture_item, current_item in zip(capture_value, current_value, strict=True):
            child, child_dropped = _drop_current_only_closure_expansion(capture_item, current_item)
            rebuilt.append(child)
            dropped += child_dropped
        return rebuilt, dropped
    return current_value, 0


def _normalize_e_replacement_chain(current, capture):
    """Normalize only the E-owned closure-depth expansion in DIT replacements.

    The live VDN object-patch normalizer runs first and proves all fifty chains are
    exact E -> reviewed W -> captured production forwards. After that proof, the
    derived ``replacement_chain_dit`` may differ solely because the provenance
    collector sees one deeper closure layer. Each of the fifty MiniMax-H3 double
    block entries must otherwise equal R exactly after removing only current-only
    keys literally named ``closure``. Any other difference remains fatal.
    """
    current_chain = current.get("replacement_chain_dit")
    capture_chain = capture.get("replacement_chain_dit")
    if not isinstance(current_chain, dict) or not isinstance(capture_chain, dict):
        raise RuntimeError("first-high Sol-local E provenance lacks DIT replacement-chain manifests")
    if set(current_chain) != set(capture_chain):
        raise RuntimeError("first-high Sol-local E DIT replacement-chain key set differs from R")

    expected = {str(("double_block", index)) for index in range(_FIRST_HIGH_SOL_LOCAL_MODULE._EXPECTED_BLOCKS)}
    if not expected.issubset(current_chain):
        missing = sorted(expected - set(current_chain))
        raise RuntimeError(f"first-high Sol-local E DIT replacement-chain is missing H3 blocks: {missing[:4]}")

    rebuilt = dict(current_chain)
    for key in sorted(expected):
        pruned, _dropped = _drop_current_only_closure_expansion(capture_chain[key], current_chain[key])
        if _FIRST_HIGH_SOL_LOCAL_MODULE._canonical_json(pruned) != _FIRST_HIGH_SOL_LOCAL_MODULE._canonical_json(
            capture_chain[key]
        ):
            detail = _FIRST_HIGH_SOL_LOCAL_MODULE._replay._provenance_diff_paths(capture_chain[key], pruned, limit=8)
            raise RuntimeError(
                f"first-high Sol-local E DIT replacement-chain differs from R for {key}"
                + ("; " + ", ".join(detail) if detail else "")
            )
        rebuilt[key] = capture_chain[key]

    result = dict(current)
    result["replacement_chain_dit"] = rebuilt
    return result


_E_LAZY_SOL_SPARSE_BLOB = "9f462591e3492d0b7af476c16024c79d1237505e"


def _normalize_e_lazy_sol_sparse_source(current, capture, source_gate):
    """Remove only E's proven pre-H3 lazy import of unchanged ``sol_h3.sparse``.

    R collected companion imports before the first high H3 call, so the production
    sparse bridge was not necessarily imported yet. Importing E installs its local
    attention wrapper and therefore imports that unchanged bridge before the E
    provenance gate runs. Treating this exact self-import as a runtime delta is
    valid only when the live file sits beside the already source-gated Sol package,
    has the exact PR #11/#12 blob, and the provenance snapshot reports the same
    SHA-256. Any other replay-only Sol source remains visible to the ordinary
    cross-process provenance comparator.
    """
    current_groups = current.get("loaded_companion_sources")
    capture_groups = capture.get("loaded_companion_sources")
    if not isinstance(current_groups, dict) or not isinstance(capture_groups, dict):
        raise RuntimeError("first-high Sol-local E provenance lacks companion source inventories")

    current_group = current_groups.get("sol_h3")
    capture_group = capture_groups.get("sol_h3")
    current_map, current_problems = _FIRST_HIGH_SOL_LOCAL_MODULE._replay._companion_source_map(current_group)
    capture_map, capture_problems = _FIRST_HIGH_SOL_LOCAL_MODULE._replay._companion_source_map(capture_group)
    if current_problems or capture_problems:
        detail = current_problems + capture_problems
        raise RuntimeError("first-high Sol-local E Sol companion source inventory is invalid: " + "; ".join(detail))

    sol_init = Path(_FIRST_HIGH_SOL_LOCAL_MODULE._source_path(source_gate, "sol", "sol_h3")).resolve(strict=True)
    sparse_path = (sol_init.parent / "sparse.py").resolve(strict=True)
    if not sparse_path.is_file() or sparse_path.parent != sol_init.parent:
        raise RuntimeError("first-high Sol-local E sparse bridge is outside the reviewed Sol package")
    if _FIRST_HIGH_SOL_LOCAL_MODULE._git_blob_sha(sparse_path) != _E_LAZY_SOL_SPARSE_BLOB:
        raise RuntimeError("first-high Sol-local E sparse bridge differs from reviewed PR #11/#12 bytes")

    sparse_key = str(sparse_path)
    current_entry = current_map.get(sparse_key)
    if current_entry is None:
        return current
    disk_sha256 = _FIRST_HIGH_SOL_LOCAL_MODULE._sha256_file(sparse_path)
    if current_entry.get("sha256") != disk_sha256:
        raise RuntimeError("first-high Sol-local E sparse provenance SHA-256 differs from loaded file bytes")

    capture_entry = capture_map.get(sparse_key)
    if capture_entry is not None:
        if capture_entry.get("sha256") != disk_sha256:
            raise RuntimeError("first-high Sol-local E captured sparse source differs from reviewed file bytes")
        return current

    rebuilt_group = []
    removed = 0
    for entry in current_group:
        if not isinstance(entry, dict):
            rebuilt_group.append(entry)
            continue
        file_info = entry.get("file")
        raw_path = None
        if isinstance(file_info, dict):
            raw_path = file_info.get("resolved_path") or file_info.get("path")
        if raw_path == sparse_key:
            if file_info.get("sha256") != disk_sha256:
                raise RuntimeError("first-high Sol-local E sparse alias reports inconsistent source bytes")
            removed += 1
            continue
        rebuilt_group.append(entry)
    if removed < 1:
        raise RuntimeError("first-high Sol-local E could not isolate its lazy sparse source entry")

    rebuilt_groups = dict(current_groups)
    rebuilt_groups["sol_h3"] = rebuilt_group
    result = dict(current)
    result["loaded_companion_sources"] = rebuilt_groups
    return result


_ORIGINAL_E_VDN_PROVENANCE_NORMALIZER = _FIRST_HIGH_SOL_LOCAL_MODULE._normalize_vdn_object_patches


def _normalize_e_provenance_delta(current, capture, guider, source_gate):
    normalized = _ORIGINAL_E_VDN_PROVENANCE_NORMALIZER(current, capture, guider, source_gate)
    normalized = _normalize_e_replacement_chain(normalized, capture)
    return _normalize_e_lazy_sol_sparse_source(normalized, capture, source_gate)


_FIRST_HIGH_SOL_LOCAL_MODULE._normalize_vdn_object_patches = _normalize_e_provenance_delta

_ORIGINAL_E_MEMORY_PREFLIGHT = _FIRST_HIGH_SOL_LOCAL_MODULE._memory_preflight
_ORIGINAL_E_MEMORY_COMPLETION = _FIRST_HIGH_SOL_LOCAL_MODULE._memory_completion


def _e_runtime_cuda_device(device):
    """Resolve E's CUDA preflight target independently of replay tensor storage.

    R legitimately captured the first-high latent on CPU in the sampler lifetime,
    while the H3 transformer and Sol-Attn execute on CUDA. A CPU replay tensor is
    therefore not evidence that the runtime lacks CUDA. When exactly one CUDA
    device exists, that device is unambiguous; multi-GPU CPU replay remains
    fail-closed because E cannot infer which GPU owns the forthcoming H3 call.
    """
    torch = _FIRST_HIGH_SOL_LOCAL_MODULE.torch
    replay_device = torch.device(device)
    if not torch.cuda.is_available():
        raise RuntimeError("first-high Sol-local E requires CUDA SM120")
    if replay_device.type == "cuda":
        runtime_device = replay_device
        if runtime_device.index is None:
            runtime_device = torch.device("cuda", int(torch.cuda.current_device()))
    else:
        if int(torch.cuda.device_count()) != 1:
            raise RuntimeError(
                "first-high Sol-local E replay tensor is non-CUDA and the runtime CUDA device is ambiguous"
            )
        runtime_device = torch.device("cuda", 0)

    capability = tuple(int(value) for value in torch.cuda.get_device_capability(runtime_device))
    if capability != (12, 0):
        sm = capability[0] * 10 + capability[1]
        raise RuntimeError(
            f"first-high Sol-local E requires CUDA SM120; runtime device {runtime_device} reports SM{sm}"
        )
    return replay_device, runtime_device, capability


def _memory_preflight_on_runtime_cuda(device):
    replay_device, runtime_device, capability = _e_runtime_cuda_device(device)
    result = dict(_ORIGINAL_E_MEMORY_PREFLIGHT(runtime_device))
    result["replay_tensor_device"] = str(replay_device)
    result["cuda_preflight_device"] = str(runtime_device)
    result["cuda_compute_capability"] = [int(capability[0]), int(capability[1])]
    return result


_FIRST_HIGH_SOL_LOCAL_MODULE._memory_preflight = _memory_preflight_on_runtime_cuda


def _memory_completion_on_runtime_cuda(device, preflight, evidence):
    replay_device, runtime_device, capability = _e_runtime_cuda_device(device)
    expected_capability = [int(capability[0]), int(capability[1])]
    if preflight.get("cuda_preflight_device") != str(runtime_device):
        raise RuntimeError(
            "first-high Sol-local E completion CUDA device differs from the validated preflight device"
        )
    if preflight.get("cuda_compute_capability") != expected_capability:
        raise RuntimeError(
            "first-high Sol-local E completion CUDA capability differs from the validated preflight capability"
        )
    if preflight.get("replay_tensor_device") != str(replay_device):
        raise RuntimeError(
            "first-high Sol-local E completion replay device differs from the validated preflight replay device"
        )
    return _ORIGINAL_E_MEMORY_COMPLETION(runtime_device, preflight, evidence)


_FIRST_HIGH_SOL_LOCAL_MODULE._memory_completion = _memory_completion_on_runtime_cuda

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    **TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    **DECODE_NODE_CLASS_MAPPINGS,
    **HANDOFF_DIAGNOSTIC_NODE_CLASS_MAPPINGS,
    **GUIDANCE_ANCHOR_VALIDATION_NODE_CLASS_MAPPINGS,
    **EXECUTION_CONTRACT_NODE_CLASS_MAPPINGS,
    **EXECUTION_CONTRACT_RUNTIME_NODE_CLASS_MAPPINGS,
    **SAME_STATE_REPLAY_NODE_CLASS_MAPPINGS,
    **FIRST_HIGH_OPERATOR_NODE_CLASS_MAPPINGS,
    **FIRST_HIGH_SOL_LOCAL_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    **TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    **DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    **HANDOFF_DIAGNOSTIC_NODE_DISPLAY_NAME_MAPPINGS,
    **GUIDANCE_ANCHOR_VALIDATION_NODE_DISPLAY_NAME_MAPPINGS,
    **EXECUTION_CONTRACT_NODE_DISPLAY_NAME_MAPPINGS,
    **EXECUTION_CONTRACT_RUNTIME_NODE_DISPLAY_NAME_MAPPINGS,
    **SAME_STATE_REPLAY_NODE_DISPLAY_NAME_MAPPINGS,
    **FIRST_HIGH_OPERATOR_NODE_DISPLAY_NAME_MAPPINGS,
    **FIRST_HIGH_SOL_LOCAL_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
