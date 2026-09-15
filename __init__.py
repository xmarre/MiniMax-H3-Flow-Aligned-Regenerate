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
    unrelated wrapper change remains visible to the ordinary provenance diff.
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
