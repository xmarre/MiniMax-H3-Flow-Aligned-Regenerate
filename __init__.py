# ruff: noqa: I001
# Import order is intentional: the PR35 checkpoint diagnostic must install first
# so the learned-anchor validation wrapper can sit outside it and make those
# diagnostics observe the transported target-grid guidance representation. The
# execution-contract recorder is imported after both; its runtime-observation
# extension is imported immediately after the base recorder. The same-state
# replay and first-high operator diagnostics compose only additional wrappers
# around that recorder.
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


_FIRST_HIGH_OPERATOR_MODULE._resolve_source_entry = _resolve_w_source_entry

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
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
