# ruff: noqa: I001
# Import order is intentional: the PR35 checkpoint diagnostic must install first
# so the learned-anchor validation wrapper can sit outside it and make those
# diagnostics observe the transported target-grid guidance representation. The
# execution-contract recorder is imported after both; its runtime-observation
# extension is imported immediately after the base recorder. The same-state
# replay and production-candidate validation layers compose only additional
# wrappers around that recorder.
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
    from .h3_flow_regenerate import production_mapped_neighbor_candidate as _PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_MODULE
    from .h3_flow_regenerate.production_mapped_neighbor_candidate import (
        NODE_CLASS_MAPPINGS as PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.production_mapped_neighbor_candidate import (
        NODE_DISPLAY_NAME_MAPPINGS as PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_DISPLAY_NAME_MAPPINGS,
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
    from h3_flow_regenerate import production_mapped_neighbor_candidate as _PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_MODULE
    from h3_flow_regenerate.production_mapped_neighbor_candidate import (
        NODE_CLASS_MAPPINGS as PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.production_mapped_neighbor_candidate import (
        NODE_DISPLAY_NAME_MAPPINGS as PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from h3_flow_regenerate.target_sparse_node import (
        NODE_CLASS_MAPPINGS as TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.target_sparse_node import (
        NODE_DISPLAY_NAME_MAPPINGS as TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    )


_CANDIDATE_OWNER_PACKAGES = {
    "flow": "h3_flow_regenerate",
    "sol": "sol_h3",
    "vdn": "vdn_h3",
}
_CANDIDATE_CONTRACT_MODULES = (
    "sol_h3.provenance",
    "vdn_h3.softmax_provider",
)
_CANDIDATE_FLOW_BASE = "4ae2e35f77ed961151ab5695bef9e1dbe277cc54"


def _candidate_loaded_module_matches(module_name):
    """Return one loaded exact/suffix module identity, failing on distinct files."""
    suffix = f".{module_name}"
    matches = []
    for loaded_name, module in tuple(sys.modules.items()):
        if module is None or not (loaded_name == module_name or loaded_name.endswith(suffix)):
            continue
        raw_file = getattr(module, "__file__", None)
        if not isinstance(raw_file, str) or not raw_file:
            continue
        try:
            path = Path(raw_file).resolve(strict=True)
        except OSError:
            continue
        matches.append((loaded_name, module, path))
    by_path = {}
    for loaded_name, module, path in matches:
        by_path.setdefault(str(path), (loaded_name, module, path))
    if len(by_path) > 1:
        raise RuntimeError(f"production candidate source module resolves ambiguously: {module_name}: {sorted(by_path)}")
    return None if not by_path else next(iter(by_path.values()))


def _candidate_owner_root(owner):
    package_name = _CANDIDATE_OWNER_PACKAGES.get(str(owner))
    if package_name is None:
        raise RuntimeError(f"production candidate source entry has unknown owner: {owner!r}")
    loaded = _candidate_loaded_module_matches(package_name)
    if loaded is None:
        raise RuntimeError(
            f"production candidate owner package is not loaded under the active Comfy namespace: {package_name}"
        )
    _loaded_name, _module, package_file = loaded
    if package_file.name != "__init__.py":
        raise RuntimeError(f"production candidate owner package has unexpected source path: {package_file}")
    return package_file.parent.parent


def _resolve_production_candidate_source_entry(entry):
    """Resolve reviewed bytes without importing optional production modules.

    Comfy can load custom nodes below generated package names, so bare ``sol_h3``
    imports are not a valid source locator. Prefer an already-loaded exact/suffix
    module. If a reviewed child is intentionally lazy (notably the SM120 CuTe
    kernel), derive its path from the already-loaded owner package instead of
    importing it. Distinct source identities fail closed.
    """
    module_name = entry.get("module")
    relative = entry.get("relative_path", ".")
    owner = entry.get("owner")
    if not isinstance(module_name, str) or not module_name or not isinstance(relative, str):
        raise RuntimeError("production candidate source entry has invalid module/path metadata")

    loaded = _candidate_loaded_module_matches(module_name)
    if loaded is not None:
        base = loaded[2]
    else:
        root = _candidate_owner_root(owner)
        candidate = root.joinpath(*module_name.split("."))
        package_init = candidate / "__init__.py"
        base = package_init if package_init.is_file() else candidate.with_suffix(".py")
        try:
            base = base.resolve(strict=True)
        except OSError as exc:
            raise RuntimeError(
                f"production candidate source module is not loaded and has no reviewed on-disk path: {module_name}"
            ) from exc
    candidate = base if relative == "." else (base.parent / relative).resolve(strict=True)
    if not candidate.is_file():
        raise RuntimeError(f"production candidate source path is not a file: {candidate}")
    return candidate


def _same_resolved_source(left, right):
    left_file = getattr(left, "__file__", None)
    right_file = getattr(right, "__file__", None)
    if not isinstance(left_file, str) or not isinstance(right_file, str):
        return False
    try:
        return Path(left_file).resolve(strict=True) == Path(right_file).resolve(strict=True)
    except OSError:
        return False


_ORIGINAL_PRODUCTION_CANDIDATE_SOURCE_VERIFY = _PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_MODULE._verify_source_manifest


def _verify_production_candidate_sources():
    """Expose loader aliases without importing reviewed child modules early.

    The original verifier hashes every allowlisted source before importing the
    lightweight Sol/VDN contract modules.  Preserve that ordering: alias only
    owner packages that Comfy has already loaded, plus contract children that are
    already loaded.  If a child is still lazy, the original verifier imports it
    only after all reviewed bytes have passed their Git-blob checks.  Optional
    SM120 CuTe modules are never imported for source discovery.
    """
    manifest, _manifest_digest = _PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_MODULE._load_source_manifest()
    if manifest.get("flow_base") != _CANDIDATE_FLOW_BASE:
        raise RuntimeError("production candidate source-delta manifest targets the wrong Flow R base")

    introduced = []
    absent_contracts = []
    try:
        for package_name in ("sol_h3", "vdn_h3"):
            package = _candidate_loaded_module_matches(package_name)
            if package is None:
                raise RuntimeError(f"production candidate contract package is not loaded: {package_name}")
            existing = sys.modules.get(package_name)
            if existing is None:
                sys.modules[package_name] = package[1]
                introduced.append(package_name)
            elif existing is not package[1] and not _same_resolved_source(existing, package[1]):
                raise RuntimeError(
                    f"production candidate bare package alias is owned by another source: {package_name}"
                )

        for module_name in _CANDIDATE_CONTRACT_MODULES:
            loaded = _candidate_loaded_module_matches(module_name)
            existing = sys.modules.get(module_name)
            if existing is None:
                absent_contracts.append(module_name)
                if loaded is not None:
                    sys.modules[module_name] = loaded[1]
                    introduced.append(module_name)
            elif loaded is not None and existing is not loaded[1] and not _same_resolved_source(existing, loaded[1]):
                raise RuntimeError(
                    f"production candidate bare contract alias is owned by another source: {module_name}"
                )

        return _ORIGINAL_PRODUCTION_CANDIDATE_SOURCE_VERIFY()
    finally:
        # A child that was absent before this gate may have been imported by the
        # original verifier after its source bytes were validated.  Do not leak
        # that temporary bare alias into the long-lived Comfy process.
        for module_name in reversed(absent_contracts):
            sys.modules.pop(module_name, None)
        for module_name in reversed(introduced):
            sys.modules.pop(module_name, None)


_PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_MODULE._resolve_source_entry = _resolve_production_candidate_source_entry
_PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_MODULE._verify_source_manifest = _verify_production_candidate_sources

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    **TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    **DECODE_NODE_CLASS_MAPPINGS,
    **HANDOFF_DIAGNOSTIC_NODE_CLASS_MAPPINGS,
    **GUIDANCE_ANCHOR_VALIDATION_NODE_CLASS_MAPPINGS,
    **EXECUTION_CONTRACT_NODE_CLASS_MAPPINGS,
    **EXECUTION_CONTRACT_RUNTIME_NODE_CLASS_MAPPINGS,
    **SAME_STATE_REPLAY_NODE_CLASS_MAPPINGS,
    **PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_CLASS_MAPPINGS,
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
    **PRODUCTION_MAPPED_NEIGHBOR_CANDIDATE_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
