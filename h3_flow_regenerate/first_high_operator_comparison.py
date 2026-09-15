"""Bounded first-high operator comparison W.

W reuses an existing, hash-validated experiment-R bundle and executes exactly the
first high Euler model call in a fresh process.  It changes only the explicit VDN
local-attention operator selected by ``mode``.  The original high sigma suffix is
kept intact so MiniMax-H3 PDD/final-head selection sees the same schedule.

This module is diagnostic-only.  It never changes production progressive policy,
never regenerates the R bundle, and never reports R ``attribution_valid`` for an
operator-modified run.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import comfy_compat as _comfy_compat
from . import execution_contract_diagnostics as _diag
from . import runtime as _runtime
from . import same_state_high_replay as _replay

SCHEMA_VERSION = 1
STATE_KEY = "h3_flow_first_high_operator_comparison_v1"
REQUEST_KEY = "h3_first_high_operator_diagnostic_v1"
RECEIPTS_KEY = "h3_first_high_operator_receipts_v1"
_OUTER_KEY = "h3_flow_regenerate.first_high_operator.execute.v1"
_SAMPLER_KEY = "h3_flow_regenerate.first_high_operator.sampler_entry.v1"
_SOURCE_MANIFEST = "first_high_w_source_delta.json"
_ALLOWED_MODES = ("native_window", "native_full_support")
_REQUIRED_SUFFIX = [0.8780487775802612, 0.800000011920929, 0.6315789222717285, 0.0]
_MAX_REPORTS = 4


class _FirstCallComplete(BaseException):
    def __init__(self, token: object, x0: torch.Tensor):
        super().__init__("first-high W diagnostic completed its single model call")
        self.token = token
        self.x0 = x0


@dataclass(slots=True)
class _State:
    replay: _replay._ReplayState
    mode: str
    source_gate: dict[str, Any]
    complete: deque[dict[str, Any]] = field(default_factory=lambda: deque(maxlen=_MAX_REPORTS))


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha(path: Path) -> str:
    data = path.read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def _load_source_manifest() -> tuple[dict[str, Any], str]:
    path = Path(__file__).with_name(_SOURCE_MANIFEST)
    try:
        raw = path.read_text(encoding="utf-8")
        manifest = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("first-high W source-delta manifest is missing or invalid") from exc
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise RuntimeError("first-high W source-delta manifest schema is unsupported")
    entries = manifest.get("entries")
    if not isinstance(entries, list) or not entries:
        raise RuntimeError("first-high W source-delta manifest has no reviewed entries")
    digest = _sha_json(manifest)
    return manifest, digest


def _resolve_source_entry(entry: dict[str, Any]) -> Path:
    module_name = entry.get("module")
    relative = entry.get("relative_path", ".")
    if not isinstance(module_name, str) or not module_name or not isinstance(relative, str):
        raise RuntimeError("first-high W source-delta entry has invalid path metadata")
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"first-high W required source module is not importable: {module_name}") from exc
    raw_file = getattr(module, "__file__", None)
    if not isinstance(raw_file, str) or not raw_file:
        raise RuntimeError(f"first-high W source module has no file: {module_name}")
    base = Path(raw_file).resolve(strict=True)
    candidate = base if relative == "." else (base.parent / relative).resolve(strict=True)
    if not candidate.is_file():
        raise RuntimeError(f"first-high W source-delta path is not a file: {candidate}")
    return candidate


def _verify_source_manifest() -> dict[str, Any]:
    manifest, digest = _load_source_manifest()
    observed = []
    for raw in manifest["entries"]:
        if not isinstance(raw, dict):
            raise RuntimeError("first-high W source-delta entry is not a dictionary")
        path = _resolve_source_entry(raw)
        expected_blob = raw.get("candidate_git_blob_sha")
        if not isinstance(expected_blob, str) or len(expected_blob) != 40:
            raise RuntimeError("first-high W source-delta entry lacks candidate Git blob identity")
        actual_blob = _git_blob_sha(path)
        if actual_blob != expected_blob:
            raise RuntimeError(
                f"first-high W source bytes differ from reviewed candidate: {path}: {actual_blob} != {expected_blob}"
            )
        observed.append(
            {
                "owner": raw.get("owner"),
                "path": str(path),
                "git_blob_sha": actual_blob,
                "sha256": _sha256_file(path),
                "reason": raw.get("reason"),
            }
        )
    return {"manifest_digest": digest, "entries": observed}


def _target_shapes_digest(manifest: dict[str, Any]) -> str:
    return _sha_json([[int(dim) for dim in shape] for shape in manifest["target_shapes"]])


def _request_tuple(state: _State) -> tuple[tuple[str, Any], ...]:
    manifest = state.replay.manifest
    high = [float(v) for v in manifest.get("high_sigmas", [])]
    if high != _REQUIRED_SUFFIX:
        raise RuntimeError(f"first-high W requires the original high suffix; got {high}")
    return (
        ("api", 1),
        ("capture_id", str(manifest["capture_id"])),
        ("mode", state.mode),
        ("stage", "high"),
        ("logical_call_limit", 1),
        ("sigma", float(high[0])),
        ("target_shapes_digest", _target_shapes_digest(manifest)),
        ("source_contract_digest", str(state.source_gate["manifest_digest"])),
    )


def _sampling_runtime_identity(guider: Any) -> dict[str, Any]:
    patcher = getattr(guider, "model_patcher", None)
    base = getattr(patcher, "model", None)
    sampling = getattr(base, "model_sampling", None)
    latent = getattr(base, "latent_format", None)
    diffusion = getattr(base, "diffusion_model", None)
    final_layer = getattr(diffusion, "final_layer", None)
    video_out = getattr(final_layer, "video_out", None)
    head_count = None
    weight = getattr(video_out, "weight", None)
    out_features = getattr(video_out, "out_features", None)
    if torch.is_tensor(weight) and type(out_features) is int and out_features > 0:
        head_count = int(weight.shape[0]) // int(out_features)
    return {
        "model_sampling_class": (
            None if sampling is None else f"{type(sampling).__module__}.{type(sampling).__qualname__}"
        ),
        "latent_format_class": (
            None if latent is None else f"{type(latent).__module__}.{type(latent).__qualname__}"
        ),
        "final_head_count": head_count,
    }


def _allowed_provenance_difference(path: str, source_gate: dict[str, Any]) -> bool:
    lowered = path.lower()
    if "first_high_operator" in lowered:
        return True
    for entry in source_gate.get("entries", []):
        source_path = str(entry.get("path", ""))
        if source_path and source_path in path:
            return True
        name = Path(source_path).name if source_path else ""
        if name and name in path and any(token in lowered for token in ("loaded_companion_sources", "source")):
            return True
    return False


def _provenance_gate(state: _State, record: _diag._Record) -> dict[str, Any]:
    capture = _replay._bundle_provenance_equivalence_identity(state.replay.manifest)
    current = _replay._provenance_equivalence_identity(record.state.manifest)
    differences = _replay._cross_process_provenance_diff_paths(capture, current, limit=64)
    unexpected = [path for path in differences if not _allowed_provenance_difference(path, state.source_gate)]
    return {
        "differences": differences,
        "unexpected_differences": unexpected,
        "exact_except_reviewed_w_delta": not unexpected,
    }


def _snapshot_hash(record: _diag._Record, name: str) -> str | None:
    snapshot = record.snapshots.get(name)
    return None if snapshot is None else _replay._tensor_sha256(snapshot.tensor)


def _snapshot_report(record: _diag._Record, manifest: dict[str, Any]) -> dict[str, Any]:
    expected = manifest.get("expected_snapshot_hashes") or {}
    names = (
        "first_high_sampler_input_video",
        "first_high_sampler_input_audio",
        "first_high_h3_input_video",
        "first_high_h3_input_audio",
        "first_high_h3_velocity_video",
        "first_high_h3_velocity_audio",
        "first_high_model_raw_video",
        "first_high_model_raw_audio",
        "first_high_pre_guidance_video",
    )
    result = {}
    for name in names:
        current = _snapshot_hash(record, name)
        result[name] = {
            "capture_sha256": expected.get(name),
            "w_sha256": current,
            "entry_equal": (
                current is not None
                and expected.get(name) is not None
                and current == expected.get(name)
                if name
                in {
                    "first_high_sampler_input_video",
                    "first_high_sampler_input_audio",
                    "first_high_h3_input_video",
                    "first_high_h3_input_audio",
                }
                else None
            ),
        }
    return result


def _validate_receipts(receipts: list[Any], mode: str) -> dict[str, Any]:
    subcalls = [item for item in receipts if isinstance(item, dict)]
    kinds = Counter(str(item.get("kind")) for item in subcalls)
    local_routes = Counter(
        str(item.get("provider_route")) for item in subcalls if item.get("kind") == "local"
    )
    packed_rows = {int(item.get("packed_rows", -1)) for item in subcalls}
    video_spans = {
        (int(item.get("video_start", -1)), int(item.get("video_end", -1))) for item in subcalls
    }
    local = [item for item in subcalls if item.get("kind") == "local"]
    expected_local_route = (
        "vdn_local_native_window_w" if mode == "native_window" else "vdn_local_native_full_w"
    )
    support_ok = all(
        item.get("support_mode") == ("restricted_window" if mode == "native_window" else "canonical_full")
        for item in local
    )
    complement_ok = all(
        bool(item.get("complement_executed")) == (mode == "native_window") for item in local
    )
    full_kv_ok = True
    if mode == "native_full_support":
        full_kv_ok = all(
            item.get("canonical_full_kv") is True and int(item.get("kv_rows", -1)) == 56349
            for item in local
        )
    fingerprints_ok = bool(subcalls) and all(
        isinstance(item.get("gate_fingerprint"), str) and isinstance(item.get("adapter_fingerprint"), str)
        for item in subcalls
    )
    expected = kinds == Counter({"local": 550, "global": 50, "anchor": 100})
    return {
        "count": len(subcalls),
        "kind_counts": dict(kinds),
        "local_routes": dict(local_routes),
        "packed_rows": sorted(packed_rows),
        "video_spans": sorted(video_spans),
        "support_ok": support_ok,
        "complement_ok": complement_ok,
        "canonical_full_kv_ok": full_kv_ok,
        "fingerprints_present": fingerprints_ok,
        "expected_700_subcalls": expected and len(subcalls) == 700,
        "expected_local_route": local_routes == Counter({expected_local_route: 550}),
        "first_block_pre_attention_qkv_digest": next(
            (
                item.get("pre_attention_qkv_digest")
                for item in subcalls
                if item.get("block") == 0 and item.get("kind") == "global"
            ),
            None,
        ),
    }


def _sampler_entry_wrapper(
    executor,
    model_wrap,
    sigmas,
    extra_args,
    callback,
    noise,
    latent_image=None,
    denoise_mask=None,
    disable_pbar=False,
):
    options = extra_args.get("model_options") if isinstance(extra_args, dict) else None
    state = (options or {}).get(STATE_KEY)
    if isinstance(state, _State) and _diag._stage_name(options) == "high":
        record = _diag._ACTIVE.get()
        if record is None:
            raise RuntimeError("first-high W sampler-entry validation requires active execution diagnostics")
        _replay._validate_replay_sampler_entry(state.replay, record)
    return executor(model_wrap, sigmas, extra_args, callback, noise, latent_image, denoise_mask, disable_pbar)


def _outer_wrapper(
    executor,
    noise,
    latent_image,
    sampler,
    sigmas,
    denoise_mask=None,
    callback=None,
    disable_pbar=False,
    seed=None,
    latent_shapes=None,
):
    guider = executor.class_obj
    options = getattr(guider, "model_options", None)
    state = (options or {}).get(STATE_KEY)
    if not isinstance(state, _State):
        return executor(
            noise,
            latent_image,
            sampler,
            sigmas,
            denoise_mask,
            callback,
            disable_pbar,
            seed,
            latent_shapes=latent_shapes,
        )
    record = _diag._ACTIVE.get()
    if record is None or not record.state.strict_provenance or not record.state.manifest.get("gate_complete"):
        raise RuntimeError("first-high W requires strict, complete execution-contract diagnostics")
    replay = state.replay
    _replay._assert_checkpoint_identity_unchanged(replay.exact_checkpoint_identity)
    if not isinstance(options, dict):
        raise RuntimeError("first-high W requires mutable guider model options")
    config = options.get(_runtime.PROGRESSIVE_KEY)
    if not _diag._eligible(config):
        raise RuntimeError("first-high W requires the original progressive Target Input config")
    current_sampler = _runtime.sampler_name(sampler)
    if str(replay.manifest.get("sampler")) != current_sampler or current_sampler != "sample_euler":
        raise RuntimeError("first-high W is bounded to the captured Euler sampler")
    if int(seed or 0) != int(replay.manifest["seed"]):
        raise RuntimeError("first-high W seed differs from R capture")
    if denoise_mask is not None or "high_denoise_mask" in replay.tensors:
        raise RuntimeError("first-high W does not support protected/masked replay")
    if not isinstance(latent_shapes, list):
        raise RuntimeError("first-high W requires mutable target latent-shape metadata")
    target_shapes = [tuple(int(dim) for dim in shape) for shape in replay.manifest["target_shapes"]]
    if [tuple(int(dim) for dim in shape) for shape in latent_shapes] != target_shapes:
        raise RuntimeError("first-high W target geometry differs from R capture")
    if _runtime._schedule_signature(sigmas) != str(replay.manifest["original_schedule_digest"]):
        raise RuntimeError("first-high W caller schedule differs from R capture")
    if record.pristine_cond_digest != str(replay.manifest["pristine_conditioning_digest"]):
        raise RuntimeError("first-high W pristine target conditioning differs from R capture")

    provenance_gate = _provenance_gate(state, record)
    if not provenance_gate["exact_except_reviewed_w_delta"]:
        raise RuntimeError(
            "first-high W runtime provenance differs beyond the reviewed source delta: "
            + ", ".join(provenance_gate["unexpected_differences"][:12])
        )

    binding = _runtime._resolve_binding(guider)
    if binding is None:
        raise RuntimeError("first-high W requires the active Flow binding")
    if binding.active_capture is not None or binding.active_guidance_run is not None:
        raise RuntimeError("first-high W entered with stale Flow capture/guidance state")
    current_guidance = _replay._guidance_config_dict(binding)
    if not _replay._dict_equal(current_guidance, replay.manifest.get("guidance_config")):
        raise RuntimeError("first-high W Flow guidance config differs from R capture")
    current_spectrum = _replay._spectrum_config_dict(guider)
    if not _replay._dict_equal(current_spectrum, replay.manifest.get("spectrum_config")):
        raise RuntimeError("first-high W Spectrum config differs from R capture")

    trajectory, run_id = _replay._rebuild_trajectory(replay.manifest, replay.tensors)
    if current_guidance is not None and current_guidance.get("mode") != "off" and trajectory is None:
        raise RuntimeError("first-high W guidance is active but R bundle has no captured trajectory")

    handoff_index = int(replay.manifest["handoff_index"])
    high_sigmas = sigmas[handoff_index:]
    captured_high = [float(value) for value in replay.manifest["high_sigmas"]]
    live_high = [float(value) for value in high_sigmas.detach().to(device="cpu", dtype=torch.float64).tolist()]
    if captured_high != _REQUIRED_SUFFIX or live_high != captured_high:
        raise RuntimeError("first-high W must keep the complete original captured high suffix")

    replay_latent = _replay._tensor_for_replay(replay, "high_latent_image")
    replay_noise = _replay._tensor_for_replay(replay, "high_noise_argument")
    request = _request_tuple(state)
    receipt_sink: list[Any] = []
    transformer = options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("first-high W requires mutable transformer options")
    for key in (REQUEST_KEY, RECEIPTS_KEY):
        if key in transformer:
            raise RuntimeError(f"first-high W private transformer option already exists: {key}")
    sol_config = transformer.get("sol_h3_runtime_v1")
    if not isinstance(sol_config, dict) or sol_config.get("backend") != "sol":
        raise RuntimeError("first-high W requires the active Sol-H3 backend companion")

    metric_start = binding.metrics.counters
    event_start = len(binding.metrics.events)
    previous_progressive = options.pop(_runtime.PROGRESSIVE_KEY)
    previous_trajectory = binding.trajectory
    previous_guidance_run_id = binding.guidance_run_id
    previous_capture_enabled = binding.capture_enabled
    binding.trajectory = trajectory
    binding.guidance_run_id = run_id
    binding.capture_enabled = False
    transformer[REQUEST_KEY] = request
    transformer[RECEIPTS_KEY] = receipt_sink
    sentinel_token = object()
    completed = False
    diagnostic_x0 = None
    replay_error: BaseException | None = None

    def stop_after_first(step, x0, x, _total):
        _ = x
        if int(step) != 0:
            raise RuntimeError("first-high W callback reached a second Euler step")
        raise _FirstCallComplete(sentinel_token, x0)

    try:
        if record.pristine_conds is None:
            raise RuntimeError("first-high W record is missing pristine target conditions")
        _runtime._reset_guider_conds(guider, template=record.pristine_conds)
        try:
            with _runtime._flow_stage_contract(guider, "high"), _runtime._high_stage_contract(guider):
                executor(
                    replay_noise,
                    replay_latent,
                    sampler,
                    high_sigmas,
                    None,
                    stop_after_first,
                    disable_pbar,
                    seed,
                    latent_shapes=latent_shapes,
                )
        except _FirstCallComplete as exc:
            if exc.token is not sentinel_token:
                raise
            completed = True
            diagnostic_x0 = exc.x0
        if not completed or diagnostic_x0 is None:
            raise RuntimeError("first-high W Euler callback sentinel did not terminate after the first model call")
        return diagnostic_x0
    except BaseException as exc:
        replay_error = exc
        raise
    finally:
        transformer.pop(REQUEST_KEY, None)
        transformer.pop(RECEIPTS_KEY, None)
        binding.trajectory = previous_trajectory
        binding.guidance_run_id = previous_guidance_run_id
        binding.capture_enabled = previous_capture_enabled
        options[_runtime.PROGRESSIVE_KEY] = previous_progressive

        if completed and replay_error is None:
            metric_delta = _diag._counter_delta(metric_start, binding.metrics.counters)
            events = binding.metrics.events[event_start:]
            upscaler_calls = sum(1 for event in events if event.kind == "handoff_learned_upscale_wall")
            topology = {
                "logical": int(metric_delta.get("sampler_logical_calls", 0)),
                "actual": int(metric_delta.get("transformer_actual_nfe", 0)),
                "forecast": int(metric_delta.get("spectrum_forecast_calls", 0)),
                "high_logical": int(metric_delta.get("sampler_logical_calls_high", 0)),
                "high_actual": int(metric_delta.get("transformer_actual_nfe_high", 0)),
                "high_forecast": int(metric_delta.get("spectrum_forecast_calls_high", 0)),
                "upscaler_calls": int(upscaler_calls),
            }
            expected_topology = {
                "logical": 1,
                "actual": 1,
                "forecast": 0,
                "high_logical": 1,
                "high_actual": 1,
                "high_forecast": 0,
                "upscaler_calls": 0,
            }
            snapshots = _snapshot_report(record, replay.manifest)
            entry_exact = all(
                snapshots[name]["entry_equal"] is True
                for name in (
                    "first_high_sampler_input_video",
                    "first_high_sampler_input_audio",
                    "first_high_h3_input_video",
                    "first_high_h3_input_audio",
                )
            )
            receipt_report = _validate_receipts(receipt_sink, state.mode)
            companion = _replay._high_first_companion_observation(record) or {}
            sol_after = (
                ((companion.get("after") or {}).get("sol") or {}) if isinstance(companion, dict) else {}
            )
            sol_zero = bool(
                int(sol_after.get("vdn_local_sol_calls", 0) or 0) == 0
                and int(sol_after.get("sparse_calls", 0) or 0) == 0
                and int(sol_after.get("external_mixed_sol_calls", 0) or 0) == 0
            )
            invariants = bool(
                topology == expected_topology
                and entry_exact
                and receipt_report["expected_700_subcalls"]
                and receipt_report["expected_local_route"]
                and receipt_report["support_ok"]
                and receipt_report["complement_ok"]
                and receipt_report["canonical_full_kv_ok"]
                and receipt_report["fingerprints_present"]
                and sol_zero
                and provenance_gate["exact_except_reviewed_w_delta"]
            )
            state.complete.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "minimax_h3_first_high_operator_comparison_report",
                    "capture_id": replay.manifest["capture_id"],
                    "mode": state.mode,
                    "completed_first_call_only": True,
                    "final_trajectory_available": False,
                    "operator_modified_from_r": True,
                    "r_attribution_valid_not_applicable": True,
                    "w_invariants_valid": invariants,
                    "topology": topology,
                    "topology_exact": topology == expected_topology,
                    "source_gate": state.source_gate,
                    "provenance_gate": provenance_gate,
                    "sampling_runtime": _sampling_runtime_identity(guider),
                    "request": dict(request),
                    "snapshot_hashes": snapshots,
                    "entry_state_exact": entry_exact,
                    "receipts": receipt_report,
                    "sol_local_sparse_zero": sol_zero,
                    "sol_after": sol_after,
                    "diagnostic_x0_sha256": _replay._tensor_sha256(diagnostic_x0),
                    "diagnostic_x0_shape": [int(dim) for dim in diagnostic_x0.shape],
                    "decision_gate": (
                        "valid-arm: compare decoded first-high diagnostic media"
                        if invariants
                        else "invalid-arm: do not interpret media"
                    ),
                }
            )


def patch_first_high_operator_comparison(
    model: Any,
    manifest_path: str,
    mode: str,
) -> tuple[Any, _State]:
    if mode not in _ALLOWED_MODES:
        raise ValueError(f"unsupported first-high W mode: {mode}")
    diagnostic = (getattr(model, "model_options", None) or {}).get(_diag.DIAGNOSTIC_KEY)
    if not isinstance(diagnostic, _diag._State):
        raise RuntimeError("first-high W must be applied after MiniMax H3 Execution Contract Diagnostics")
    if not diagnostic.strict_provenance or not diagnostic.manifest.get("gate_complete"):
        raise RuntimeError("first-high W requires strict, complete installed-runtime provenance")
    resolved, manifest, tensors = _replay._load_bundle(manifest_path)
    exact_checkpoint = _replay._checkpoint_identity(diagnostic.manifest)
    if not _replay._dict_equal(exact_checkpoint, manifest.get("exact_checkpoint_identity")):
        raise RuntimeError("first-high W exact checkpoint identity differs from R capture")
    source_gate = _verify_source_manifest()
    replay_state = _replay._ReplayState(
        manifest_path=str(resolved),
        manifest=manifest,
        tensors=tensors,
        exact_checkpoint_identity=exact_checkpoint,
    )
    patched = model.clone()
    _comfy_compat._copy_model_options(patched)
    transformer = patched.model_options.setdefault("transformer_options", {})
    if not isinstance(transformer, dict):
        raise RuntimeError("first-high W requires mutable transformer options")
    if "h3_flow_untwist_clock_trial_v1" in transformer or "h3_flow_sampling_context" in transformer:
        raise RuntimeError("first-high W requires the established no-Untwist R control")
    if (
        STATE_KEY in patched.model_options
        or _replay.CAPTURE_STATE_KEY in patched.model_options
        or _replay.REPLAY_STATE_KEY in patched.model_options
    ):
        raise RuntimeError("first-high W cannot be combined with R capture/replay wrappers")
    state = _State(replay=replay_state, mode=mode, source_gate=source_gate)
    patched.model_options[STATE_KEY] = state

    import comfy.patcher_extension

    _diag._insert_relative(
        patched,
        comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
        _runtime.OUTER_WRAPPER_KEY,
        _OUTER_KEY,
        _outer_wrapper,
        before=True,
    )
    _diag._insert_relative(
        patched,
        comfy.patcher_extension.WrappersMP.SAMPLER_SAMPLE,
        _diag._SAMPLER_KEY,
        _SAMPLER_KEY,
        _sampler_entry_wrapper,
        before=False,
    )
    return patched, state


def _saved_bundles() -> list[str]:
    return [name for name in _replay._saved_replay_manifests() if name.endswith(".json")]


class H3FirstHighOperatorComparison:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Experiment W: replay the exact saved R first-high state and stop after one real Euler model call. "
        "native_window keeps VDN restricted support + linear complement; native_full_support keeps the same Q groups "
        "with canonical full packed K/V and no complement. Diagnostic only."
    )
    RETURN_TYPES = ("MODEL", "H3_FLOW_FIRST_HIGH_OPERATOR_W")
    RETURN_NAMES = ("model", "comparison")
    FUNCTION = "apply"

    @classmethod
    def INPUT_TYPES(cls):
        bundles = _saved_bundles()
        if not bundles:
            bundles = ["<no saved R bundle>"]
        return {
            "required": {
                "model": ("MODEL",),
                "bundle": (bundles,),
                "mode": (list(_ALLOWED_MODES), {"default": "native_window"}),
            }
        }

    def apply(self, model, bundle, mode):
        if str(bundle) == "<no saved R bundle>":
            raise RuntimeError("no saved R bundle exists under output/h3_flow_replay")
        path = _replay._resolve_replay_bundle_selector(str(bundle))
        return patch_first_high_operator_comparison(model, str(path), str(mode))


class H3FirstHighOperatorComparisonReport:
    CATEGORY = "MiniMax H3/flow regenerate/diagnostic"
    DESCRIPTION = (
        "Emit W invariants after the one-call sampler returns its diagnostic x0. "
        "final_trajectory_available is always false; do not treat the trigger as a completed diffusion trajectory."
    )
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "extract"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "comparison": ("H3_FLOW_FIRST_HIGH_OPERATOR_W",),
                "trigger": ("LATENT",),
            }
        }

    def extract(self, comparison, trigger):
        _ = trigger
        if not isinstance(comparison, _State):
            raise TypeError("invalid first-high W comparison handle")
        if not comparison.complete:
            raise RuntimeError("no completed first-high W report is available")
        return (json.dumps(comparison.complete.pop(), indent=2, sort_keys=True, default=str),)


NODE_CLASS_MAPPINGS = {
    "H3FirstHighOperatorComparison": H3FirstHighOperatorComparison,
    "H3FirstHighOperatorComparisonReport": H3FirstHighOperatorComparisonReport,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3FirstHighOperatorComparison": "MiniMax H3 First-High Operator Comparison W",
    "H3FirstHighOperatorComparisonReport": "MiniMax H3 First-High Operator W Report",
}
