from pathlib import Path

runtime_path = Path("h3_flow_regenerate/execution_contract_runtime_observation.py")
text = runtime_path.read_text()
replacements = [
    (
        '''    block_encoded = json.dumps(block_state, sort_keys=True, separators=(",", ":"), default=str).encode()
    modified_modules = _modified_module_manifest(diffusion)
    lifecycle_material = {
        "blocks": block_state,
        "modified_modules": modified_modules["modules"],
    }
''',
        '''    block_encoded = json.dumps(block_state, sort_keys=True, separators=(",", ":"), default=str).encode()
    modified_modules = _modified_module_manifest(diffusion)
    active_injections = _injection_manifest(model)
    active_object_patches = _object_patch_manifest(model)
    effective_wrapper_order = _keyed_callable_manifest(transformer.get("wrappers", {}))
    effective_callbacks = _keyed_callable_manifest(transformer.get("callbacks", {}))
    lifecycle_material = {
        "blocks": block_state,
        "modified_modules": modified_modules["modules"],
        "active_injections": active_injections,
        "active_object_patches": active_object_patches,
        "effective_wrapper_order": effective_wrapper_order,
        "effective_callbacks": effective_callbacks,
    }
''',
        "runtime lifecycle material",
    ),
    (
        '''        "active_injections": _injection_manifest(model),
        "active_object_patches": _object_patch_manifest(model),
        "effective_wrapper_order": _keyed_callable_manifest(transformer.get("wrappers", {})),
        "effective_callbacks": _keyed_callable_manifest(transformer.get("callbacks", {})),
''',
        '''        "active_injections": active_injections,
        "active_object_patches": active_object_patches,
        "effective_wrapper_order": effective_wrapper_order,
        "effective_callbacks": effective_callbacks,
''',
        "runtime lifecycle return",
    ),
    (
        '''    stage_lifecycle_complete = all(by_stage.get(stage) for stage in _REQUIRED_STAGES)
    injection_state_consistent = stage_lifecycle_complete and all(
''',
        '''    stage_lifecycle_counts = {stage: len(by_stage.get(stage, [])) for stage in _REQUIRED_STAGES}
    stage_lifecycle_complete = all(stage_lifecycle_counts[stage] == 1 for stage in _REQUIRED_STAGES)
    injection_state_consistent = stage_lifecycle_complete and all(
''',
        "stage lifecycle gate",
    ),
    (
        '''    hook_runtime_stable = bool(lifecycle_digests) and len(set(lifecycle_digests)) == 1
    modified_states = [
''',
        '''    hook_runtime_stable = len(lifecycle_digests) == len(_REQUIRED_STAGES) and len(set(lifecycle_digests)) == 1
    modified_states = [
''',
        "lifecycle digest gate",
    ),
    (
        '''    companion_slots_complete = all(slot in companion_calls for slot in _REQUIRED_COMPANION_SLOTS)

    high_first = companion_calls.get("high_first") or {}
''',
        '''    companion_slots_complete = all(slot in companion_calls for slot in _REQUIRED_COMPANION_SLOTS)
    expected_actual_counts = {"low": 4, "probe": 1, "high": 2}
    actual_counts = extension.get("diffusion_actual_counts") or {}
    observed_actual_counts = {stage: int(actual_counts.get(stage, 0)) for stage in _REQUIRED_STAGES}
    actual_call_counts_exact = observed_actual_counts == expected_actual_counts

    high_first = companion_calls.get("high_first") or {}
''',
        "actual call count gate",
    ),
    (
        '''            modified_runtime_visible,
            companion_slots_complete,
            sol_visible,
''',
        '''            modified_runtime_visible,
            companion_slots_complete,
            actual_call_counts_exact,
            sol_visible,
''',
        "runtime contract tuple",
    ),
    (
        '''        "stage_lifecycle_complete": stage_lifecycle_complete,
        "injection_state_consistent": injection_state_consistent,
''',
        '''        "stage_lifecycle_complete": stage_lifecycle_complete,
        "stage_lifecycle_counts": stage_lifecycle_counts,
        "injection_state_consistent": injection_state_consistent,
''',
        "runtime gate report lifecycle",
    ),
    (
        '''        "companion_slots_complete": companion_slots_complete,
        "sol_active_high_first_visible": sol_visible,
''',
        '''        "companion_slots_complete": companion_slots_complete,
        "actual_call_counts_exact": actual_call_counts_exact,
        "expected_actual_call_counts": expected_actual_counts,
        "observed_actual_call_counts": observed_actual_counts,
        "sol_active_high_first_visible": sol_visible,
''',
        "runtime gate report counts",
    ),
]
for old, new, name in replacements:
    if text.count(old) != 1:
        raise SystemExit(f"{name} anchor mismatch: {text.count(old)}")
    text = text.replace(old, new)
runtime_path.write_text(text)

test_path = Path("tests/test_execution_contract_runtime_observation.py")
text = test_path.read_text()
replacements = [
    (
        "from __future__ import annotations\n\nimport contextvars\n",
        "from __future__ import annotations\n\nimport contextvars\nimport copy\n",
        "test import",
    ),
    (
        '''        "schema_version": 2,
        "stage_lifecycles": [
            {"stage": "low", **lifecycle},
            {"stage": "probe", **lifecycle},
            {"stage": "high", **lifecycle},
        ],
        "companion_calls": {
            "low_last": {"stage": "low", "actual_index": 4, **companion},
            "probe": {"stage": "probe", "actual_index": 1, **companion},
            "high_first": {"stage": "high", "actual_index": 1, **companion},
            "high_last": {"stage": "high", "actual_index": 2, **companion},
        },
        "observation_errors": [],
''',
        '''        "schema_version": 2,
        "stage_lifecycles": [
            {"stage": "low", **copy.deepcopy(lifecycle)},
            {"stage": "probe", **copy.deepcopy(lifecycle)},
            {"stage": "high", **copy.deepcopy(lifecycle)},
        ],
        "diffusion_actual_counts": {"low": 4, "probe": 1, "high": 2},
        "companion_calls": {
            "low_last": {"stage": "low", "actual_index": 4, **copy.deepcopy(companion)},
            "probe": {"stage": "probe", "actual_index": 1, **copy.deepcopy(companion)},
            "high_first": {"stage": "high", "actual_index": 1, **copy.deepcopy(companion)},
            "high_last": {"stage": "high", "actual_index": 2, **copy.deepcopy(companion)},
        },
        "observation_errors": [],
''',
        "complete observation helper",
    ),
    (
        '''    assert runtime["stage_lifecycle_complete"] is True
    assert runtime["injection_state_consistent"] is True
''',
        '''    assert runtime["stage_lifecycle_complete"] is True
    assert runtime["stage_lifecycle_counts"] == {"low": 1, "probe": 1, "high": 1}
    assert runtime["injection_state_consistent"] is True
''',
        "complete gate lifecycle assertion",
    ),
    (
        '''    assert runtime["modified_runtime_visible"] is True
    assert runtime["observation_error_free"] is True
''',
        '''    assert runtime["modified_runtime_visible"] is True
    assert runtime["actual_call_counts_exact"] is True
    assert runtime["observed_actual_call_counts"] == {"low": 4, "probe": 1, "high": 2}
    assert runtime["observation_error_free"] is True
''',
        "complete gate count assertion",
    ),
]
for old, new, name in replacements:
    if text.count(old) != 1:
        raise SystemExit(f"{name} anchor mismatch: {text.count(old)}")
    text = text.replace(old, new)
text += '''


def test_report_rejects_duplicate_stage_lifecycle_capture():
    runtime_observation = _complete_runtime_observation()
    duplicate = copy.deepcopy(runtime_observation["stage_lifecycles"][2])
    runtime_observation["stage_lifecycles"].append(duplicate)
    state = _state()
    state.complete = deque(
        [
            {
                "provenance": {observation._EXTENSION_KEY: runtime_observation},
                "observation_gate": {"structural_candidate": True},
                "promotion": {"production_fix_authorized": False},
            }
        ],
        maxlen=4,
    )

    report_text = observation.H3ExecutionContractReport().extract(state, None)[0]
    report = json.loads(report_text)
    runtime = report["observation_gate"]["runtime_contract"]
    assert runtime["stage_lifecycle_counts"] == {"low": 1, "probe": 1, "high": 2}
    assert runtime["stage_lifecycle_complete"] is False
    assert runtime["runtime_contract_complete"] is False
    assert report["observation_gate"]["structural_candidate"] is False


def test_report_rejects_unexpected_actual_diffusion_call_count():
    runtime_observation = _complete_runtime_observation()
    runtime_observation["diffusion_actual_counts"]["high"] = 3
    state = _state()
    state.complete = deque(
        [
            {
                "provenance": {observation._EXTENSION_KEY: runtime_observation},
                "observation_gate": {"structural_candidate": True},
                "promotion": {"production_fix_authorized": False},
            }
        ],
        maxlen=4,
    )

    report_text = observation.H3ExecutionContractReport().extract(state, None)[0]
    report = json.loads(report_text)
    runtime = report["observation_gate"]["runtime_contract"]
    assert runtime["actual_call_counts_exact"] is False
    assert runtime["observed_actual_call_counts"] == {"low": 4, "probe": 1, "high": 3}
    assert runtime["runtime_contract_complete"] is False
    assert report["observation_gate"]["structural_candidate"] is False
'''
test_path.write_text(text)
