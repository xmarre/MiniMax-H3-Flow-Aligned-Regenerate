from __future__ import annotations

import inspect
from types import SimpleNamespace

from h3_flow_regenerate import first_high_operator_comparison as w


def test_w_postprocess_passes_provenance_manifest_to_companion_observer():
    observation = {"after": {"sol": {"vdn_local_sol_calls": 0}}}
    provenance = {
        w._replay._RUNTIME_OBSERVATION_KEY: {
            "companion_calls": {"high_first": observation},
        }
    }
    record = SimpleNamespace(state=SimpleNamespace(manifest=provenance))

    assert w._replay._high_first_companion_observation(record.state.manifest) is observation

    source = inspect.getsource(w._outer_wrapper)
    assert "_replay._high_first_companion_observation(record.state.manifest)" in source
    assert "_replay._high_first_companion_observation(record)" not in source
