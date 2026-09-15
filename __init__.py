# ruff: noqa: I001
# Import order is intentional: the PR35 checkpoint diagnostic must install first
# so the learned-anchor validation wrapper can sit outside it and make those
# diagnostics observe the transported target-grid guidance representation. The
# execution-contract recorder is imported after both; its runtime-observation
# extension is imported immediately after the base recorder. The same-state
# replay and first-high operator diagnostics compose only additional wrappers
# around that recorder.
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
