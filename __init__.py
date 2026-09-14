# ruff: noqa: I001
# Import order is intentional: PR35 checkpoint capture installs first, PR36
# learned-anchor guidance transport sits outside it, and the target-state
# validation wrapper installs last so it can replace only the final handoff state.
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
    from .h3_flow_regenerate.target_input_state_transport_validation import (
        install_target_input_state_transport_validation,
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
    from h3_flow_regenerate.target_input_state_transport_validation import (
        install_target_input_state_transport_validation,
    )
    from h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from h3_flow_regenerate.target_sparse_node import (
        NODE_CLASS_MAPPINGS as TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.target_sparse_node import (
        NODE_DISPLAY_NAME_MAPPINGS as TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    )

install_target_input_state_transport_validation()

NODE_CLASS_MAPPINGS = {
    **NODE_CLASS_MAPPINGS,
    **TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    **DECODE_NODE_CLASS_MAPPINGS,
    **HANDOFF_DIAGNOSTIC_NODE_CLASS_MAPPINGS,
    **GUIDANCE_ANCHOR_VALIDATION_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    **TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    **DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    **HANDOFF_DIAGNOSTIC_NODE_DISPLAY_NAME_MAPPINGS,
    **GUIDANCE_ANCHOR_VALIDATION_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
