try:
    from .h3_flow_regenerate.audio_boundary_audit import (
        NODE_CLASS_MAPPINGS as AUDIO_AUDIT_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.audio_boundary_audit import (
        NODE_DISPLAY_NAME_MAPPINGS as AUDIO_AUDIT_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.decode_context import (
        NODE_CLASS_MAPPINGS as DECODE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.decode_context import (
        NODE_DISPLAY_NAME_MAPPINGS as DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.local_boundary_audit import (
        NODE_CLASS_MAPPINGS as LOCAL_AUDIT_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.local_boundary_audit import (
        NODE_DISPLAY_NAME_MAPPINGS as LOCAL_AUDIT_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .h3_flow_regenerate.partitioned_node import (
        NODE_CLASS_MAPPINGS as PARTITIONED_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.partitioned_node import (
        NODE_DISPLAY_NAME_MAPPINGS as PARTITIONED_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.target_sparse_node import (
        NODE_CLASS_MAPPINGS as TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.target_sparse_node import (
        NODE_DISPLAY_NAME_MAPPINGS as TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    )
except ImportError:  # Direct-file import used by packaging and test smoke checks.
    from h3_flow_regenerate.audio_boundary_audit import (
        NODE_CLASS_MAPPINGS as AUDIO_AUDIT_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.audio_boundary_audit import (
        NODE_DISPLAY_NAME_MAPPINGS as AUDIO_AUDIT_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.decode_context import (
        NODE_CLASS_MAPPINGS as DECODE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.decode_context import (
        NODE_DISPLAY_NAME_MAPPINGS as DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.local_boundary_audit import (
        NODE_CLASS_MAPPINGS as LOCAL_AUDIT_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.local_boundary_audit import (
        NODE_DISPLAY_NAME_MAPPINGS as LOCAL_AUDIT_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from h3_flow_regenerate.partitioned_node import (
        NODE_CLASS_MAPPINGS as PARTITIONED_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.partitioned_node import (
        NODE_DISPLAY_NAME_MAPPINGS as PARTITIONED_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.target_sparse_node import (
        NODE_CLASS_MAPPINGS as TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.target_sparse_node import (
        NODE_DISPLAY_NAME_MAPPINGS as TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    )

NODE_CLASS_MAPPINGS = {
    **LOCAL_AUDIT_NODE_CLASS_MAPPINGS,
    **AUDIO_AUDIT_NODE_CLASS_MAPPINGS,
    **NODE_CLASS_MAPPINGS,
    **TARGET_SPARSE_NODE_CLASS_MAPPINGS,
    **PARTITIONED_NODE_CLASS_MAPPINGS,
    **DECODE_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **LOCAL_AUDIT_NODE_DISPLAY_NAME_MAPPINGS,
    **AUDIO_AUDIT_NODE_DISPLAY_NAME_MAPPINGS,
    **NODE_DISPLAY_NAME_MAPPINGS,
    **TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    **PARTITIONED_NODE_DISPLAY_NAME_MAPPINGS,
    **DECODE_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
