from __future__ import annotations

import logging
import os
import sys

# PR #32 is itself the opt-in surface when materialized by ComfyUI-Patcher.
# Activate the diagnostic automatically only when this root custom-node package
# is being loaded by a live ComfyUI process. Plain package/test imports must stay
# inert so the diagnostic cannot leak into unrelated execution modes.
_PR32_AUDIO_GUIDED_OVERLAP_ENV = "H3_FLOW_PR32_AUDIO_GUIDED_OVERLAP_TICKS"
_PR32_COMFYUI_RUNTIME = "folder_paths" in sys.modules and "comfy" in sys.modules
if _PR32_COMFYUI_RUNTIME:
    os.environ.setdefault(_PR32_AUDIO_GUIDED_OVERLAP_ENV, "4")
    logging.getLogger(__name__).info(
        "PR #32 audio guided overlap default=%s ticks (Patcher-applied diagnostic)",
        os.environ.get(_PR32_AUDIO_GUIDED_OVERLAP_ENV, ""),
    )

try:
    from .h3_flow_regenerate.decode_context import (
        NODE_CLASS_MAPPINGS as DECODE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.decode_context import (
        NODE_DISPLAY_NAME_MAPPINGS as DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from .h3_flow_regenerate.pr32_audio_boundary_pipeline import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_PIPELINE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.pr32_audio_boundary_pipeline import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_PIPELINE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.pr32_audio_decode_context import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_DECODE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.pr32_audio_decode_context import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.pr32_audio_decode_oracle import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_ORACLE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.pr32_audio_decode_oracle import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_ORACLE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from .h3_flow_regenerate.pr32_audio_phase_align import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_PHASE_NODE_CLASS_MAPPINGS,
    )
    from .h3_flow_regenerate.pr32_audio_phase_align import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_PHASE_NODE_DISPLAY_NAME_MAPPINGS,
    )
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
    from h3_flow_regenerate.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    from h3_flow_regenerate.pr32_audio_boundary_pipeline import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_PIPELINE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.pr32_audio_boundary_pipeline import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_PIPELINE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.pr32_audio_decode_context import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_DECODE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.pr32_audio_decode_context import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.pr32_audio_decode_oracle import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_ORACLE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.pr32_audio_decode_oracle import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_ORACLE_NODE_DISPLAY_NAME_MAPPINGS,
    )
    from h3_flow_regenerate.pr32_audio_phase_align import (
        NODE_CLASS_MAPPINGS as PR32_AUDIO_PHASE_NODE_CLASS_MAPPINGS,
    )
    from h3_flow_regenerate.pr32_audio_phase_align import (
        NODE_DISPLAY_NAME_MAPPINGS as PR32_AUDIO_PHASE_NODE_DISPLAY_NAME_MAPPINGS,
    )
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
    **PR32_AUDIO_DECODE_NODE_CLASS_MAPPINGS,
    **PR32_AUDIO_ORACLE_NODE_CLASS_MAPPINGS,
    **PR32_AUDIO_PHASE_NODE_CLASS_MAPPINGS,
    **PR32_AUDIO_PIPELINE_NODE_CLASS_MAPPINGS,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    **NODE_DISPLAY_NAME_MAPPINGS,
    **TARGET_SPARSE_NODE_DISPLAY_NAME_MAPPINGS,
    **DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    **PR32_AUDIO_DECODE_NODE_DISPLAY_NAME_MAPPINGS,
    **PR32_AUDIO_ORACLE_NODE_DISPLAY_NAME_MAPPINGS,
    **PR32_AUDIO_PHASE_NODE_DISPLAY_NAME_MAPPINGS,
    **PR32_AUDIO_PIPELINE_NODE_DISPLAY_NAME_MAPPINGS,
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
