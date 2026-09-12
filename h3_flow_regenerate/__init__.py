"""MiniMax H3 flow-aligned regeneration primitives and ComfyUI nodes."""

# Temporary PR #32 diagnostic overlay. Import before ComfyUI node modules so
# runtime target-sparse dispatch is instrumented before comfy_compat binds the
# outer wrapper. This file is removed rather than promoted when the diagnostic
# branch is retired.
from . import pr32_target_sparse_audio_probe as _pr32_target_sparse_audio_probe  # noqa: F401
from .contracts import H3FlowTrajectory, TrajectoryRun, TrajectorySample
from .geometry import H3Geometry, normalize_target_geometry
from .sigma import audio_sigma, flow_shift, inverse_flow_shift

__all__ = [
    "H3FlowTrajectory",
    "H3Geometry",
    "TrajectoryRun",
    "TrajectorySample",
    "audio_sigma",
    "flow_shift",
    "inverse_flow_shift",
    "normalize_target_geometry",
]
