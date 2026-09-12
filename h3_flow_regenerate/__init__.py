"""MiniMax H3 flow-aligned regeneration primitives and ComfyUI nodes."""

from .contracts import H3FlowTrajectory, TrajectoryRun, TrajectorySample
from .geometry import H3Geometry, normalize_target_geometry
from .sigma import audio_sigma, flow_shift, inverse_flow_shift

# Temporary PR #32 diagnostic overlay. Import before ComfyUI node modules so
# comfy_compat binds the instrumented high-stage prediction wrapper. This file
# is removed rather than promoted when the diagnostic branch is retired.
from . import pr32_prefix_proxy as _pr32_prefix_proxy  # noqa: F401,E402

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
