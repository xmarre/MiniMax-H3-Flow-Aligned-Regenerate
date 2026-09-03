"""MiniMax H3 flow-aligned regeneration primitives and ComfyUI nodes."""

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
