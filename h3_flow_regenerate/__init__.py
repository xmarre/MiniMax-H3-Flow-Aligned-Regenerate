"""MiniMax H3 flow-aligned regeneration primitives and ComfyUI nodes."""

from __future__ import annotations

import logging
import os

# PR #32 is a diagnostic overlay applied through ComfyUI-Patcher. Applying the
# PR must therefore activate the experiment without requiring an unrelated shell
# environment step. Keep the env variable as an explicit override/bisect hook:
# setting H3_FLOW_PR32_AUDIO_GUIDED_OVERLAP_TICKS=0 disables it.
_PR32_AUDIO_GUIDED_OVERLAP_ENV = "H3_FLOW_PR32_AUDIO_GUIDED_OVERLAP_TICKS"
os.environ.setdefault(_PR32_AUDIO_GUIDED_OVERLAP_ENV, "4")
logging.getLogger(__name__).info(
    "PR #32 audio guided overlap default=%s ticks (Patcher-applied diagnostic)",
    os.environ.get(_PR32_AUDIO_GUIDED_OVERLAP_ENV, ""),
)

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
