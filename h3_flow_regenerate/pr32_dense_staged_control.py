from __future__ import annotations

from . import runtime
from . import target_sparse
from .target_sparse_node import H3ProgressiveTargetSparseHandoff


# 00413 showed that the previous "dense" control was not a matched sparse-vs-dense
# comparison: exact-prefix continuation fell all the way back to one ordinary
# target-grid sampler lifetime, which also removed the progressive low/high
# history boundary and Flow guidance and allowed Spectrum to forecast four calls.
# Keep the exact staged Target-Sparse execution skeleton instead, but make its
# early video token selection mathematically dense for the whole generated suffix.
# This isolates query-row pruning/lifting while retaining the 00412 sampler split,
# conditioning resets, exact-prefix mask/noise handling, high-stage guidance and
# companion interop contracts.
_PR32_FULL_SUFFIX_DENSE_COLLAR_T = 1 << 30


_original_target_sparse_exact_prefix = runtime._run_target_sparse_exact_prefix
if getattr(_original_target_sparse_exact_prefix, "_pr32_dense_staged_control", False):
    _dense_staged_exact_prefix = _original_target_sparse_exact_prefix
else:

    def _dense_staged_exact_prefix(*args, **kwargs):
        previous = target_sparse._BOUNDARY_DENSE_COLLAR_T
        target_sparse._BOUNDARY_DENSE_COLLAR_T = _PR32_FULL_SUFFIX_DENSE_COLLAR_T
        try:
            return _original_target_sparse_exact_prefix(*args, **kwargs)
        finally:
            target_sparse._BOUNDARY_DENSE_COLLAR_T = previous

    _dense_staged_exact_prefix._pr32_dense_staged_control = True
    _dense_staged_exact_prefix._pr32_original = _original_target_sparse_exact_prefix
    runtime._run_target_sparse_exact_prefix = _dense_staged_exact_prefix


class H3ProgressiveDenseStagedControl(H3ProgressiveTargetSparseHandoff):
    """PR #32 matched full-target-grid continuation control.

    The historical Target-Sparse class id is retained for workflow compatibility,
    but exact-prefix continuation uses the same staged low/probe/high execution
    skeleton as the sparse experiment while selecting every target-grid video row
    through the early H3 transformer. The final lifter therefore overwrites every
    row with its native transformer result and performs no effective interpolation.
    """

    EXACT_PREFIX_MODE = "target_sparse_lifter"
    DESCRIPTION = (
        "PR #32 matched dense-staged Continuum control using the historical Target-Sparse node id. "
        "Exact Native Masked continuation keeps the same low/high sampler lifetimes, conditioning resets, "
        "history boundary, exact-prefix mask/noise semantics and high-stage Flow guidance as Target-Sparse, "
        "but every generated target-grid video query row is retained through H3. This removes query pruning "
        "and effective hidden lifting without changing the staged continuation architecture. No extra logical "
        "H3 evaluations are introduced by this control; media validation remains required."
    )


NODE_CLASS_MAPPINGS = {
    "H3ProgressiveTargetSparseHandoff": H3ProgressiveDenseStagedControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ProgressiveTargetSparseHandoff": "MiniMax H3 Progressive Dense Continuation (PR32 Control)",
}
