from __future__ import annotations

import torch

from . import comfy_compat, runtime, target_sparse
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


# 00414 reached the intended all-row plan (selected_video_rows == target_video_rows)
# but then failed before the first continuation H3 evaluation. The ordinary
# Target-Sparse block wrapper still published VDN's "external reduced sequence"
# contract even though the hidden stream was no longer reduced. VDN correctly
# rejects that stale contract on a full native sequence. Keep the same installed
# wrapper chain, but make the mathematically-dense case a true identity transport:
# no row indexing, no reduced RoPE/modulation metadata, no VDN reduced-sequence
# contract, and no final interpolation. Partial sparse plans still delegate to the
# original implementation unchanged.
_original_make_target_sparse_block_wrapper = comfy_compat.make_target_sparse_block_wrapper
if getattr(_original_make_target_sparse_block_wrapper, "_pr32_dense_identity_aware", False):
    _dense_identity_aware_target_sparse_block_wrapper = _original_make_target_sparse_block_wrapper
else:

    def _dense_identity_aware_target_sparse_block_wrapper(
        layer: int,
        num_layers: int,
        metrics,
        previous=None,
    ):
        sparse_wrapper = _original_make_target_sparse_block_wrapper(
            layer,
            num_layers,
            metrics,
            previous=previous,
        )

        def call_next(args, extra):
            if previous is not None:
                return previous(args, extra)
            return extra["original_block"](args)

        def wrapper(args, extra):
            transformer = args.get("transformer_options", {}) or {}
            contract = transformer.get(target_sparse.TARGET_SPARSE_CONTRACT_KEY)
            plan = contract.get("plan") if isinstance(contract, dict) else None
            dense_identity = (
                isinstance(contract, dict)
                and contract.get("active") is True
                and int(contract.get("api", -1)) == target_sparse.TARGET_SPARSE_API_VERSION
                and isinstance(plan, target_sparse.TargetSparsePlan)
                and plan.selected_video_row_count == plan.target_video_rows
            )
            if not dense_identity:
                return sparse_wrapper(args, extra)

            layout = args.get("layout")
            video_start, video_stop = target_sparse._video_segment(layout)
            if video_stop - video_start != plan.target_video_rows:
                raise RuntimeError(
                    "PR32 dense staged plan/video layout mismatch: "
                    f"plan={plan.target_video_rows} layout={video_stop - video_start}"
                )
            img = args.get("img")
            rope = args.get("rope_freqs")
            if not torch.is_tensor(img) or img.ndim != 2 or int(img.shape[0]) != video_stop:
                raise RuntimeError("PR32 dense staged control requires the full native packed hidden stream")
            if not torch.is_tensor(rope) or rope.ndim < 2 or int(rope.shape[1]) != video_stop:
                raise RuntimeError("PR32 dense staged control requires full target-grid RoPE rows")
            if target_sparse.VDN_EXTERNAL_SEQUENCE_KEY in transformer:
                raise RuntimeError("PR32 dense staged control received a stale VDN external-sequence contract")

            output = call_next(args, extra)
            if not isinstance(output, dict) or "img" not in output or not torch.is_tensor(output["img"]):
                raise RuntimeError("PR32 dense staged wrapped H3 block must return {'img': tensor}")
            result = output["img"]
            if result.ndim != 2 or int(result.shape[0]) != video_stop:
                raise RuntimeError("PR32 dense staged wrapped H3 block changed the full native hidden length")

            if layer == 0:
                metrics.increment("target_sparse_actual_calls")
                metrics.event(
                    "target_sparse_transformer",
                    layer=layer,
                    full_sequence_rows=video_stop,
                    reduced_sequence_rows=video_stop,
                    full_video_rows=plan.target_video_rows,
                    selected_video_rows=plan.selected_video_row_count,
                    anchor_video_rows=plan.anchor_video_row_count,
                    protected_video_rows=plan.protected_video_row_count,
                    dense_collar_video_rows=plan.dense_collar_video_row_count,
                    dense_collar_t=plan.dense_collar_t,
                    video_row_fraction=plan.video_row_fraction,
                    source_hw=(plan.source_h, plan.source_w),
                    target_hw=(plan.target_h, plan.target_w),
                    target_t=plan.target_t,
                    exact_protected_rows_retained=True,
                    target_grid_rope_retained=True,
                    vdn_external_sequence_mode="native_full_grid",
                    dense_identity=True,
                )

            if layer == num_layers - 1:
                metrics.event(
                    "target_sparse_lift",
                    layer=layer,
                    restored_sequence_rows=video_stop,
                    restored_video_rows=plan.target_video_rows,
                    retained_rows_overwritten_exactly=plan.selected_video_row_count,
                    interpolation="identity_full_grid",
                    dense_identity=True,
                )
            return output

        wrapper._h3_flow_target_sparse_wrapper = True
        wrapper._h3_flow_target_sparse_previous = previous
        wrapper._h3_flow_target_sparse_metrics = metrics
        wrapper._pr32_dense_identity_aware = True
        return wrapper

    _dense_identity_aware_target_sparse_block_wrapper._pr32_dense_identity_aware = True
    _dense_identity_aware_target_sparse_block_wrapper._pr32_original = _original_make_target_sparse_block_wrapper
    comfy_compat.make_target_sparse_block_wrapper = _dense_identity_aware_target_sparse_block_wrapper


class H3ProgressiveDenseStagedControl(H3ProgressiveTargetSparseHandoff):
    """PR #32 matched full-target-grid continuation control.

    The historical Target-Sparse class id is retained for workflow compatibility,
    but exact-prefix continuation uses the same staged low/probe/high execution
    skeleton as the sparse experiment while selecting every target-grid video row
    through the early H3 transformer. The mathematically-dense wrapper case keeps
    the native full packed sequence end-to-end, so VDN never receives a reduced-
    sequence contract and the final sparse lifter is an identity operation.
    """

    EXACT_PREFIX_MODE = "target_sparse_lifter"
    DESCRIPTION = (
        "PR #32 matched dense-staged Continuum control using the historical Target-Sparse node id. "
        "Exact Native Masked continuation keeps the same low/high sampler lifetimes, conditioning resets, "
        "history boundary, exact-prefix mask/noise semantics and high-stage Flow guidance as Target-Sparse, "
        "but every generated target-grid video query row is retained through H3. The full native packed sequence "
        "is preserved through VDN and no bilinear hidden reconstruction is performed. No extra logical H3 "
        "evaluations are introduced by this control; media validation remains required."
    )


NODE_CLASS_MAPPINGS = {
    "H3ProgressiveTargetSparseHandoff": H3ProgressiveDenseStagedControl,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "H3ProgressiveTargetSparseHandoff": "MiniMax H3 Progressive Dense Continuation (PR32 Control)",
}
