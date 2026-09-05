from types import SimpleNamespace

import pytest
import torch

from h3_flow_regenerate.contracts import H3FlowTrajectory
from h3_flow_regenerate.geometry import pack_streams
from h3_flow_regenerate.guidance import GuidanceConfig
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.runtime import FlowBinding, _run_progressive
from h3_flow_regenerate.target_sparse import (
    TARGET_SPARSE_CONTRACT_KEY,
    TargetSparsePlan,
    _lift_video_hidden,
    build_target_sparse_plan,
    exact_protected_video_patch_rows,
    make_target_sparse_block_wrapper,
    target_sparse_contract,
)


def _masked_target(*, target_t=2, target_h=4, target_w=4):
    video = torch.randn(1, 24, target_t, target_h, target_w)
    audio = torch.randn(1, 32, 2, 5)
    latent, shapes = pack_streams((video, audio))
    video_mask = torch.ones_like(video)
    video_mask[:, :, 0] = 0
    audio_mask = torch.ones_like(audio)
    denoise_mask, mask_shapes = pack_streams((video_mask, audio_mask))
    assert mask_shapes == shapes
    return latent, shapes, denoise_mask


def _layout_for(shapes):
    target_t, target_h, target_w = shapes[0][-3:]
    video_rows = target_t * (target_h // 2) * (target_w // 2)
    video_start = 4
    return SimpleNamespace(
        segments=[(0, 2, "text"), (2, 4, "audio"), (video_start, video_start + video_rows, "video")],
        signature=(2, target_t, target_h, target_w, 1),
        seq_len=video_start + video_rows,
    )


def test_exact_protected_rows_match_native_2x2_max_semantics():
    video = torch.ones(1, 24, 1, 4, 4)
    audio = torch.ones(1, 32, 2, 2)
    video[:, :, :, :2, :2] = 0
    # One non-zero value in the neighbouring patch means that row is generated.
    video[:, :, :, :2, 2:] = 0
    video[:, :, :, 0, 2] = 0.25
    packed, shapes = pack_streams((video, audio))

    rows = exact_protected_video_patch_rows(packed, shapes)

    assert rows.tolist() == [0]


def test_target_sparse_plan_retains_every_protected_row_plus_coarse_anchors():
    _latent, shapes, denoise_mask = _masked_target(target_t=2, target_h=8, target_w=8)
    plan = build_target_sparse_plan(denoise_mask, shapes, source_h=4, source_w=4)

    assert plan.target_video_rows == 32
    assert plan.protected_video_rows.tolist() == list(range(16))
    assert plan.anchor_video_row_count == 8
    assert set(plan.protected_video_rows.tolist()).issubset(set(plan.selected_video_rows.tolist()))
    assert set(plan.anchor_video_rows.tolist()).issubset(set(plan.selected_video_rows.tolist()))
    assert plan.selected_video_row_count < plan.target_video_rows


@pytest.mark.parametrize(
    ("target_h", "target_w", "source_h", "source_w"),
    [(2, 8, 2, 4), (8, 2, 4, 2)],
)
def test_target_sparse_lifter_endpoint_aligns_non_singleton_axis(target_h, target_w, source_h, source_w):
    selected = torch.tensor([0, 3], dtype=torch.long)
    plan = TargetSparsePlan(
        target_t=1,
        target_h=target_h,
        target_w=target_w,
        source_h=source_h,
        source_w=source_w,
        selected_video_rows=selected,
        anchor_video_rows=selected,
        protected_video_rows=torch.empty(0, dtype=torch.long),
    )
    compact = torch.tensor([[0.0], [1.0]])

    lifted = _lift_video_hidden(
        compact,
        plan=plan,
        selected_video=selected,
        anchor_positions=torch.tensor([0, 1], dtype=torch.long),
    )

    assert torch.allclose(lifted[:, 0], torch.tensor([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0]))


def test_target_sparse_block_contract_reduces_then_restores_dict_output():
    _latent, shapes, denoise_mask = _masked_target(target_t=2, target_h=4, target_w=4)
    plan = build_target_sparse_plan(denoise_mask, shapes, source_h=2, source_w=2)
    contract = target_sparse_contract(plan)
    layout = _layout_for(shapes)
    full_len = layout.seq_len
    hidden = 3
    full_img = torch.arange(full_len * hidden, dtype=torch.float32).reshape(full_len, hidden)
    rope = torch.arange(full_len, dtype=torch.float32).reshape(1, full_len, 1, 1, 1, 1)
    video_start = layout.segments[-1][0]
    video_mod = torch.arange(plan.target_video_rows, dtype=torch.long) + 10
    mod_segments = [(0, 2, 1), (2, 4, 2), (video_start, full_len, video_mod)]
    transformer_options = {TARGET_SPARSE_CONTRACT_KEY: contract}
    metrics = H3FlowMetrics()
    seen = []

    def first_previous(args, _extra):
        seen.append(args)
        return {"img": args["img"] + 1.0, "sentinel": "keep"}

    first = make_target_sparse_block_wrapper(0, 2, metrics, previous=first_previous)
    first_out = first(
        {
            "img": full_img,
            "t_emb": torch.zeros(1),
            "mod_segments": mod_segments,
            "rope_freqs": rope,
            "layout": layout,
            "transformer_options": transformer_options,
        },
        {"original_block": None},
    )

    reduced_len = video_start + plan.selected_video_row_count
    assert first_out["sentinel"] == "keep"
    assert first_out["img"].shape == (reduced_len, hidden)
    assert seen[0]["rope_freqs"].shape[1] == reduced_len
    reduced_video_mod = seen[0]["mod_segments"][-1][2]
    assert torch.equal(reduced_video_mod, video_mod.index_select(0, plan.selected_video_rows))

    def last_previous(args, _extra):
        return {"img": args["img"] * 2.0, "sentinel": "still-keep"}

    last = make_target_sparse_block_wrapper(1, 2, metrics, previous=last_previous)
    last_out = last(
        {
            "img": first_out["img"],
            "t_emb": torch.zeros(1),
            "mod_segments": mod_segments,
            "rope_freqs": rope,
            "layout": layout,
            "transformer_options": transformer_options,
        },
        {"original_block": None},
    )

    assert last_out["sentinel"] == "still-keep"
    assert last_out["img"].shape == (full_len, hidden)
    compact_after_last = first_out["img"][video_start:] * 2.0
    restored_selected = last_out["img"][video_start:].index_select(0, plan.selected_video_rows)
    assert torch.equal(restored_selected, compact_after_last)
    assert metrics.counters["target_sparse_actual_calls"] == 1
    assert any(event.kind == "target_sparse_lift" for event in metrics.events)


def test_target_sparse_wrapper_requires_native_block_replacement_dict_contract():
    _latent, shapes, denoise_mask = _masked_target()
    plan = build_target_sparse_plan(denoise_mask, shapes, source_h=2, source_w=2)
    layout = _layout_for(shapes)
    wrapper = make_target_sparse_block_wrapper(
        0,
        1,
        H3FlowMetrics(),
        previous=lambda args, _extra: args["img"],
    )
    full_len = layout.seq_len

    with pytest.raises(RuntimeError, match=r"must return .*img.*tensor"):
        wrapper(
            {
                "img": torch.zeros(full_len, 2),
                "mod_segments": [(0, 2, 1), (2, 4, 2), (4, full_len, 0)],
                "rope_freqs": torch.zeros(1, full_len, 1, 1, 1, 1),
                "layout": layout,
                "transformer_options": {TARGET_SPARSE_CONTRACT_KEY: target_sparse_contract(plan)},
            },
            {"original_block": None},
        )


class _IdentitySamplerModel:
    def __init__(self):
        self.model_sampling = SimpleNamespace(noise_scale=1.0)
        self.latent_shapes = None

    def process_latent_in(self, value):
        return value


def test_target_sparse_runtime_splits_two_target_grid_lifetimes_without_resizing_exact_prefix():
    target_latent, target_shapes, denoise_mask = _masked_target(target_t=2, target_h=8, target_w=6)
    target_noise = torch.randn_like(target_latent)
    sigmas = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])

    def native():
        pass

    native.__name__ = "sample_sa_solver_pece"
    sampler = SimpleNamespace(sampler_function=native, extra_options={})
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=_IdentitySamplerModel()),
        original_conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
        conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
    )
    calls = []
    binding = FlowBinding(
        trajectory=H3FlowTrajectory(),
        guidance=GuidanceConfig(mode="off"),
        capture_enabled=True,
    )

    class Executor:
        class_obj = guider

        def __call__(
            self,
            noise,
            latent,
            call_sampler,
            call_sigmas,
            mask,
            callback,
            disable_pbar,
            seed,
            *,
            latent_shapes,
        ):
            del callback, disable_pbar, seed
            transformer = dict(guider.model_options["transformer_options"])
            calls.append(
                (
                    noise.clone(),
                    latent,
                    call_sampler,
                    call_sigmas.clone(),
                    mask,
                    list(latent_shapes),
                    transformer,
                )
            )
            if len(calls) == 1:
                assert TARGET_SPARSE_CONTRACT_KEY in transformer
                return latent / (1.0 - float(call_sigmas[-1]))
            assert TARGET_SPARSE_CONTRACT_KEY not in transformer
            binding.metrics.event("model_call", actual=True, stage="high")
            return latent

    mutable_shapes = list(target_shapes)
    result = _run_progressive(
        Executor(),
        guider,
        binding,
        ProgressiveTargetInputConfig(
            source_latent_h=4,
            source_latent_w=4,
            handoff_coordinate=0.3,
            exact_prefix_mode="target_sparse_lifter",
        ),
        target_noise,
        target_latent,
        sampler,
        sigmas,
        denoise_mask,
        None,
        True,
        7,
        mutable_shapes,
    )

    assert result is target_latent
    assert len(calls) == 2
    assert calls[0][1] is target_latent
    assert calls[0][4] is denoise_mask
    assert calls[0][5] == target_shapes
    assert calls[1][1] is target_latent
    assert calls[1][4] is denoise_mask
    assert calls[1][5] == target_shapes
    protected = denoise_mask == 0
    assert torch.equal(calls[1][0][protected], target_noise[protected])
    assert mutable_shapes == target_shapes
    handoff = [event for event in binding.metrics.events if event.kind == "handoff_complete"][-1]
    assert handoff.fields["input_mode"] == "target_grid_sparse"
    assert handoff.fields["exact_probe_performed"] is False
    assert handoff.fields["source_latent_resize_performed"] is False
    assert handoff.fields["learned_transfer_performed"] is False
    assert handoff.fields["sampler_invocation_count"] == 2
    assert handoff.fields["history_boundary_count"] == 1
    assert binding.metrics.counters["progressive_target_sparse_runs"] == 1
    assert binding.metrics.counters["progressive_sampler_invocations"] == 2
    assert binding.metrics.counters["progressive_history_boundaries"] == 1


def test_target_sparse_high_failure_invalidates_committed_low_trajectory():
    target_latent, target_shapes, denoise_mask = _masked_target(target_t=2, target_h=8, target_w=6)
    sigmas = torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0])

    def native():
        pass

    native.__name__ = "sample_euler"
    sampler = SimpleNamespace(sampler_function=native, extra_options={})
    guider = SimpleNamespace(
        model_options={"transformer_options": {}},
        model_patcher=SimpleNamespace(model=_IdentitySamplerModel()),
        original_conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
        conds={"positive": [{"cross_attn": torch.zeros(1, 2, 4)}]},
    )
    calls = 0

    class Executor:
        class_obj = guider

        def __call__(self, noise, latent, call_sampler, call_sigmas, *args, latent_shapes):
            nonlocal calls
            del noise, call_sampler, args, latent_shapes
            calls += 1
            if calls == 1:
                return latent / (1.0 - float(call_sigmas[-1]))
            raise RuntimeError("synthetic target-sparse high failure")

    trajectory = H3FlowTrajectory()
    binding = FlowBinding(
        trajectory=trajectory,
        guidance=GuidanceConfig(mode="off"),
        capture_enabled=True,
    )
    with pytest.raises(RuntimeError, match="synthetic target-sparse high failure"):
        _run_progressive(
            Executor(),
            guider,
            binding,
            ProgressiveTargetInputConfig(
                source_latent_h=4,
                source_latent_w=4,
                handoff_coordinate=0.3,
                exact_prefix_mode="target_sparse_lifter",
            ),
            torch.randn_like(target_latent),
            target_latent,
            sampler,
            sigmas,
            denoise_mask,
            None,
            True,
            7,
            list(target_shapes),
        )

    assert len(trajectory.runs) == 1
    assert not trajectory.runs[0].complete
    assert "target-sparse progressive continuation failed" in trajectory.runs[0].abort_reason
    assert any(event.kind == "trajectory_invalidate" for event in binding.metrics.events)
