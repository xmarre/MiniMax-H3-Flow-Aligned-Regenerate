from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from test_geometric_bridge import pattern, transform_video
from test_handoff import FakeLearnedProvider

from h3_flow_regenerate.geometric_bridge import geometric_seam_bridge
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.handoff import ProgressiveTargetInputConfig, deterministic_video_noise
from h3_flow_regenerate.nodes import H3ProgressiveTargetInputHandoff
from h3_flow_regenerate.runtime import FlowBinding, _run_progressive
from h3_flow_regenerate.seam_diagnostics import recover_conditional_clean_for_diagnostics
from h3_flow_regenerate.target_sparse_node import H3ProgressiveMixedGridHandoff, H3ProgressiveTargetSparseHandoff
from h3_flow_regenerate.tone_bridge import apply_suffix_dc_bridge, map_clean_bridge_to_conditional_state


@pytest.mark.parametrize("enabled,shift", [(False, True), (True, False), (True, True)])
@pytest.mark.parametrize("dc", [False, True])
def test_runtime_pre_renoise_order_and_released_fallback(monkeypatch, enabled, shift, dc):
    fake = ModuleType("comfy")
    fake.samplers = ModuleType("comfy.samplers")
    fake.samplers.KSAMPLER = lambda function, **kw: SimpleNamespace(sampler_function=function, extra_options={})
    monkeypatch.setitem(sys.modules, "comfy", fake)
    monkeypatch.setitem(sys.modules, "comfy.samplers", fake.samplers)
    learned = pattern(7).repeat(1, 3, 1, 1, 1)
    exact = transform_video(learned[:, :, :3], (1.02, 0.99, 0.75, -0.5)) if shift else learned[:, :, :3].clone()
    exact = exact + 2
    video = learned.clone()
    video[:, :, :3] = exact
    audio = torch.randn(1, 32, 2, 11)
    packed, shapes = pack_streams((video, audio))
    vm = torch.ones_like(video)
    vm[:, :, :3] = 0
    mask = pack_streams((vm, torch.ones_like(audio)))[0]
    caller_noise = torch.randn_like(packed)
    saved_noise, saved_mask, saved_learned = caller_noise.clone(), mask.clone(), learned.clone()
    base = SimpleNamespace(process_latent_in=lambda x: x, diffusion_model=SimpleNamespace(blocks=[]))
    guider = SimpleNamespace(
        model_options={"transformer_options": {}}, model_patcher=SimpleNamespace(model=base), conds={"positive": []}
    )
    binding = FlowBinding()
    provider = FakeLearnedProvider(learned)
    config = ProgressiveTargetInputConfig(
        source_latent_h=40,
        source_latent_w=54,
        exact_prefix_mode="mixed_grid_low_suffix",
        transfer_mode="learned_3d",
        learned_upscaler=provider,
        suffix_dc_bridge=dc,
        suffix_geometric_bridge=enabled,
    )
    captured = []

    def execute(noise, latent, sampler, sigmas, call_mask, *args, latent_shapes):
        stage = guider.model_options["transformer_options"]["h3_flow_stage"]
        if stage == "high":
            assert torch.equal(call_mask, mask)
            captured.append((noise.clone(), float(sigmas[0])))
            binding.metrics.event("model_call", actual=True)
            return packed.clone()
        if stage == "probe":
            return latent.clone()
        return latent / (1 - sigmas[-1])

    # Reuse binding and guider to exercise invocation-local state and chunk lifetimes.
    for _ in range(2):
        result = _run_progressive(
            execute,
            guider,
            binding,
            config,
            caller_noise,
            packed,
            SimpleNamespace(sampler_function=lambda: None, extra_options={}),
            torch.tensor([1.0, 0.9, 0.7, 0.4, 0.0]),
            mask,
            None,
            True,
            7,
            list(shapes),
        )
        assert torch.equal(result, packed)
    assert torch.equal(captured[0][0], captured[1][0])
    high_noise, sigma = captured[0]
    handoff_noise = deterministic_video_noise(
        tuple(learned.shape), seed=7 + config.seed_offset, device=learned.device, dtype=learned.dtype
    )
    geometric, aligned, report = geometric_seam_bridge(learned, exact, source_hw=(40, 54), requested=enabled)
    if report["accepted"]:
        if dc:
            calibrated = geometric.clone()
            calibrated[:, :, 2] = aligned
            geometric, _ = apply_suffix_dc_bridge(calibrated, exact)
        state = (1 - sigma) * geometric + sigma * handoff_noise
    else:
        # Exact released v0.3.2 calculation, including inverse recovery rounding.
        state = (1 - sigma) * learned + sigma * handoff_noise
        if dc:
            recovered = recover_conditional_clean_for_diagnostics(state, handoff_noise, sigma=sigma)
            corrected, _ = apply_suffix_dc_bridge(recovered, exact)
            state = map_clean_bridge_to_conditional_state(
                state, recovered, corrected, sigma=sigma, prefix_t=3, corrected_tokens=1
            )
    expected = (state - (1 - sigma) * video) / sigma
    got_video, got_audio = unpack_streams(high_noise, shapes)
    assert torch.equal(got_video[:, :, 3:], expected[:, :, 3:])
    # Audio low-stage state is unaffected by either bridge; same deterministic fixture.
    expected_audio = (audio - (1 - sigma) * audio) / sigma
    assert torch.allclose(got_audio, expected_audio, atol=1e-6, rtol=1e-6)
    original_video_noise, _ = unpack_streams(caller_noise, shapes)
    assert torch.equal(got_video[:, :, :3], original_video_noise[:, :, :3])
    assert torch.equal(caller_noise, saved_noise) and torch.equal(mask, saved_mask)
    assert torch.equal(learned, saved_learned)
    assert len(provider.calls) == 2
    geometry = [e.fields for e in binding.metrics.events if e.kind == "mixed_grid_geometry"]
    assert len(geometry) == 2 and all(e["final_prefix_exact"] for e in geometry)
    assert all(e["accepted"] == (enabled and shift) for e in geometry)
    assert "h3_flow_mixed_grid_v1" not in guider.model_options["transformer_options"]
    assert binding.metrics.counters["handoff_exact_probe_nfe"] == 2


def test_node_scope_and_old_workflow_default():
    for node in (H3ProgressiveTargetInputHandoff, H3ProgressiveTargetSparseHandoff):
        inputs = node.INPUT_TYPES()
        assert all("suffix_geometric_bridge" not in fields for fields in inputs.values())
    assert H3ProgressiveMixedGridHandoff.INPUT_TYPES()["optional"]["suffix_geometric_bridge"][1]["default"] is False
    assert ProgressiveTargetInputConfig(source_scale=0.7).suffix_geometric_bridge is False
    with pytest.raises(ValueError, match="mixed-grid"):
        ProgressiveTargetInputConfig(source_scale=0.7, suffix_geometric_bridge=True)
    with pytest.raises(TypeError, match="boolean"):
        ProgressiveTargetInputConfig(source_scale=0.7, suffix_geometric_bridge="false")
