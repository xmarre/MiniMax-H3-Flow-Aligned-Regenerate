from types import SimpleNamespace

import pytest
import torch

import h3_flow_regenerate.first_high_counterfactual as counterfactual
from h3_flow_regenerate.first_high_counterfactual import (
    FIRST_HIGH_COUNTERFACTUAL_ARMS,
    FIRST_HIGH_COUNTERFACTUAL_KEY,
    FirstHighCounterfactualRequest,
    H3FirstHighCounterfactual,
    arm_tensor_name,
    maybe_run_first_high_counterfactual,
)
from h3_flow_regenerate.geometry import pack_streams, unpack_streams
from h3_flow_regenerate.metrics import H3FlowMetrics
from h3_flow_regenerate.partitioned_stage import PARTITIONED_STAGE_KEY, PartitionedStagePlan, PartitionedStageRuntime
from h3_flow_regenerate.residual_evidence import BoundaryWindowEvidence

PREFIX_T, TEMPORAL, H, W = 12, 22, 4, 6
SPECTRUM_KEYS = {"spectrum_h3_runtime": object(), "spectrum_h3_run_id": 3, "spectrum_h3_step_id": 7}


class Guider:
    def __init__(self, request):
        self.model_options = {FIRST_HIGH_COUNTERFACTUAL_KEY: request} if request is not None else {}
        self.inner_model = SimpleNamespace(
            latent_shapes=[(1, 24, TEMPORAL, H, W), (1, 32, 2, 8)],
            model_sampling=SimpleNamespace(noise_scale=1.0),
            diffusion_model=SimpleNamespace(final_layer=SimpleNamespace(video_out=torch.nn.Linear(4, 8))),
        )
        self.inner_model.diffusion_model.final_layer.video_out.out_features = 4
        self.calls = []

    def predict_noise(self, x, timestep, model_options, seed):
        self.calls.append((x.clone(), model_options))
        torch.randn(3)
        return x * 0.5


ALL_ARMS = FirstHighCounterfactualRequest(FIRST_HIGH_COUNTERFACTUAL_ARMS, 5)


def _setup(request=ALL_ARMS):
    torch.manual_seed(0)
    exact = torch.randn(1, 24, PREFIX_T, H, W)
    provider = torch.randn(1, 24, TEMPORAL, H, W)
    pre_high = torch.randn(1, 24, TEMPORAL, H, W)
    pre_high[:, :, :PREFIX_T] = exact
    evidence = BoundaryWindowEvidence(exact, TEMPORAL, capture_full_video=True)
    evidence.capture("provider_native_clean", provider)
    evidence.capture("pre_high_exact_restored", pre_high)
    sigma = 0.878
    prefix_noise = torch.randn_like(exact)
    video = (1 - sigma) * pre_high + sigma * torch.randn_like(pre_high)
    video[:, :, :PREFIX_T] = 0.999 * exact + 0.001 * prefix_noise
    audio = torch.randn(1, 32, 2, 8)
    x = pack_streams((video, audio))[0]
    production = x * 0.25
    evidence.observe_prediction(
        unpack_streams(production, Guider(None).inner_model.latent_shapes)[0],
        point="before_flow",
        call_index=0,
        sigma=sigma,
        actual=True,
    )
    metrics = H3FlowMetrics()
    runtime = PartitionedStageRuntime(
        plan=PartitionedStagePlan(prefix=exact, temporal=TEMPORAL, source_h=H, source_w=W, prefix_noise=prefix_noise),
        metrics=metrics,
    )
    options = {"transformer_options": {PARTITIONED_STAGE_KEY: runtime, "sample_sigmas": torch.tensor([sigma, 0.5])}}
    options["transformer_options"].update(SPECTRUM_KEYS)
    guider = Guider(request)
    return SimpleNamespace(
        guider=guider,
        evidence=evidence,
        trace=SimpleNamespace(window_evidence=evidence),
        x=x,
        production=production,
        options=options,
        metrics=metrics,
        runtime=runtime,
        exact=exact,
        provider=provider,
        pre_high=pre_high,
        sigma=sigma,
    )


def _run(case, call_index=0):
    maybe_run_first_high_counterfactual(
        case.guider,
        case.x,
        torch.tensor([case.sigma]),
        case.options,
        11,
        production=case.production,
        trace=case.trace,
        call_index=call_index,
        sigma=case.sigma,
        actual=True,
        metrics=case.metrics,
        stage_key=PARTITIONED_STAGE_KEY,
    )


def test_each_arm_changes_only_its_operand_and_is_isolated():
    case = _setup()
    production = case.production.clone()
    x = case.x.clone()
    rng = torch.random.get_rng_state()
    _run(case)
    assert torch.equal(case.production, production) and torch.equal(case.x, x)
    assert torch.equal(torch.random.get_rng_state(), rng)
    shapes = case.guider.inner_model.latent_shapes
    calls = dict(zip(FIRST_HIGH_COUNTERFACTUAL_ARMS, case.guider.calls, strict=True))
    base_video, base_audio = unpack_streams(case.x, shapes)
    for arm, (arm_x, options) in calls.items():
        transformer = options["transformer_options"]
        assert not set(SPECTRUM_KEYS) & set(transformer)
        assert torch.equal(unpack_streams(arm_x, shapes)[1], base_audio)
        window = case.evidence.tensors[arm_tensor_name(arm)]
        assert window.shape == (1, 24, 7, H, W)
        assert torch.allclose(window, unpack_streams(arm_x * 0.5, shapes)[0][:, :, 10:17])
        if arm == "native_high_attention":
            assert PARTITIONED_STAGE_KEY not in transformer
        else:
            assert transformer[PARTITIONED_STAGE_KEY].metrics is not case.metrics
    assert torch.equal(calls["replay"][0], case.x)
    assert torch.equal(calls["native_high_attention"][0], case.x)
    provider_video = unpack_streams(calls["provider_prefix"][0], shapes)[0]
    expected = base_video[:, :, :PREFIX_T] + 0.999 * (case.provider[:, :, :PREFIX_T] - case.exact)
    assert torch.allclose(provider_video[:, :, :PREFIX_T], expected)
    assert torch.equal(provider_video[:, :, PREFIX_T:], base_video[:, :, PREFIX_T:])
    plan = calls["provider_prefix"][1]["transformer_options"][PARTITIONED_STAGE_KEY].plan
    assert torch.equal(plan.prefix, case.provider[:, :, :PREFIX_T])
    assert case.runtime.plan.prefix is not plan.prefix
    noise_video = unpack_streams(calls["suffix_noise_reseed"][0], shapes)[0]
    assert torch.equal(noise_video[:, :, :PREFIX_T], base_video[:, :, :PREFIX_T])
    assert not torch.allclose(noise_video[:, :, PREFIX_T:], base_video[:, :, PREFIX_T:])
    receipt = case.evidence.plan["first_high_counterfactual"]
    assert [row["status"] for row in receipt["arms"]] == ["ok"] * 4
    assert receipt["pdd_heads"]["video_head_bank_count"] == 2
    assert receipt["high_entry_ledger"]["implied_noise_std"] == pytest.approx(1.0, abs=0.05)
    assert receipt["sample_sigmas"] == pytest.approx([case.sigma, 0.5])
    assert receipt["status"] == "captured"
    assert set(receipt["latent_boundary_trajectories"]) == {"pre_high", "production"}
    for row in receipt["arms"]:
        assert set(row["latent_boundary_trajectories"]) == {"upper45", "full"}
        assert len(row["latent_boundary_trajectories"]["full"]["pairwise_dx"]) == 5
    assert [e.kind for e in case.metrics.events] == ["partitioned_first_high_counterfactual"]
    _run(case)
    assert len(case.guider.calls) == 4


def test_disabled_request_and_later_calls_do_not_evaluate():
    case = _setup(request=None)
    _run(case)
    assert case.guider.calls == [] and "first_high_counterfactual" not in case.evidence.plan
    case = _setup()
    _run(case, call_index=1)
    assert case.guider.calls == []


def test_missing_full_stage_evidence_reports_once_without_evaluating():
    case = _setup()
    case.trace = SimpleNamespace(window_evidence=None)
    _run(case)
    _run(case)
    assert case.guider.calls == []
    assert [e.fields["status"] for e in case.metrics.events] == ["unavailable"]


def test_failed_arm_is_recorded_and_others_complete():
    case = _setup()
    original = case.guider.predict_noise

    def predict(x, timestep, options, seed):
        if PARTITIONED_STAGE_KEY not in options["transformer_options"]:
            raise RuntimeError("backend refused")
        return original(x, timestep, options, seed)

    case.guider.predict_noise = predict
    _run(case)
    rows = {row["arm"]: row for row in case.evidence.plan["first_high_counterfactual"]["arms"]}
    assert rows["native_high_attention"]["status"] == "failed"
    assert "backend refused" in rows["native_high_attention"]["error"]
    assert arm_tensor_name("native_high_attention") not in case.evidence.tensors
    assert all(rows[arm]["status"] == "ok" for arm in ("replay", "provider_prefix", "suffix_noise_reseed"))


def test_insufficient_device_headroom_skips_every_arm(monkeypatch):
    case = _setup()
    monkeypatch.setattr(counterfactual, "_free_device_bytes", lambda device: 0)
    _run(case)
    receipt = case.evidence.plan["first_high_counterfactual"]
    assert case.guider.calls == []
    assert receipt["status"] == "skipped_insufficient_headroom" and receipt["extra_h3_predictions"] == 0
    assert not any(name.startswith("first_high_counterfactual_") for name in case.evidence.tensors)


def test_node_is_off_by_default_and_publishes_selected_arms_on_a_clone():
    model = SimpleNamespace(model_options={})
    model.clone = lambda: SimpleNamespace(model_options=dict(model.model_options))
    spec = H3FirstHighCounterfactual.INPUT_TYPES()["required"]
    assert all(spec[arm][1]["default"] is False for arm in FIRST_HIGH_COUNTERFACTUAL_ARMS)
    (unchanged,) = H3FirstHighCounterfactual().patch(model, False, False, False, False, 9)
    assert unchanged is model and FIRST_HIGH_COUNTERFACTUAL_KEY not in model.model_options
    (patched,) = H3FirstHighCounterfactual().patch(model, True, False, True, True, 9)
    assert FIRST_HIGH_COUNTERFACTUAL_KEY not in model.model_options
    request = patched.model_options[FIRST_HIGH_COUNTERFACTUAL_KEY]
    assert request.arms == ("replay", "suffix_noise_reseed", "native_high_attention")
    with pytest.raises(ValueError):
        FirstHighCounterfactualRequest(("unknown",), 0)
