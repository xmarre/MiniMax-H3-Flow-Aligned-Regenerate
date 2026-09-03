from __future__ import annotations

import argparse
import ast
from pathlib import Path


def require(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{path}: missing required contract fragments: {missing}")


def require_symbols(path: Path, *, functions: tuple[str, ...] = (), classes: tuple[str, ...] = ()) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found_functions = {
        node.name for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    found_classes = {node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)}
    missing_functions = sorted(set(functions) - found_functions)
    missing_classes = sorted(set(classes) - found_classes)
    if missing_functions or missing_classes:
        raise SystemExit(f"{path}: missing structural symbols functions={missing_functions} classes={missing_classes}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pinned H3 sibling-source contracts")
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--continuum", type=Path, required=True)
    parser.add_argument("--diffaid", type=Path, required=True)
    parser.add_argument("--refdelta", type=Path, required=True)
    parser.add_argument("--untwist", type=Path, required=True)
    parser.add_argument("--upscaler", type=Path, required=True)
    args = parser.parse_args()

    require(
        args.comfy / "comfy/ldm/minimax/model.py",
        "latents_dim=24",
        "audio_latents_dim=32",
        "patch_size=(1, 2, 2)",
        "sigma_shift_video=12.0",
        "sigma_shift_audio=3.0",
        'segments = [("text", text_len)]',
        'segments.append(("audio"',
        'segments.append(("video"',
        "time_shift_sigma(sigma_v, shift_v, shift_a)",
        "mask=None, skip_reshape=True",
    )
    require(
        args.comfy / "comfy_extras/nodes_minimax_h3.py",
        "CANVAS_MULTIPLE = 32",
        "BASE_SHORT_EDGE = 768",
        "MAX_PIXELS = 768 * 1344",
        "def adapt_canvas(width, height):",
        '"""768-short-edge canvas with 768*1344 area cap, per-axis round to 32."""',
    )
    require(
        args.comfy / "comfy/model_base.py",
        "class MiniMaxH3(BaseModel):",
        "return self.model_sampling.audio_scale",
        'payload["audio_scale"] = self.audio_scale()',
    )
    require(
        args.comfy / "comfy/model_sampling.py",
        "class ModelSamplingAV(ModelSamplingDiscreteFlow):",
        "return model_input - model_output * sigma",
        "return self.shift / self.audio_shift",
        "return sigma * (s * noise) + (1.0 - sigma) * latent_image",
        "return latent / (1.0 - sigma)",
    )
    require(
        args.comfy / "comfy/samplers.py",
        "latent_shapes=latent_shapes",
        "unpack_latents(output, latent_shapes)",
        "WrappersMP.OUTER_SAMPLE",
        "WrappersMP.PREDICT_NOISE",
        "#Returns denoised",
        "inverse_noise_scaling(sigmas[-1], samples)",
        "preprocess_conds_hooks(self.conds)",
        "filter_registered_hooks_on_conds(self.conds, self.model_options)",
        "self.conds = process_conds(",
        "noise = noise.to(device=device, dtype=torch.float32)",
        "latent_image = latent_image.to(device=device, dtype=torch.float32)",
    )
    require(
        args.comfy / "comfy/sampler_helpers.py",
        'temp["uuid"] = uuid.uuid4()',
    )
    require(
        args.comfy / "execution.py",
        'elif hasattr(class_def, "IS_CHANGED")',
        'node["is_changed"] = float("NaN")',
    )
    require(
        args.comfy / "comfy_execution/caching.py",
        "signature = [class_type, await self.is_changed_cache.get(node_id)]",
    )
    require(
        args.comfy / "comfy/model_patcher.py",
        "def add_callback_with_key(",
        "def remove_callbacks_with_key(",
        "self.get_all_callbacks(CallbacksMP.ON_CLONE)",
        "callback(self, n)",
    )
    require(
        args.comfy / "comfy/sampler_helpers.py",
        'merge_nested_dicts(model_options["transformer_options"].setdefault("wrappers", {}), model.wrappers',
    )
    require(
        args.comfy / "comfy/patcher_extension.py",
        "for w in wrappers.get(wrapper_type, {}).values():",
        "return self.wrappers[self.idx](self, *args, **kwargs)",
    )
    require(
        args.spectrum / "comfyui_spectrum_h3/refinement_compat.py",
        'REFINEMENT_REQUEST_KEY = "h3_refinement"',
        'request.get("min_actual_prefix_steps")',
        'request.get("sigma_reference")',
    )
    require(
        args.spectrum / "comfyui_spectrum_h3/sampling.py",
        'BINDING_KEY = "spectrum_h3_binding"',
        'ACTUAL_KEY = "spectrum_h3_actual"',
        'SOLVER_PHASE_KEY = "spectrum_h3_solver_phase"',
        'OUTER_STEP_ID_KEY = "spectrum_h3_outer_step_id"',
        "class SpectrumH3Binding:",
        "def copy_model_options_with_step(",
        'transformer_options[ACTUAL_KEY] = bool(decision["actual"])',
        "return executor(x, timestep, patched, seed)",
    )
    require(
        args.spectrum / "comfyui_spectrum_h3/runtime.py",
        "class _StepState:",
        "mode: str",
        "def active_run_id(",
        "def active_step_id(",
        "def active_solver_phase(",
        "def active_policy_step_id(",
        "def last_completed_mode(",
        "def last_completed_step_id(",
    )
    require(
        args.continuum / "model_patch.py",
        '"api": CONTINUUM_INTEROP_API',
        '"chunk_index": int(chunk_index)',
        '"context_frames": int(context_frames)',
    )
    require_symbols(
        args.continuum / "v3/driving_nodes.py",
        functions=("_refine_state_output", "_fresh_refine_model"),
        classes=("H3ContinuumSamplerV34",),
    )
    require(
        args.continuum / "v3/driving_nodes.py",
        '"H3_CONTINUUM_REFINE_STATE"',
        '"emit_refine_conditioning"',
        '"refine_state"',
    )
    require(
        args.continuum / "v2/sampling.py",
        "def _record_chunk_refine_state(",
        '"model": model',
        '"positive": conditioning',
        '"noise_mask": noise_mask',
    )
    require(
        args.continuum / "v3/driving_nodes.py",
        "def _attach_refine_masks(",
        'record.get("noise_mask")',
        'video_item["noise_mask"]',
        'audio_item["noise_mask"]',
        '"model": _fresh_refine_model',
        '"positive": positive',
    )
    require(
        args.diffaid / "spectrum_h3_compat.py",
        '"h3_refinement"',
        '"sigma_reference"',
    )
    require(
        args.untwist / "flux_untwist/spectrum_h3.py",
        "VISUAL_PATCH_SCHEMA_VERSION = 1",
        'VISUAL_PATCH_ARCHITECTURE = "minimax_h3"',
        'VISUAL_PATCH_RUNTIME_KEY = "spectrum_h3_visual_reference_patch_runtime"',
        '"schedule_progress"',
        '"active"',
    )
    require_symbols(
        args.upscaler / "nodes/minimax_h3_refine.py",
        functions=(
            "_resolve_refine_contract",
            "_model_with_refinement_contract",
            "build_clean_h3_upscale",
            "run_h3_refinement",
        ),
        classes=("MinimaxH3LatentUpscaler3DRefine",),
    )
    require(
        args.upscaler / "nodes/minimax_h3_refine.py",
        '"H3_CONTINUUM_REFINE_STATE"',
        "transformer_options[H3_REFINEMENT_REQUEST_KEY]",
        "resize_h3_target_conditioning",
        'noise_mask = latent.get("noise_mask")',
        "denoise_mask=noise_mask",
    )
    refdelta_text = "\n".join(path.read_text(encoding="utf-8") for path in args.refdelta.rglob("*.py"))
    for name in ("sa_solver", "sa_solver_pece", "seeds_2", "seeds_3", "er_sde"):
        if name not in refdelta_text:
            raise SystemExit(f"{args.refdelta}: missing sampler contract {name}")
    print("Pinned ComfyUI/H3 sibling contracts validated")


if __name__ == "__main__":
    main()
