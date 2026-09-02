from __future__ import annotations

import argparse
from pathlib import Path


def require(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{path}: missing required contract fragments: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pinned H3 sibling-source contracts")
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--continuum", type=Path, required=True)
    parser.add_argument("--diffaid", type=Path, required=True)
    parser.add_argument("--refdelta", type=Path, required=True)
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
        args.comfy / "comfy/model_base.py",
        "class MiniMaxH3(BaseModel):",
        "return self.model_sampling.audio_scale",
        'payload["audio_scale"] = self.audio_scale()',
    )
    require(
        args.comfy / "comfy/model_sampling.py",
        "class ModelSamplingAV(ModelSamplingDiscreteFlow):",
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
    )
    require(
        args.spectrum / "comfyui_spectrum_h3/refinement_compat.py",
        'REFINEMENT_REQUEST_KEY = "h3_refinement"',
        'request.get("min_actual_prefix_steps")',
        'request.get("sigma_reference")',
    )
    require(
        args.continuum / "model_patch.py",
        '"api": CONTINUUM_INTEROP_API',
        '"chunk_index": int(chunk_index)',
        '"context_frames": int(context_frames)',
    )
    require(
        args.diffaid / "spectrum_h3_compat.py",
        '"h3_refinement"',
        '"sigma_reference"',
    )
    refdelta_text = "\n".join(path.read_text(encoding="utf-8") for path in args.refdelta.rglob("*.py"))
    for name in ("sa_solver", "sa_solver_pece", "seeds_2", "seeds_3", "er_sde"):
        if name not in refdelta_text:
            raise SystemExit(f"{args.refdelta}: missing sampler contract {name}")
    print("Pinned ComfyUI/H3 sibling contracts validated")


if __name__ == "__main__":
    main()
