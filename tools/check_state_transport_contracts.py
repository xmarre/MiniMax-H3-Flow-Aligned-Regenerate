from __future__ import annotations

import argparse
from pathlib import Path


def require(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{path}: missing state-transport contract fragments: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate pinned ComfyUI state-transport assumptions")
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    args = parser.parse_args()

    require(
        args.comfy / "comfy/model_sampling.py",
        "class CONST:",
        "return model_input - model_output * sigma",
        's = getattr(self, "noise_scale", 1.0)',
        "return sigma * (s * noise) + (1.0 - sigma) * latent_image",
        "return latent / (1.0 - sigma)",
    )
    require(
        args.comfy / "comfy/samplers.py",
        "class KSamplerX0Inpaint:",
        "latent_mask = 1. - denoise_mask",
        "noise=self.noise, latent_image=self.latent_image, denoise_mask=denoise_mask",
        "out = out * denoise_mask + self.latent_image * latent_mask",
        "self.model_options = comfy.model_patcher.create_model_options_clone(self.model_options)",
        "extra_model_options = comfy.model_patcher.create_model_options_clone(self.model_options)",
        "WrappersMP.OUTER_SAMPLE",
        "WrappersMP.PREDICT_NOISE",
        "WrappersMP.SAMPLER_SAMPLE",
    )
    require(
        args.comfy / "comfy/model_patcher.py",
        "def create_model_options_clone(orig_model_options: dict):",
        "return comfy.patcher_extension.copy_nested_dicts(orig_model_options)",
    )
    require(
        args.comfy / "comfy/patcher_extension.py",
        "def copy_nested_dicts(input_dict: dict):",
        "new_dict[key] = copy_nested_dicts(value)",
        "new_dict[key] = value.copy()",
        "self.wrappers = wrappers.copy()",
        "return self.wrappers[self.idx](self, *args, **kwargs)",
        "for w in wrappers.get(wrapper_type, {}).values():",
    )
    require(
        args.spectrum / "comfyui_spectrum_h3/sampling.py",
        "def copy_model_options_with_step(",
        'transformer_options[ACTUAL_KEY] = bool(decision["actual"])',
        "return executor(x, timestep, patched, seed)",
    )
    print("Pinned state-transport sampler, mask, wrapper, and model-options contracts validated")


if __name__ == "__main__":
    main()
