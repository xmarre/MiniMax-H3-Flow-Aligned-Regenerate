from __future__ import annotations

import argparse
from pathlib import Path


def require(path: Path, *needles: str) -> None:
    text = path.read_text(encoding="utf-8")
    missing = [needle for needle in needles if needle not in text]
    if missing:
        raise SystemExit(f"{path}: missing target-sparse contract fragments: {missing}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate target-sparse H3 integration contracts")
    parser.add_argument("--comfy", type=Path, required=True)
    parser.add_argument("--spectrum", type=Path, required=True)
    parser.add_argument("--continuum", type=Path, required=True)
    parser.add_argument("--diffaid", type=Path, required=True)
    parser.add_argument("--vdn", type=Path, required=True)
    args = parser.parse_args()

    require(
        args.comfy / "comfy/ldm/minimax/model.py",
        'segments.append(("audio"',
        'segments.append(("video"',
        'if ("double_block", i) in blocks_replace:',
        '"mod_segments": mod_segments',
        '"rope_freqs": rope_freqs',
        '"layout": layout',
        '})["img"]',
        "def mask_row_values(mask, latent_t, lat_h, lat_w):",
        "amax(dim=(2, 4))",
    )
    require(
        args.spectrum / "comfyui_spectrum_h3/minimax_h3.py",
        'existing = dit_replacements.get(("double_block", last_index))',
        "output = existing(args, replacement_context) if existing is not None",
        "\"final MiniMax H3 block replacement did not return {'img': tensor}\"",
        "target = hidden[aa:vb].unsqueeze(0)",
    )
    require(
        args.continuum / "masked_continuation.py",
        "def apply_native_masked_continuation(",
        "video[:, :, :video_steps].copy_(video_context)",
        '"continuation video geometry does not match the new target latent"',
        "video_mask = torch.ones(",
        "video_mask[:, :, :video_steps] = 0",
        'output["noise_mask"] = comfy.nested_tensor.NestedTensor((video_mask, audio_mask))',
    )
    require(
        args.diffaid / "nodes.py",
        "def _minimax_h3_language_ranges(",
        'args.get("mod_segments", None)',
        'new_args["img"] = new_img',
    )
    require(
        args.vdn / "vdn_h3/hybrid.py",
        'VDN_EXTERNAL_SEQUENCE_KEY = "vdn_h3_external_sequence_v1"',
        'VDN_EXTERNAL_SEQUENCE_MODE = "dense_gate_no_linear"',
        "def _external_reduced_sequence_active(",
        'contract.get("full_sequence_rows", -1)',
        'contract.get("reduced_sequence_rows", -1)',
        "window_active = not layout.full_cover and not external_reduced",
        'vdn_forward._vdn_external_sequence_api = VDN_EXTERNAL_SEQUENCE_API_VERSION',
    )


if __name__ == "__main__":
    main()
