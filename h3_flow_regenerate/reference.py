from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from .geometry import normalize_target_geometry, resize_video, validate_video


@dataclass(frozen=True, slots=True)
class ReferenceBudgetReport:
    qwen_rows: int
    direct_video_rows_before: int
    direct_video_rows_after: int
    direct_audio_rows: int
    changed_direct_refs: int
    mode: str


def _conditioning_rows(conditioning: list[Any]) -> int:
    rows = 0
    for entry in conditioning:
        if isinstance(entry, (list, tuple)) and entry and torch.is_tensor(entry[0]):
            rows += int(entry[0].shape[-2])
    return rows


def _fit_reference(video: torch.Tensor, max_rows: int, mode: str) -> torch.Tensor:
    validate_video(video)
    temporal = int(video.shape[2])
    current_rows = temporal * (int(video.shape[-2]) // 2) * (int(video.shape[-1]) // 2)
    if current_rows <= max_rows:
        return video
    if max_rows < temporal:
        raise ValueError("direct-reference row budget is smaller than one spatial patch per frame")
    scale = math.sqrt(max_rows / current_rows)
    target_h, target_w = normalize_target_geometry(
        source_h=int(video.shape[-2]),
        source_w=int(video.shape[-1]),
        scale=scale,
        policy="nearest",
    )
    while temporal * (target_h // 2) * (target_w // 2) > max_rows and min(target_h, target_w) > 2:
        if target_h >= target_w:
            target_h -= 2
        else:
            target_w -= 2
    return resize_video(video, target_h, target_w, mode=mode)


def apply_reference_budget(
    conditioning: list[Any],
    *,
    mode: str = "native",
    max_direct_video_rows: int = 2048,
    resize_mode: str = "area",
) -> tuple[list[Any], ReferenceBudgetReport]:
    if mode not in {"native", "diagnostic", "decoupled_direct_experimental"}:
        raise ValueError(f"unsupported reference budget mode {mode!r}")
    if max_direct_video_rows < 1:
        raise ValueError("max_direct_video_rows must be positive")
    if mode == "decoupled_direct_experimental":
        result = []
        for entry in conditioning:
            if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
                result.append(entry)
                continue
            metadata = dict(entry[1])
            metadata["minimax_refs"] = [dict(ref) for ref in metadata.get("minimax_refs") or []]
            result.append([entry[0], metadata, *entry[2:]])
    else:
        result = conditioning
    before = after = audio = changed = 0
    for entry in result:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2 or not isinstance(entry[1], dict):
            continue
        refs = entry[1].get("minimax_refs") or []
        for ref in refs:
            latent = ref.get("latent") if isinstance(ref, dict) else None
            if torch.is_tensor(latent):
                rows_before = int(latent.shape[2]) * (int(latent.shape[-2]) // 2) * (int(latent.shape[-1]) // 2)
                before += rows_before
                if mode == "decoupled_direct_experimental":
                    fitted = _fit_reference(latent, max_direct_video_rows, resize_mode)
                    ref["latent"] = fitted
                    changed += int(fitted.shape != latent.shape)
                    latent = fitted
                after += int(latent.shape[2]) * (int(latent.shape[-2]) // 2) * (int(latent.shape[-1]) // 2)
            audio_latent = ref.get("audio_latent") if isinstance(ref, dict) else None
            if torch.is_tensor(audio_latent):
                audio += int(audio_latent.shape[-1]) * 2
    return result, ReferenceBudgetReport(
        qwen_rows=_conditioning_rows(conditioning),
        direct_video_rows_before=before,
        direct_video_rows_after=after,
        direct_audio_rows=audio,
        changed_direct_refs=changed,
        mode=mode,
    )
