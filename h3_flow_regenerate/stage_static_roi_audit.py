"""Read-only static-background ROI audit for native H3 decoded replay stages.

Metrics compare frames *within* each native-resolution stage. Absolute
sharpness across different decoded grids is not directly comparable.
Coordinates are fractional XYXY canvas coordinates. No production edits.
"""
from __future__ import annotations

import json
import math
from typing import Mapping

import torch
import torch.nn.functional as F

POLICY = "h3_stage_static_background_roi_v1"
ROOM_01784 = {
    "bookshelf": (0.005, 0.423, 0.131, 0.595),
    "framed_picture": (0.146, 0.081, 0.283, 0.272),
    "right_curtain": (0.827, 0.040, 0.988, 0.358),
    "white_wall": (0.290, 0.105, 0.397, 0.305),
}
PROFILES = ("off", "01784_room", "custom")
_LUMA = (0.2126, 0.7152, 0.0722)


def parse_static_rois(profile: str, custom_json: str = "") -> dict[str, tuple[float, float, float, float]]:
    if profile not in PROFILES:
        raise ValueError("unknown static ROI profile")
    if profile == "off":
        return {}
    if profile == "01784_room":
        return dict(ROOM_01784)
    if not isinstance(custom_json, str) or len(custom_json) > 8192:
        raise ValueError("static ROI JSON must be a bounded string")
    try:
        parsed = json.loads(custom_json)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("static ROI JSON must map names to [x0,y0,x1,y1]") from exc
    if not isinstance(parsed, dict) or not 2 <= len(parsed) <= 12:
        raise ValueError("static ROI JSON must define 2 to 12 rectangles")
    rois = {}
    for name, raw in parsed.items():
        if not isinstance(name, str) or not name or len(name) > 40 or not isinstance(raw, list) or len(raw) != 4:
            raise ValueError("invalid static ROI name or coordinate count")
        if any(type(v) not in (float, int) or not math.isfinite(v) for v in raw):
            raise ValueError("static ROI coordinates must be finite numbers")
        x0, y0, x1, y1 = (float(v) for v in raw)
        if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1 and x1-x0 >= .03 and y1-y0 >= .03):
            raise ValueError("static ROI must be inside the canvas and at least 3% wide and high")
        rois[name] = (x0, y0, x1, y1)
    return rois


def _crop(frame: torch.Tensor, rect: tuple[float, float, float, float]) -> torch.Tensor:
    h, w = frame.shape
    x0, y0, x1, y1 = rect
    ax, bx, ay, by = round(x0*w), round(x1*w), round(y0*h), round(y1*h)
    if min(bx-ax, by-ay) < 8:
        raise ValueError("static ROI requires at least 8 native decoded pixels per side")
    return frame[ay:by, ax:bx]


def _gradient_energy(luma: torch.Tensor) -> float:
    v = luma[None, None].float()
    wx = v.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]]).view(1,1,3,3)/8
    return float((F.conv2d(v,wx).square()+F.conv2d(v,wx.transpose(-1,-2)).square()).mean().item())


def _subpixel_peak(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> float:
    den = float((a-2*b+c).item())
    return max(-.5,min(.5,.5*float((a-c).item())/den)) if abs(den)>1e-12 else 0.


def _phase_displacement(ref: torch.Tensor, candidate: torch.Tensor) -> dict:
    """Candidate content shift vs reference, in native pixels; diagnostic only."""
    if ref.shape != candidate.shape:
        raise ValueError("ROI correspondence requires identical decoded geometry")
    h,w = ref.shape
    if min(h,w)<8:
        return {"status":"insufficient_pixels"}
    window = torch.hann_window(h,periodic=False)[:,None]*torch.hann_window(w,periodic=False)[None,:]
    a = (ref.float()-ref.float().mean())*window
    b = (candidate.float()-candidate.float().mean())*window
    if float(a.square().mean())<1e-7 or float(b.square().mean())<1e-7:
        return {"status":"low_texture"}
    cross = torch.fft.fft2(a)*torch.fft.fft2(b).conj()
    cross /= cross.abs().clamp_min(1e-10)
    corr = torch.fft.ifft2(cross).real
    yy,xx = divmod(int(corr.argmax().item()),w)
    a_y,b_y,c_y = corr[(yy-1)%h,xx],corr[yy,xx],corr[(yy+1)%h,xx]
    a_x,b_x,c_x = corr[yy,(xx-1)%w],corr[yy,xx],corr[yy,(xx+1)%w]
    dy = float(yy if yy<=h//2 else yy-h)+_subpixel_peak(a_y,b_y,c_y)
    dx = float(xx if xx<=w//2 else xx-w)+_subpixel_peak(a_x,b_x,c_x)
    return {"status":"measured","dx_px":-dx,"dy_px":-dy,
            "peak_dominance":float(corr.max().item()/corr.abs().mean().clamp_min(1e-8).item())}


def _apparent_scale(regions: Mapping[str,dict], width: int) -> dict:
    pts = [(row["center_x"], row["displacement"]["dx_px"]/width)
           for row in regions.values()
           if row["displacement"].get("status")=="measured"
           and row["displacement"]["peak_dominance"]>=3.]
    if len(pts)<2:
        return {"status":"insufficient_landmarks","supported_regions":len(pts)}
    mx = sum(x for x,y in pts)/len(pts)
    my = sum(y for x,y in pts)/len(pts)
    var = sum((x-mx)**2 for x,y in pts)
    if var<.05:
        return {"status":"insufficient_spatial_spread","supported_regions":len(pts)}
    slope = sum((x-mx)*(y-my) for x,y in pts)/var
    residual = math.sqrt(sum((y-my-slope*(x-mx))**2 for x,y in pts)/len(pts))
    return {"status":"estimated","scale_change_percent":100*slope,
            "spatial_residual_fraction":residual,
            "supported_regions":len(pts),"not_camera_ground_truth":True}


def measure_stage_static_rois(
    frames: torch.Tensor, frame_labels: list[int], *,
    join_frame: int, rois: Mapping[str,tuple[float,float,float,float]], target_after: int=5,
) -> dict:
    """Stage-local f(join-1)→f(join+5) texture and apparent geometry.

    Uses same RGB coordinates within each stage; no cross-resolution absolute
    sharpness comparisons and no input tensor changes.
    """
    if not torch.is_tensor(frames) or frames.ndim!=4 or frames.shape[-1]<3:
        raise ValueError("expected decoded IMAGE video [T,H,W,C]")
    if len(frame_labels)!=len(frames) or len(set(frame_labels))!=len(frame_labels):
        raise ValueError("frame labels do not match decoded frames")
    if type(join_frame) is not int or type(target_after) is not int or not 1<=target_after<=24:
        raise ValueError("invalid join/target frame")
    indices = {label:i for i,label in enumerate(frame_labels)}
    before,after = join_frame-1,join_frame+target_after
    preceding = sorted(label for label in indices if label<before)
    if before not in indices or after not in indices or not preceding:
        return {"policy":POLICY,"status":"insufficient_timing_window",
                "requested_pair":[before,after],"present_frame_range":[min(frame_labels),max(frame_labels)]}
    start = preceding[max(0,len(preceding)-3)]
    samples = [frames[indices[i]] for i in (start,before,after)]
    if not all(bool(torch.isfinite(f).all().item()) for f in samples):
        raise ValueError("nonfinite decoded ROI pixels")
    lumas = []
    for f in samples:
        rgb = f[..., :3].detach().to(device="cpu",dtype=torch.float32)
        lumas.append(rgb@rgb.new_tensor(_LUMA))
    rows = {}
    for name,rect in rois.items():
        prev,base,post = (_crop(f,rect) for f in lumas)
        e0,e1,e2 = (_gradient_energy(f) for f in (prev,base,post))
        post_ratio = e2/max(e1,1e-12)
        disp = _phase_displacement(base,post)
        if disp.get("status")=="measured" and (
            abs(disp["dx_px"])>12 or abs(disp["dy_px"])>12 or not .25<=post_ratio<=4.
        ):
            disp["status"]="unreliable_correspondence_or_texture_change"
        rows[name] = {
            "bounds_xyxy":list(rect),"center_x":(rect[0]+rect[2])/2,
            "native_roi_hw":list(base.shape),
            "sobel_energy_by_frame":{str(start):e0,str(before):e1,str(after):e2},
            "sharpness_post_over_pre":post_ratio,
            "sharpness_pretrend_ratio":e1/max(e0,1e-12),
            "sharpness_excess_over_pretrend":post_ratio/(e1/max(e0,1e-12)),
            "mean_luma_delta":float((post-base).mean().item()),
            "displacement":disp,
            "preceding_displacement":_phase_displacement(prev,base),
        }
    return {"policy":POLICY,"status":"measured","join_frame":join_frame,
            "pre_frame":before,"post_frame":after,"pretrend_frame":start,
            "native_grid_hw":list(frames.shape[1:3]),
            "intra_stage_comparisons_only":True,"absolute_sharpness_across_grids_comparable":False,
            "production_modified":False,"extra_vae_calls":0,
            "regions":rows,"background_scale":_apparent_scale(rows,int(frames.shape[2]))}
