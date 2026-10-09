"""Read-only static-background ROI audit for native H3 decoded replay stages.

Metrics compare frames *within* each native-resolution stage. Absolute
sharpness across different decoded grids is not directly comparable.
Coordinates are fractional XYXY canvas coordinates. No production edits.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping

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


def _common_grid_luma(luma: torch.Tensor, long_side: int = 704) -> torch.Tensor:
    """Antialiased downsample for fair 704-vs-992 stage texture comparison.

    Never upsample: hallucinating details in the 704px native carrier would
    bias the cross-stage gradient metric.
    """
    h, w = map(int, luma.shape)
    if max(h, w) <= long_side:
        return luma
    scale = long_side / max(h, w)
    shape = (max(8, round(h*scale)), max(8, round(w*scale)))
    return F.interpolate(luma[None,None].float(), size=shape, mode="area")[0,0]


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


def _apparent_scale(regions: Mapping[str,dict], width: int, height: int) -> dict:
    # Fit independent apparent X/Y expansions from spatially separated,
    # temporally static room landmarks; not a physical camera calibration.
    eligible = [row for row in regions.values()
                if row.get("verified_pre_join_static")
                and row["displacement"].get("status")=="measured"
                and row["displacement"]["peak_dominance"]>=3.]
    if len(eligible)<2:
        return {"status":"insufficient_landmarks","supported_regions":len(eligible)}

    def fit(axis, disp, denom):
        pts = [(row[axis],row["displacement"][disp]/denom) for row in eligible]
        mx = sum(x for x,y in pts)/len(pts)
        my = sum(y for x,y in pts)/len(pts)
        variance = sum((x-mx)**2 for x,y in pts)
        if variance<.015:
            return {"status":"insufficient_spatial_spread"}
        slope = sum((x-mx)*(y-my) for x,y in pts)/variance
        residual = math.sqrt(sum((y-my-slope*(x-mx))**2 for x,y in pts)/len(pts))
        return {"status":"estimated","scale_percent":slope*100,
                "spatial_residual_fraction":residual}

    horizontal=fit("center_x","dx_px",width)
    vertical=fit("center_y","dy_px",height)
    return {"status":horizontal["status"],
            "scale_change_percent":horizontal.get("scale_percent"),
            "spatial_residual_fraction":horizontal.get("spatial_residual_fraction"),
            "vertical":vertical,"supported_regions":len(eligible),
            "not_camera_ground_truth":True}


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
    common_lumas = [_common_grid_luma(frame) for frame in lumas]
    rows = {}
    for name,rect in rois.items():
        prev,base,post = (_crop(f,rect) for f in lumas)
        e0,e1,e2 = (_gradient_energy(f) for f in (prev,base,post))
        post_ratio = e2/max(e1,1e-12)
        common_prev, common_base, common_post = (_crop(f,rect) for f in common_lumas)
        ce0,ce1,ce2 = (_gradient_energy(f) for f in (common_prev,common_base,common_post))
        disp = _phase_displacement(base,post)
        prior_shift = _phase_displacement(prev,base)
        pretrend_ratio = e1/max(e0,1e-12)
        verified_pre = bool(
            prior_shift.get("status")=="measured"
            and max(abs(prior_shift["dx_px"]),abs(prior_shift["dy_px"]))<=.75
            and prior_shift["peak_dominance"]>=3
            and .70<=pretrend_ratio<=1.35
        )
        if disp.get("status")=="measured" and (
            abs(disp["dx_px"])>12 or abs(disp["dy_px"])>12 or not .25<=post_ratio<=4.
        ):
            disp["status"]="unreliable_correspondence_or_texture_change"
        rows[name] = {
            "bounds_xyxy":list(rect),"center_x":(rect[0]+rect[2])/2,
            "center_y":(rect[1]+rect[3])/2,
            "native_roi_hw":list(base.shape),
            "sobel_energy_by_frame":{str(start):e0,str(before):e1,str(after):e2},
            "sharpness_post_over_pre":post_ratio,
            "sharpness_pretrend_ratio":pretrend_ratio,
            "sharpness_excess_over_pretrend":post_ratio/pretrend_ratio,
            "common_grid_hw":list(common_base.shape),
            "common_grid_sobel_energy_by_frame":{str(start):ce0,str(before):ce1,str(after):ce2},
            "common_grid_sharpness_post_over_pre":ce2/max(ce1,1e-12),
            "common_grid_sharpness_excess_over_pretrend":(ce2/max(ce1,1e-12))/(ce1/max(ce0,1e-12)),
            "verified_pre_join_static":verified_pre,
            "mean_luma_delta":float((post-base).mean().item()),
            "displacement":disp,
            "preceding_displacement":prior_shift,
        }
    return {"policy":POLICY,"status":"measured","join_frame":join_frame,
            "common_grid_max_side_px":704,
            "pre_frame":before,"post_frame":after,"pretrend_frame":start,
            "native_grid_hw":list(frames.shape[1:3]),
            "intra_stage_comparisons_only":True,"absolute_sharpness_across_grids_comparable":False,
            "production_modified":False,"extra_vae_calls":0,
            "regions":rows,"background_scale":_apparent_scale(rows,int(frames.shape[2]),int(frames.shape[1]))}


def compare_same_frame_stage_rois(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    frame_labels: list[int],
    *,
    join_frame: int,
    rois: Mapping[str,tuple[float,float,float,float]],
) -> dict:
    """Paired pixel-time texture comparison of two native decoder stages.

    Antialias to a common 704px-or-smaller grid *before* computing gradient
    energy. Compare relative stage energy at f174 versus f176..181; this
    cancels stable differences in representation and decoder characteristics.
    """
    if reference.ndim != 4 or candidate.ndim != 4 or reference.shape[0] != candidate.shape[0]:
        raise ValueError("stage comparison needs equally indexed decoded IMAGE videos")
    if len(frame_labels) != len(reference):
        raise ValueError("stage comparison frame labels do not match")
    if reference.shape[-1] < 3 or candidate.shape[-1] < 3:
        raise ValueError("stage comparison requires RGB")
    indexed = {frame:i for i,frame in enumerate(frame_labels)}
    baseline = join_frame-1
    targets = [frame for frame in range(join_frame+1,join_frame+7) if frame in indexed]
    if baseline not in indexed or not targets:
        return {"policy":POLICY,"status":"insufficient_same_frame_timing_window"}
    heights = [int(reference.shape[1]), int(candidate.shape[1])]
    widths = [int(reference.shape[2]), int(candidate.shape[2])]
    ratio_h = min(h/w for h,w in zip(heights,widths))
    common_w = min(704, *widths)
    common_h = max(8, round(common_w*ratio_h))
    if common_h > 704:
        common_w = max(8, round(common_w*704/common_h))
        common_h = 704
    common_hw = (common_h, common_w)

    def projected(frame):
        v = frame[...,:3].detach().to(device="cpu",dtype=torch.float32)
        luma = v @ v.new_tensor(_LUMA)
        if tuple(luma.shape) != common_hw:
            luma = F.interpolate(luma[None,None],size=common_hw,mode="area")[0,0]
        return luma

    comparison = {}
    for name, rect in rois.items():
        series = {}
        for label in [baseline,*targets]:
            idx = indexed[label]
            a = _crop(projected(reference[idx]),rect)
            b = _crop(projected(candidate[idx]),rect)
            e0,e1 = _gradient_energy(a),_gradient_energy(b)
            series[label] = {
                "reference_common_sobel":e0,
                "candidate_common_sobel":e1,
                "candidate_over_reference":e1/max(e0,1e-12),
                "same_time_displacement":_phase_displacement(a,b),
            }
        anchor = series[baseline]["candidate_over_reference"]
        for frame, row in series.items():
            row["relative_to_same_frame_prefix_energy_ratio"] = (
                row["candidate_over_reference"]/max(anchor,1e-12)
            )
        comparison[name] = {
            "fractional_bounds_xyxy":list(rect),
            "same_frame_measurements":{str(frame):row for frame,row in series.items()},
            "median_post_relative_sharpness":float(torch.tensor(
                [series[f]["relative_to_same_frame_prefix_energy_ratio"] for f in targets]
            ).median().item()),
        }
    return {
        "policy":"h3_stage_static_background_same_frame_v1",
        "status":"measured",
        "join_frame":join_frame,
        "baseline_frame":baseline,
        "post_frames":targets,
        "common_canvas_hw":list(common_hw),
        "candidate_vs_reference_normalized_on_prefix":True,
        "extra_vae_calls":0,
        "production_modified":False,
        "regions":comparison,
    }
