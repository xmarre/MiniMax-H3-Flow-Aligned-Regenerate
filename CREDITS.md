# Research credits and implementation provenance

MiniMax-H3-Flow-Aligned-Regenerate is an independent, training-free research implementation. It does **not** reproduce MiniMax's private H3-Regenerate-2K implementation or its unreleased sparse-attention topology.

The algorithms, design hypotheses, failure models, and integration contracts in this repository were developed after studying the papers and source implementations below. Credit is split by what actually transferred into this project so that an experimental node is not presented as an original algorithm when it is an H3-specific adaptation of published work.

These references are attribution, not relicensing. The research repositories listed here are not bundled as vendored dependencies by this project; their own licenses and copyright notices continue to apply. Where this project interoperates with a sibling ComfyUI repository at runtime, that relationship is listed separately.

## Node-by-node research lineage

| Public node / feature | Primary credited work | What transfers here | Important boundary |
|---|---|---|---|
| **MiniMax H3 Flow Trajectory** / **Trajectory Capture** | MiniMax-H3, ComfyUI, Spectrum, SA-Solver | H3-native AV trajectory coordinates, exact/forecast provenance, and predictor/corrector phase-aware capture | The transactional trajectory schema and H3 binding are this repository's implementation; Spectrum forecasts are never silently relabeled as exact anchors. |
| **MiniMax H3 Flow-Aligned Regenerate** / **Flow-Aligned Refine State** | **HiFlow**; **FrescoDiffusion** | Time-matched low-resolution trajectory guidance; low-frequency direction alignment; experimental adjacent-velocity acceleration; use of a low-resolution video trajectory as a global prior | HiFlow is an image method and FrescoDiffusion uses tiled prior-regularized fusion. This project adapts the ideas to H3 predicted-clean AV sampling rather than copying either pipeline. |
| `direction+temporal` guidance | **TokenFlow**, **FRESCO**, **MoVideo**, **Upscale-A-Video**, **LatentWarp** | Inter-frame correspondence, latent/feature propagation, confidence/visibility gating, and the need to avoid transporting content through ambiguous or occluded regions | No external optical-flow model is loaded. H3 clean-state latents are locally matched with strict similarity, uniqueness, and reverse-cycle gates. |
| **MiniMax H3 Progressive Handoff** | **RALU**, **Self-Cascade Diffusion**, **CineScale**, **Just-in-Time** | Coarse-to-fine sampling, explicit spatial transition semantics, re-noising after scale change, and deferring expensive high-resolution computation | The H3 handoff uses an exact probe plus a rectified-flow conditional state and fresh sampler/Spectrum lifetimes. It is not RALU NT-Matching, JiT DMF, or MiniMax H3-Regenerate-2K. |
| **MiniMax H3 Progressive Target-Sparse Continuum [Experimental]** | **Just-in-Time**; **RALU**; MiniMax-H3 / ComfyUI native mask and packed-layout contracts | Deferring dense target-grid work with target-grid anchor tokens and a hidden-space lifter while retaining every exact protected video row and every non-video row | Independently written H3 adaptation. It does not implement JiT SAG-ODE/DMF/deterministic micro-flow, RALU NT-Matching, OpenVDN's trained hybrid path, or MiniMax's private sparse attention. Structural validity does not establish decoded-media quality or speedup. |
| **Progressive Handoff (Target Input)** `learned_3d` | Companion **Comfyui_Minimax_h3_latent_Upscaler**; inherited **LBH-123-AI**, **ComfyUi_NNLatentUpscale**, and **LTX spatial-upscaler** lineage | A single learned clean-video spatial transform at the exact handoff boundary | Flow only consumes the companion provider API. The learned model/architecture lineage and checkpoints remain owned and credited by their upstream projects; audio, target noise, sigma law, masks, and H3 NFEs are unchanged. |
| **MiniMax H3 Refine Target Geometry** | Companion **Comfyui_Minimax_h3_latent_Upscaler** | Mirrors the companion node's scale-by-multiplier geometry contract for schedule metadata | No latent upscale or sampling occurs in this helper. |
| **MiniMax H3 Resolution-Aware Sigmas** | **simple diffusion**; **Scaling Rectified Flow Transformers for High-Resolution Image Synthesis** | Resolution/SNR motivation and the fractional-linear resolution-dependent flow-time map | The SD3 constant-observation derivation is only used as an experimental relative correction composed with H3's native shift; it is not claimed to be an optimal H3 schedule. |
| **MiniMax H3 Attention Lab** | **OpenVDN**, **FreeSwim**, **HRDiT**, **ResDiT**; MiniMax-H3's public sparse-attention statement | Output-neutral retention diagnostics; OpenVDN-inspired 5-frame chunk-local temporal reference topology with previous/current/next chunks and boundary anchors; native-scale spatial/local-global research | H3 uses one packed multimodal stream. The VDN topology path is an independently written dense-mask correctness oracle, not OpenVDN's trained hybrid model, FlexAttention kernel, weights, or a production acceleration claim. |
| **MiniMax H3 Reference Budget** | MiniMax-H3 / ComfyUI packed-reference contract; secondary context from **PAB**, **DiffCR/VideoDiffCR** and token-reduction literature | Measurement of direct-reference row growth and a bounded experimental direct-reference cap | PAB and VideoDiffCR are **not** evidence that H3 should use a fixed reference budget. No published token-pruning policy is copied, and already encoded Qwen3-VL context is not modified. |
| **Runtime Metrics Probe** / **Metrics JSON** | ComfyUI, Spectrum, SA-Solver and the sibling integration APIs | Auditable logical-call, actual/forecast, phase, geometry, timing, and reset provenance | These are instrumentation nodes, not implementations of the cited research algorithms. |

## Primary flow-guidance and high-resolution research

The **Progressive Mixed-Grid Continuum** path independently combines ComfyUI's
native H3 patch projection, packed positions, per-row timestep modulation, and
output projection with Continuum's original protected prefix and the companion
learned upscaler. It reuses the existing progressive handoff mathematics.
Mixed-grid packing is this repository's H3-specific implementation; it does
not claim a new implementation of RALU, HRDiT, or another paper's algorithm.
VDN's explicit mixed-sequence mode retains its learned dense softmax gate while
omitting the geometry-dependent branch during that stage.

### HiFlow — Training-free High-Resolution Image Generation with Flow-Aligned Guidance

Jiazi Bu, Pengyang Ling, Yujie Zhou, Pan Zhang, Xiaoyi Dong, Yuhang Zang, Yuhang Cao, Tong Wu, Dahua Lin, Jiaqi Wang.

- Paper: https://arxiv.org/abs/2504.06232
- Project: https://bujiazi.github.io/hiflow.github.io/
- Code: https://github.com/Bujiazi/HiFlow
- Direct influence here: reference-flow thinking; trajectory rather than endpoint-only guidance; direction alignment; adjacent velocity-difference acceleration.

HiFlow is the primary algorithmic credit for the `direction` / `direction+acceleration` research path. H3-specific reconstruction of velocity from `x0 = x - sigma * v`, low-frequency filtering, PECE same-coordinate handling, and the final correction guard are local adaptations.

### FrescoDiffusion — 4K Image-to-Video with Prior-Regularized Tiled Diffusion

Hugo Caselles-Dupré, Mathis Koroglu, Guillaume Jeanneret, Arnaud Dapogny, Matthieu Cord.

- Paper: https://arxiv.org/abs/2603.17555
- Project: https://obvious-research.github.io/frescodiffusion/
- Direct influence here: treating a low-resolution video trajectory as a global spatiotemporal prior throughout higher-resolution generation rather than using only the final low-resolution endpoint.

FrescoDiffusion's tiled least-squares velocity fusion and spatial activity mask are not implemented here.

### RALU — Training-free Mixed-Resolution Latent Upsampling for Spatially Accelerated Diffusion Transformers

Wongi Jeong, Kyungryeol Lee, Hoigi Seo, Se Young Chun.

- Paper: https://arxiv.org/abs/2507.08422
- Project: https://ignoww.github.io/RALU_project/
- Code: https://github.com/ignoww/RALU
- Direct influence here: the warning that a noisy latent cannot simply be resized mid-trajectory; resolution transitions must account for changed noise/timestep statistics.

The progressive H3 handoff does not copy RALU's mixed-resolution mask or NT-Matching formula. It uses H3's own rectified-flow conditional state and a fresh solver lifetime. The experimental target-sparse path likewise does not implement NT-Matching; its retained target-grid row set and hidden-space lifter are separate from RALU's mixed-resolution transition method.

### Make a Cheap Scaling: A Self-Cascade Diffusion Model for Higher-Resolution Adaptation

Lanqing Guo, Yingqing He, Haoxin Chen, Menghan Xia, Xiaodong Cun, Yufei Wang, Siyu Huang, Yong Zhang, Xintao Wang, Qifeng Chen, Ying Shan, Bihan Wen.

- Paper: https://arxiv.org/abs/2402.10491
- Project: https://guolanqing.github.io/Self-Cascade/
- Code: https://github.com/GuoLanqing/Self-Cascade
- Direct influence here: semantic-pivot / self-cascade reasoning and staged resolution adaptation.

Its trainable upsampler modules and exact pivot-replacement method are not transplanted into this project.

### CineScale — Free Lunch in High-Resolution Cinematic Visual Generation

Haonan Qiu, Ning Yu, Ziqi Huang, Paul Debevec, Ziwei Liu.

- Paper: https://arxiv.org/abs/2508.15774
- Project: https://eyeline-labs.github.io/CineScale/
- Code: https://github.com/Eyeline-Labs/CineScale
- Direct influence here: coarse-to-fine video regeneration, upscale/re-noise/regenerate structure, and the observation that high-resolution DiT inference can fail for reasons beyond simple lack of pixels.

CineScale's Wan-specific positional/attention changes are not applied to H3 MM-RoPE.

### Just-in-Time — Training-Free Spatial Acceleration for Diffusion Transformers

Wenhao Sun, Ji Li, Zhaoqiang Liu.

- Paper: https://arxiv.org/abs/2603.10744
- Code: https://github.com/Wenhao-Sun77/Just-in-Time
- Direct influence here: the compute-allocation principle that early global-structure stages should not necessarily pay the full final-resolution spatial cost; for the experimental target-sparse Continuum path, the more specific public anchor-token/lifter idea also motivates retaining a target-grid anchor lattice and restoring a full hidden field before the native output layer.

The normal H3 progressive handoff does not implement JiT's SAG-ODE, DMF, anchor-token lifter, or deterministic micro-flow. The experimental target-sparse Continuum path independently implements only the high-level anchor/lifter compute-allocation idea: it selects native target-grid H3 rows and applies a bilinear hidden-space lifter while retaining every exact protected row. It does not reproduce JiT's SAG-ODE, deterministic micro-flow, source code, or model-specific mechanics.

### simple diffusion: End-to-end diffusion for high resolution images

Emiel Hoogeboom, Jonathan Heek, Tim Salimans.

- Paper: https://arxiv.org/abs/2301.11093
- Proceedings: https://proceedings.mlr.press/v202/hoogeboom23a.html
- Direct influence here: the resolution-dependent SNR/noise-schedule argument used to justify investigating, rather than assuming, an H3 resolution-aware schedule.

### Scaling Rectified Flow Transformers for High-Resolution Image Synthesis

Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, Dustin Podell, Tim Dockhorn, Zion English, Kyle Lacey, Alex Goodwin, Yannik Marek, Robin Rombach.

- Paper: https://arxiv.org/abs/2403.03206
- Direct influence here: the fractional-linear timestep shift and the `sqrt(target observations / reference observations)` constant-observation derivation used by the experimental resolution-aware SIGMAS node.

The node composes this as a relative factor with H3's native video shift instead of replacing H3's model sampling.

## High-resolution attention research

### OpenVDN — Video DeltaNet

- Project: https://openvdn.github.io/
- Code: https://github.com/OpenVDN/OpenVDN
- Inspected code revision: `b8cb28fbfca0266d1c7742a9f25ab8b58191de97`
- Direct influence here: complete 5-frame chunk alignment, radius-1 previous/current/next temporal windows, globally visible non-video tokens, and first/last row-and-column boundary-anchor semantics.

Flow independently implements that public topology as an analytic diagnostic and dense additive-mask reference. It does not copy OpenVDN's hybrid linear branch, compiled FlexAttention/BlockMask implementation, cache policy, training code, or checkpoints. OpenVDN's repository code is Apache-2.0, while its separately distributed VDN-H3 weights are derivatives of MiniMax H3 and are governed by the MiniMax H3 Community License Agreement, including territorial restrictions. This project does not download or distribute those weights.

### FreeSwim — Revisiting Sliding-Window Attention Mechanisms for Training-Free Ultra-High-Resolution Video Generation

Yunfeng Wu, Jiayi Song, Zhenxiong Tan, Zihao He, Songhua Liu.

- Paper: https://arxiv.org/abs/2511.14712
- Code: https://github.com/WillWu111/FreeSwim
- Direct influence here: native-scale local receptive fields, inward/window locality, and a separate global semantic path as motivation for the Attention Lab.

### HRDiT — Training-Free High-Resolution Image Generation with Off-the-Shelf Diffusion Transformer Models

Yu Xue, Haoxuan Qu, Zhuoling Li, Hongbin Xu, Jianxiong Yin, Simon See, Hossein Rahmani, Jun Liu.

- Paper: https://arxiv.org/abs/2608.07003
- Code: https://github.com/zylwithxy/HRDiT
- Direct influence here: positional-capacity diagnosis and heterogeneous/adaptive attention-head computation as research directions.

### ResDiT — Evoking the Intrinsic Resolution Scalability in Diffusion Transformers

Yiyang Ma, Feng Zhou, Xuedan Yin, Pu Cao, Yonghao Dang, Jianqin Yin.

- Paper: https://arxiv.org/abs/2512.01426
- Direct influence here: separating global-layout positional behavior from local-detail receptive-field behavior, and using local/global attention as a diagnostic model of high-resolution DiT failure.

No ResDiT PE-scaling or spectral-fusion module is implemented here.

## Temporal correspondence research

The experimental `direction+temporal` mode is not attributed to a single paper. Its design was informed by several lines of work on propagating information across video frames while respecting correspondence confidence and occlusion.

### TokenFlow — Consistent Diffusion Features for Consistent Video Editing

Michal Geyer, Omer Bar-Tal, Shai Bagon, Tali Dekel.

- Paper: https://arxiv.org/abs/2307.10373
- Project: https://diffusion-tokenflow.github.io/
- Code: https://github.com/omerbt/TokenFlow
- Influence here: explicit inter-frame correspondence in diffusion feature space as a route to temporal consistency.

### FRESCO — Spatial-Temporal Correspondence for Zero-Shot Video Translation

Shuai Yang, Yifan Zhou, Ziwei Liu, Chen Change Loy.

- Paper: https://arxiv.org/abs/2403.12962
- Code: https://github.com/williamyang1991/FRESCO
- Influence here: explicit spatial/temporal correspondence constraints and the need for stronger validity checks than unconstrained feature copying.

### MoVideo — Motion-Aware Video Generation with Diffusion Models

Jingyun Liang, Yuchen Fan, Kai Zhang, Radu Timofte, Luc Van Gool, Rakesh Ranjan.

- Paper: https://arxiv.org/abs/2311.11325
- Project/code page: https://github.com/JingyunLiang/MoVideo
- Influence here: optical-flow correspondence, warped latent guidance, and explicit occlusion-aware trust as design evidence for gating temporal propagation.

### Upscale-A-Video — Temporal-Consistent Diffusion Model for Real-World Video Super-Resolution

Shangchen Zhou, Peiqing Yang, Jianyi Wang, Yihang Luo, Chen Change Loy.

- Paper: https://arxiv.org/abs/2312.06640
- Code: https://github.com/sczhou/Upscale-A-Video
- Influence here: flow-guided recurrent latent propagation for long-range temporal stability.

### LatentWarp — Consistent Diffusion Latents for Zero-Shot Video-to-Video Translation

Yuxiang Bao, Di Qiu, Guoliang Kang, Baochang Zhang, Bo Jin, Kaiye Wang, Pengfei Yan.

- Paper: https://arxiv.org/abs/2311.00353
- Project: https://diffusion-latentwarp.github.io/
- Influence here: warping latent features along motion correspondence to constrain adjacent-frame representations.

This project does not load the optical-flow networks or reproduce the training/inference pipelines of these works. The current H3 matcher is deliberately local, latent-only, bounded, and experimentally neutral in decoded-media validation.

## Sampling and feature-forecast integration research

### Spectrum — Adaptive Spectral Feature Forecasting for Diffusion Sampling Acceleration

Jiaqi Han, Juntong Shi, Puheng Li, Haotian Ye, Qiushan Guo, Stefano Ermon.

- Paper: https://arxiv.org/abs/2603.01623
- Official code: https://github.com/hanjq17/Spectrum
- H3 sibling integration used by the validated workflow: https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3
- Influence here: forecast provenance, actual-anchor requirements, and the need to reset feature-history lifetimes across a geometry transition.

The forecasting algorithm itself lives in the Spectrum sibling repository, not in this package.

### SA-Solver — Stochastic Adams Solver for Fast Sampling of Diffusion Models

Shuchen Xue, Mingyang Yi, Weijian Luo, Shifeng Zhang, Jiacheng Sun, Zhenguo Li, Zhi-Ming Ma.

- Paper: https://arxiv.org/abs/2309.05019
- Official code: https://github.com/scxue/SA-Solver
- H3 sibling scheduler/integration used during development: https://github.com/xmarre/ComfyUI-MiniMax-H3-RefDelta-Solver
- Influence here: predictor/corrector and multistep-history semantics that the trajectory capture and progressive reset contracts must preserve.

### RACER — Disagree to Accelerate: Closing the Loop on Diffusion Feature Forecasts

Yanchao Li, Jiaqing Xie, Ben Gao, Wanhao Liu, Yanbo Wang, T. Y. Tsui, Jinfei Liu, Yuqiang Li, Tianfan Fu.

- Paper: https://arxiv.org/abs/2608.01740
- Code: https://github.com/LiZaiyuan0619/RACER
- Role here: secondary research on forecast trust / refresh decisions. This repository does **not** implement RACER's controller.

## Secondary attention/token-efficiency references

These works were consulted while evaluating possible H3 reference-token or attention experiments. They are intentionally **not** presented as direct algorithmic ancestors of the current Reference Budget node.

- **Pyramid Attention Broadcast (PAB)** — Xuanlei Zhao, Xiaolong Jin, Kai Wang, Yang You. Paper: https://arxiv.org/abs/2408.12588. PAB concerns timestep-wise attention-output reuse; it does not establish a fixed H3 reference-token budget.
- **DiffCR — Layer- and Timestep-Adaptive Differentiable Token Compression Ratios for Efficient Diffusion Transformers** — Haoran You et al. Project/code: https://www.haoranyou.com/diffcr/ and https://github.com/GATECH-EIC/DiffCR. It motivates content-aware token-efficiency research, not the specific H3 cap.
- **VideoDiffCR — Content-Aware Adaptive Token Pruning for Autoregressive Video Diffusion Transformers** — A. Shrivastava, C. Barnes, H. You, Y. Gong, Y. Kang, A. Owens, E. Shechtman. ICML 2026 From Frames to Stories workshop. It is background for content-aware video-token reduction; no VideoDiffCR pruning rule is implemented here.

## H3 and ComfyUI implementation foundations

These are executable/source contracts rather than merely conceptual citations.

- **MiniMax-H3** — https://github.com/MiniMax-AI/MiniMax-H3 — public model family, native H3 facts, public sparse-attention statements, and the documented H3-Context-IR -> H3-Base -> H3-Regenerate-2K product path. H3-Regenerate-2K internals are not public and are not claimed here.
- **ComfyUI** — https://github.com/Comfy-Org/ComfyUI — public H3 packing, joint AV sampling, flow shifts, MM-RoPE/layout behavior, model patch/wrapper semantics, masks, and sampler contracts.
- **ComfyUI-Spectrum-MiniMax-H3** — https://github.com/xmarre/ComfyUI-Spectrum-MiniMax-H3 — Spectrum provenance and exact/forecast integration used by the validated workflows.
- **ComfyUI-MiniMax-H3-RefDelta-Solver** — https://github.com/xmarre/ComfyUI-MiniMax-H3-RefDelta-Solver — H3 SA-Solver/PECE schedule and call-topology integration studied during implementation.
- **ComfyUI-H3-Continuum** — https://github.com/xmarre/ComfyUI-H3-Continuum — per-chunk H3 model/refine-state, continuation, mask, geometry, and session contracts.
- **Comfyui_Minimax_h3_latent_Upscaler** — https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler — learned H3 3D upscaler/refiner and the versioned clean-video provider consumed by `learned_3d` handoff. The learned model's inherited upstream lineage is recorded immediately below.
- **ComfyUI-DiffAid-Patches** — https://github.com/xmarre/ComfyUI-DiffAid-Patches — external patch compatibility and refinement-anchor interaction used by the benchmark workflow.
- **ComfyUI-Untwisting-RoPE** — https://github.com/xmarre/ComfyUI-Untwisting-RoPE — external H3 RoPE patch compatibility used by the benchmark workflow.

### Learned handoff: inherited model and architecture lineage

`MiniMax-H3-Flow-Aligned-Regenerate` does not load the learned-upscaler checkpoint itself. The `learned_3d` handoff consumes the versioned provider from the companion upscaler repository, so the provider's own upstream/model lineage is an indirect but material credit for this node:

- **LBH-123-AI / Comfyui_Minimax_h3_latent_Upscaler** — https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler — original MiniMax H3 learned-upscaler implementation lineage. The pretrained checkpoints used by the companion package are published at https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler.
- **Ttl / ComfyUi_NNLatentUpscale** — https://github.com/Ttl/ComfyUi_NNLatentUpscale — neural latent-upscaling approach credited by the companion package as a foundational influence.
- **Lightricks / LTX-2** — https://github.com/Lightricks/LTX-2 — the companion package credits the LTX 2.3 Spatial Upscaler (`ltx-2.3-spatial-upscaler-x2-1.1.safetensors`) as an architectural influence.
- **Tr1dae / ComfyUI-MiniMaxH3_LatentUpscaler** — https://github.com/Tr1dae/ComfyUI-MiniMaxH3_LatentUpscaler — studied by the companion package while implementing native H3 refinement integration. It is research/integration provenance, not a Flow runtime dependency.

Those projects are not vendored by this Flow package. The actual runtime dependency boundary is the companion `H3_LATENT_UPSCALER` provider API; upstream model/code licenses remain their own.

The exact executable revisions pinned by CI are listed in `docs/DEVELOPMENT.md` and `docs/RESEARCH.md`.

## Research implementation revisions audited

The supplied source ZIPs do **not** contain Git metadata, so no commit SHA is inferred from the archives themselves. During this credit/source audit, each supplied snapshot was reviewed alongside its corresponding upstream branch; the branch heads recorded below are the upstream revisions audited at that time. License state refers only to the top-level files observed in the supplied source snapshot.

| Research implementation | Upstream revision audited | Top-level license observed in supplied snapshot |
|---|---|---|
| HiFlow | `f397a1fbde770108b500ec3809fdd7b13fc61a75` | Apache-2.0 |
| RALU | `3979f8cab54332ec2d16f8646b68fa134d258990` | No top-level license file observed |
| Self-Cascade (`stable-diffusion-xl`) | `b847d4f35d36ec046d7d3886548d09dd800efb99` | No top-level project license file observed; a vendored `clean_fid` component has its own MIT notice |
| FreeSwim | `9f35a72f6382696fe2fd02ff02183a249c3090f3` | No top-level license file observed |
| HRDiT | `f3be935da797c7dc98ecff07cf4a5f01e1ddb51d` | MIT |

Absence of a top-level license in an inspected research snapshot must **not** be interpreted as permission to reuse its code. The citations above document research provenance; they do not grant rights beyond the upstream authors' actual licenses.

## Attribution policy for future changes

When a new paper or repository materially influences an algorithm, heuristic, node, or default in this project:

1. add the paper and official implementation here;
2. state exactly which idea transferred and which parts did not;
3. map it to the affected public node or experiment;
4. record an inspected source commit when code was materially consulted;
5. preserve upstream copyright/license notices if source code is ever incorporated rather than independently reimplemented.

That standard is intentionally stricter than a generic acknowledgements paragraph: users should be able to trace each research-facing node back to the work that motivated it.
