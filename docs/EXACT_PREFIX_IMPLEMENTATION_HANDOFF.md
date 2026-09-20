Implement the gated MiniMax-H3 exact-prefix continuation design in:
https://github.com/xmarre/MiniMax-H3-Flow-Aligned-Regenerate/blob/7389ff4b5ec0ef0d45ea77a6a09d97cc952a540b/docs/EXACT_PREFIX_SEMANTICS_PRESERVING_CONTINUATION_DESIGN.md

Authoritative design commit: 7389ff4b5ec0ef0d45ea77a6a09d97cc952a540b. Read the complete document before editing. It is a candidate implementation specification with mandatory empirical gates, not a claim that the production fix is already known.

Re-fetch live defaults, PR heads/bases, changed files, reviews/comments, CI and repository instructions. Reconstruct the effective layered runtime. Preserve:
- Flow #49 d1b32f530b4cc2298eba68f08c2da44a094c7cfe -> #54 b4abdc9bbf5139e328788fa698c18377f7ae54d0 -> #56 1250959f1b9231335bf6f669fba8553e9366113f -> #57 6caf039e78513c2d78ee6623590468ac21da8adb -> #58 8118a293d9e98fc43fcb2f46e3609c8fbddbde87.
- VDN #19 675cd6e7b43c9d8d63b7a91834fb6d775388a2ee -> #22 25bca63ab7ede45b3dc8af461f589475c2f9aa20 -> #23 b584070b37b3ec5ea2be97e96c420b982b5aab5f -> #26 b853289ae5c4dd35430acc4a13daa8e383e2f26a -> #27 b686f96d554609b5e00ad94bb22c44665c217688 -> #28 1c01d53504c7585b72843e0bd3d3c3b6f72d6628.
- Sol #15 93b3e03f2b7b579aaf55fa0f87f55083b259e25c.
- Continuum #24 6167dcbae8ab239ea660fed52d03ddb63aa515fb -> #28 688c754b19664b3812d462af6d5a3414bfc1b06b -> #29 922002f3debaa424bd8fe3847560276a156e8568.

Use separate implementation mirrors and GitHub checkpoints before risky work and after substantial progress. Put final changes on new, correctly stacked PRs with one clean implementation commit each. Do not rewrite diagnostic heads, consolidate foundational PR histories, touch Flow #52/#55, or give the user manual git commands. Validation uses ComfyUI Patcher overlays.

Implement the narrow Flow candidate in section 7: retain heterogeneous exact target-prefix context and full H3/VDN hidden-state evolution; change only target-audio spatial RoPE ownership in low/probe to source-carrier coordinates, behind a diagnostic selector. Preserve legacy behavior by default. Transport policy identity through the existing partitioned-layout signature recognized by Sol history. No Sol kernel change is justified. Keep VDN's raw-token anchor-coverage correction separate if implemented; it is not an established seam repair.

Preserve exact external AV-prefix ownership, native timestep/velocity-mask ordering, learned 3D handoff, seven-reference presentation, physical Continuum prompt transport, VDN learned-linear/anchor semantics, Sol mapped rectangular routing, Spectrum histories, DiffAid/Untwist preprocessing and existing VDN scratch/release lifetimes. Do not freeze prefix hidden rows, remove their queries without a justified layerwise recurrence, add smoothing, or revive closed ablations.

Verify implementation assumptions against actual installed source. Core 673afefe on patcher/stack could not be resolved publicly; do not substitute public master as installed provenance. Document justified deviations when new source/evidence contradicts the design.

The audit corrected two causal premises: 00546 cuts to the iguana exactly at frame 175, so its PT212 median does not prove same-shot seam repair; fish behavior differs before the continuation intervention. Prompt/reference receipts match, but complete conditioning/noise identity was not established.

Finish focused source/contract tests and the bounded fixture/candidate tooling even without local CUDA. Follow section 8's two-continuation comparison using one identical captured first-chunk/carry/conditioning/noise fixture; require matched SM120 video/audio review before promotion. Preserve 00546's user-accepted audio while recovering the lizard action through approximately frame 190 and accepted fish behavior.

Required external evidence: the six 00545/00546 log, metrics and MP4 files identified and hashed in section 3. Future validation also needs exact first-chunk latents/carry/noise, compiled conditioning, all seven reference assets, model/VDN/upscaler identities, installed Patcher/source manifest and pre-video-seam boundary frames. Original raw audio00545.f32 is unnecessary unless a specific decoder question is reopened. Do not claim behavioral success from CI alone.
