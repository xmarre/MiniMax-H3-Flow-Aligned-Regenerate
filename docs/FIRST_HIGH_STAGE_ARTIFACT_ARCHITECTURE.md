# First high-stage execution contract investigation

Status: investigation checkpoint; not an implementation specification yet. No production code changed.

Flow main verified at 970396db839ae7ab431b9718859f6d48a2e5019b. Production #33 head 9c400d46e990222cf723c426039b4c543192e185; diagnostic #35 a99ed7a7ca20ed3af5282c73cad4bb662e45f804; #36 cf260160df234f7a6c61787725b22eddd522c438. Preserve these separate PRs. #39 and #40 are closed, rejected state-transport experiments.

Primary runtime evidence located: metrics_00444_.json; Pasted text(20260914-043151).txt; firsthighpreguidance_00004.mp4; last_high_pre_guidance_00004.mp4. These characterize the failed #40 experiment, not an unmodified production baseline.

Concrete source lead: Sol-H3 sol_h3/interop.py dense_evaluation_warmup explicitly suppresses default first-evaluation dense warmup when h3_flow_stage=high and the Flow h3_refinement contract matches. Runtime logs show low call 1 phase=dense and high call 1 phase=sol. This is intentional policy, not evidence of leaked Request state: sol_h3/runtime.py SamplingWrapper creates a new Request for each outer invocation. Causality remains unproven. Do not remove this policy without a controlled media test and accounting of Spectrum's backend-transition behavior.

Flow rebuilds guider.conds from a shallow post-hook template per sampler lifetime; nested ownership needs auditing. Fresh target-grid video state is (1-sigma)*learned_clean+sigma*fresh_target_noise; carried audio uses source sampler state. _noise_argument reverses core initialization against target latent_image. Do not substitute x0 for the sampler state.

Required next work: verify core FLOW_AV conversion; condition aliases and native layout rebuild; VDN, Sol, Spectrum, DiffAid, Untwist and KJ ownership; installed source provenance; minimum bounded contract comparison; validation and promotion gates. No root cause is yet established.
