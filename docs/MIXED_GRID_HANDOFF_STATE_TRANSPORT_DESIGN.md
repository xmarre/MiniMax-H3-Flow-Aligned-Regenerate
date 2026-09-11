# Mixed-Grid handoff state transport: investigation checkpoint

Design-only investigation; no production change is authorized here.

Verified baseline: Flow PR #30 8dfb7a2e570458b5078734755b897c7e89ddac8c, one implementation commit. PR #32 live head is f0f0b1e5a54de6ff31a31a795a298ab9b1961dc7 (the supplied full SHA had an incorrect suffix). PR #29 remains 3afc6c63063f410afec756c42811a8cf1ad2ed77. All three have successful current PR CI and no submitted reviews.

Primary evidence materialized: /workspace/scratch/266da105e66e/evidence/ComfyUI-Sol-H3/metrics_00383_.json (libfile_a814372042488191be0a43b2a580ce5f), and Pasted text(20260911-174109).txt in the same directory (libfile_92d79c4c490c81918427d577b03ea782). Metrics reproduce 18 logical / 14 actual / 4 forecast, the repaired clean seam 0.42617952823638916 / 0.22551529109477997 / 0.1019718274474144, and first raw exact high prediction 0.5334718227386475 / 0.27810272574424744 / 0.174343079328537. Low mixed weighted SM120 log reports 200 weighted calls and 8,991,600 Q and KV rows, no compatibility fallback. Probe has 50 weighted calls; ordinary high VDN native route counters must not be misclassified as mixed weighted failures.

Source establishes CONST noise_scaling x=sigma*noise_scale*noise+(1-sigma)*latent_image; inverse_noise_scaling divides by (1-sigma). Flow recovers raw state by process_latent_in followed by (1-sigma). Probe returns denoised*(1-sigma), then inverse scaling/output processing yields clean x0. The resizing handoff ignores source video state and re-noises learned clean x0 with a separate deterministic field; packed audio state is copied. Existing representation/DC state mapping adds (1-sigma)*clean_delta. Prefix noise is restored from caller before native inpainting.

Open architectural choice: preserving effective noise residual x-(1-sigma)*x0 differs from preserving sigma-velocity residual x-x0. Both close algebraically and coincide for a linear clean lift matching the state lift; nonlinear learned clean transfer separates them. Source alone cannot prove either removes the media seam. Spatial interpolation also changes residual covariance; document this rather than claim noise-law preservation.

Exact production core commit is local patcher/stack 7803b74e (log: v0.35.0-15); upstream Contents lookup returns 404. Current upstream sampler/model sources and pinned core companion 8406da920f9df19cbc14b76aef1bcd3dc4cf1119 are accessible; exact installed stack bytes remain a production re-check.

Safety branch: checkpoint/state-transport-baseline-20260911 at PR #30. Investigation sources are under /workspace/scratch/266da105e66e/sources, diagnostic sources under sources/diag. Final design will replace this checkpoint text after solution-space audit.
