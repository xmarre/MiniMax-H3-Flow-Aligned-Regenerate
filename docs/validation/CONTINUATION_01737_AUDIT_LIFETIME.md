# Run 01737: continuation acceptance and audit ownership

The supplied generation log, `metrics_01737_.json` and three numerical audit
reports describe the same saved bundle. All reports use manifest SHA256
`1e093cd6c4c3b102055ecbfd27763c3133a7259d08d8c75bdcef290368a25ada`.
The user reports no visible bad boundary and accepts the observed throughput.
No decoded frames were supplied for independent visual review.

## Measured continuation

The run selects `progressive_uniform_source`: low/probe evaluates one native
reduced-grid trajectory, rather than independent target-band and tail authors.

| Measurement | First chunk | Continuation |
| --- | ---: | ---: |
| Low | 67.917 s | 76.343 s |
| Probe | 10.149 s | 12.934 s |
| Transfer | 0.543 s | 0.954 s |
| High | 69.058 s | 90.312 s |
| Complete sampler | 147.694 s | 181.065 s |
| Low actual / forecast calls | 7 / 3 | 6 / 4 |
| High actual / forecast calls | 4 / 2 | 4 / 2 |

Total sampler time is 328.759 s; the logged complete prompt takes 365.35 s.
Continuation sampling costs 1.226 times first-chunk sampling. The continuation
low stage forecasts successfully instead of executing all ten model calls.
These are measurements of this run, not a matched hardware benchmark against
the earlier run or a guarantee for other prompts and settings.

Final mean luminance is 0.648965 at frame 174 and 0.648225 at frame 175, a
change of -0.000740. At the former band/tail location, frame 191 is 0.646168
and frame 192 is 0.645970. The large low-derived brightness split seen in
01734 is absent in these measured windows. User visual acceptance applies to
this run; scalar measurements alone do not prove geometric or audio quality.

## Audit lifecycle regression

Avoiding managed VAE decode did not address model ownership outside decoding.
Core prepares an audit-only prompt by evaluating its fingerprints and pruning
the previous generation's output/object caches. `LoadedModel` keeps weak
references to its patcher; destroying a patcher invokes its detach lifecycle.
Post-prompt garbage collection can therefore finish releasing the generation's
models even when the audit decoder never requested their eviction.

The native regression seeds real Core patchers in generation caches, replaces
the prompt with an audit, then performs completion and garbage collection.
The old fingerprint loses all three patchers. The corrected fingerprint takes
strong owners before pruning and uses Core's public cache-provider lifecycle
to retain them after audit completion. A later non-audit prompt releases these
owners after execution. Ordinary memory admission and explicit unload remain
available; no Core function or memory-policy flag is replaced.

Fingerprints remain plain NaN. Model references are not embedded in history,
output signatures or an accumulating cache. Repeated audits replace the one
owner snapshot. Failed audits retain the same ownership protection. A VAE
already on its decode device, including partially loaded weights, is decoded
in place. Nonresident VAE admission now counts off-device weights in full and
includes Core's 110% weight reserve before entering admission.

The supplied log ends with generation completion and does not show an audit
prompt or its cleanup flags. It cannot rule out explicit unload requests or
Core's disable-smart-memory policy in that session. The cache-ownership defect
is independently reproduced with native Core classes. The new ownership fix
still requires a post-audit residency check in the user's GPU session; run
01737 predates it and cannot verify it.
