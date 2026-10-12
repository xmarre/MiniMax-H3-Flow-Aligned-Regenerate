# Full-context continuation stage comparison

Use **MiniMax H3 Local Boundary Audit** with `audit_scope=full_stage_continuity`
to compare an existing uniform-source boundary capture through the production
native video VAE. No generation or sampler execution is required.

Import [the canvas workflow](../workflows/examples/full-stage-continuity.workflow.json),
select the production native video VAE, and select the **boundary** bundle named
by `partitioned_boundary_window_evidence.fields.bundle` in the generation metrics.
The frozen low A/B/A bundle is a different contract and cannot be used here.
Set `chunk_join_frame` to the assembled first retained continuation frame. The
example uses 175; change it for other clip lengths or overlaps. Choose named
background regions that remain visible and static in the captured scene. The
example room profile is only appropriate for its bookshelf/picture/curtain scene;
otherwise choose `custom` and provide fractional XYXY rectangles.

The scope decodes seven complete saved timelines sequentially:

1. Native source clean prediction at the handoff sigma.
2. Learned provider clean output with its original native prefix.
3. That provider output with the authoritative target prefix restored.
4. Corrected clean handoff before high sampling.
5. First high prediction before Flow guidance.
6. First high prediction after Flow guidance.
7. Final high-refined clean output.

Every decode starts at native token zero. Global frame labels account for the
protected context trimmed from the continuation. Target-grid comparison stages use the
same authoritative prefix, except the explicit `provider_native` arm; the source stage retains its hash-verified source
prefix. The report identifies full decoder input lengths and origins, and
measures matching frame times from the available local boundary interval.
This avoids treating a shorter replay's artificial clip origin as full-video
decoder evidence. Full clips cost more decode time and temporary RGB memory than
cropped windows. Admission uses the existing resident-VAE memory contract.
`validate_full_video_decoder` must remain false because this scope already uses
full decodes. Mixed target-band bundles are unsupported by this scope.

`absolute_same_frame_stage_pairs` reports unregistered RGB residuals, luminance
bias, centered residuals, structural agreement, edge-energy ratios and estimated
ROI displacement at every measured frame, including the join itself. Comparison
uses an area-reduced common canvas of at most 704 pixels per side and never
upsamples the source. These measurements retain the pre-join baseline mismatch;
the existing prefix-normalized texture trends remain available separately.

With `save_stage_previews=true`, the node saves four PNGs per stage at join−1,
join, join+5 and join+15 when those times are available. PNGs retain each stage's
native spatial grid and quantize finalized RGB to eight bits by rounding. They
appear in the node preview and in a fresh directory below
`output/h3_flow_regenerate/boundary_audits/stage-previews-*`. The adjacent JSON
report records each filename, stage, frame, dimensions and quantization. No
extra VAE call is introduced by saving previews. Default settings continue to
save only the numerical report.

Use source→provider-native to inspect learned transfer with its native prefix,
provider-native→provider to isolate exact-prefix restoration and decoder response,
and provider→pre-high to inspect clean correction, pre-high→first-high to inspect
initial refinement, before→after Flow to inspect guidance, and first-high→final
to inspect subsequent refinement. These separate pairs prevent attribution of exact-prefix decoder effects to
learned transfer. Source and provider still use different latent grids and VAE
representations; their fit differences alone are not causal camera transforms.

Decoded pre-join pixels are discarded context from the current continuation,
not the preceding chunk's assembled production pixels. Image fits are not camera
ground truth, and changing/occluded regions cannot support static registration.
Full stage comparison does not rerun a target-high B arm, reconstruct the prior
chunk's assembled frames, modify production output, or qualify audio. Rendered
AV acceptance remains a separate requirement.


Release interpretation: slight darkening remains unresolved. The user reports good boundaries across multiple prompts and references. Fitted camera motion and sharpness changes can reflect conditioning and focus; they do not establish unwanted zoom or bokeh. No Spectrum cause has been established.
