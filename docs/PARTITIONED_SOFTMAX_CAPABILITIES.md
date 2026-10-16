# Partitioned softmax provider capabilities

Flow publishes its own request callback under
`vdn_partitioned_attention_provider_v1`. That callback transports the physical
partition, prefix measure, query map and diagnostic controls to the selected
attention backend. Provider capabilities must follow this callback boundary as
well as the final attention call.

For Sol-selected block options, Flow creates a stable binding that forwards calls
through the existing Flow dispatcher and carries the installed Sol provider's
optional `grouped_kv_v1` producer and explicit read-only Q guarantee. VDN can then
fuse its restricted-domain K/V gather with Sol preparation and use contiguous Q
views in continuation, just as native VDN does. The binding owns Python callables
only. Group summaries and tensors remain inside Sol's synchronous group scope.

The binding is reused across repeated block-option copies and rebinds if the Sol
provider or its capabilities change. It rejects calls with non-Sol options.
Native/custom backends retain the original callback and receive no Sol capability
or new Sol dependency. Older Sol versions without these capabilities retain their
existing gathers and attention behavior.

This changes neither the query-origin-v5 kernel ABI nor physical row geometry,
key measures, exact prefixes, dense-query policy, learned VDN operations, weights,
precision or sampler evaluation count. CPU integration tests cover the actual
Flow options publication, VDN partitioned forward, raw learned-linear inputs and
Sol group scope for equal and mixed grids. The CUDA producer and attention kernel
are replaced by CPU oracles in those tests; they do not establish GPU throughput.

With Sol #41 and VDN #42 installed, sparse continuation should report nonzero
`grouped_kv_fused_calls`, alongside unchanged complete VDN learned-linear counts.
A zero count for a dense warmup/probe is expected. Compare the same workflow's
sampler time, actual model evaluations and memory after updating through Patcher.
Full BF16 parity, output quality, GPU speed and peak/live VRAM remain separate
acceptance requirements. Preserve VDN #41 before #42 for its admission policy.
