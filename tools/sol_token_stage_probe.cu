// SPDX-License-Identifier: Apache-2.0
// Include unmodified pinned token source. Its normal launch function remains
// available to the combined stage probe; diagnostic phases expose its internals.
#include <stdexcept>
#include "sol_attn_token.cu"

extern "C" void sol_token_probe_stage(int phase,
    const void* qmean, const void* tok_ref, const void* cand_bits,
    void* gcen8, void* gcens, void* gref, void* gbits,
    const void* kiP, const void* ksb, const void* vTi,
    void* tok_tiles, void* tok_tilecnt,
    void* tok_hist, void* tok_idx, void* tok_cnt,
    void* part_o, void* part_m, void* part_l,
    void* o_part, void* m_part, void* l_part,
    int Tp, int H, int NQ, int NPAD, int NTB, int n_tok,
    int sq0, int sq1, float sl, cudaStream_t stream) {
    const int W = (NTB + 31) / 32, NG = (NQ + TOK_GROUP - 1) / TOK_GROUP;
    const int nsplit = sol_token_splits(1, H, NG);
    dim3 grid((NG + BQ - 1) / BQ, nsplit, H);
    if (phase == 0) {
        sol_token_group_kernel<<<dim3(NG, H), HD, 0, stream>>>(
            (const float*)qmean, (const float*)tok_ref, (const uint32_t*)cand_bits,
            (int8_t*)gcen8, (float*)gcens, (float*)gref, (uint32_t*)gbits,
            NQ, NPAD, NG, W);
    } else if (phase == 1) {
        sol_token_tiles_kernel<<<dim3(grid.x, H), 128, 0, stream>>>(
            (const uint32_t*)gbits, (uint16_t*)tok_tiles, (int32_t*)tok_tilecnt,
            NG, NTB, W);
        cudaMemsetAsync(tok_hist, 0, (size_t)H * NG * NB * sizeof(uint32_t), stream);
        cudaMemsetAsync(tok_cnt, 0, (size_t)H * NG * sizeof(int32_t), stream);
    } else if (phase == 2 || phase == 3) {
#define PROBE_TOKEN_ARGS \
        (const int8_t*)gcen8, (const float*)gcens, (const int8_t*)kiP, \
        (const float2*)ksb, (const int8_t*)vTi, (const uint32_t*)gbits, \
        (const uint16_t*)tok_tiles, (const int32_t*)tok_tilecnt, \
        (const float*)gref, (uint32_t*)tok_hist, (uint32_t*)tok_idx, (int32_t*)tok_cnt, \
        (__nv_bfloat16*)part_o, (float*)part_m, (float*)part_l, \
        Tp, NG, NTB, n_tok, 1, NQ, sq0, sq1, sl
        if (phase == 2) {
            const int ns1 = max(nsplit, (NTB + P1_MAX_TILES * HIST_STRIDE - 1) / (P1_MAX_TILES * HIST_STRIDE));
            sol_token_kernel<1><<<dim3(grid.x, ns1, H), NTHREADS, token_smem_bytes(1), stream>>>(PROBE_TOKEN_ARGS);
        } else {
            sol_token_kernel<2><<<grid, NTHREADS, token_smem_bytes(2), stream>>>(PROBE_TOKEN_ARGS);
        }
#undef PROBE_TOKEN_ARGS
    } else if (phase == 4) {
        sol_token_sort_kernel<<<dim3(NG, H), NTOK_MAX, 0, stream>>>(
            (uint32_t*)tok_idx, (const int32_t*)tok_cnt, NG, n_tok);
    } else if (phase == 5) {
        sol_token_merge_kernel<<<dim3(NQ, H), HD, 0, stream>>>(
            (const __nv_bfloat16*)part_o, (const float*)part_m, (const float*)part_l,
            (__nv_bfloat16*)o_part, (float*)m_part, (float*)l_part, NQ, NG, nsplit);
    } else throw std::runtime_error("unknown token diagnostic phase");
}
