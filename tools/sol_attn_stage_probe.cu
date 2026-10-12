// SPDX-License-Identifier: Apache-2.0
// Standalone diagnostic only. Compile against the verified v0.2.37 source;
// this translation unit never replaces the installed comfy-kitchen extension.
#include "sol_attn.cu"

namespace { thread_local std::string probe_error; }
extern "C" const char* sol_probe_error() { return probe_error.c_str(); }
extern "C" void sol_token_probe_stage(int,
    const void*, const void*, const void*, void*, void*, void*, void*,
    const void*, const void*, const void*, void*, void*, void*, void*, void*,
    void*, void*, void*, void*, void*, void*,
    int, int, int, int, int, int, int, int, float, cudaStream_t);

// Expose the original launch boundaries with their original arguments. The
// Python driver synchronizes and observes between stages, outside the kernels.
extern "C" int sol_probe_stage(int stage, void* workspace, void* out,
    const void* vscale, void* kmean_next, void* vamax_out,
    int T, int H, float tau, float scale, int sb0, int sb1,
    int sq0, int sq1, int n_tok, uint64_t stream_value) {
    try {
        probe_error.clear();
        validate_shape(1, T, H);
        validate_token_aug(n_tok);
        const Plan p(1, T, H, n_tok);
        char* w = reinterpret_cast<char*>(workspace);
        cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_value);
        const float sl = scale * 1.4426950408889634f;
        if (stage == 0) {
            cudaMemcpyAsync(w + p.vsc, vscale, (size_t)H * 128 * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
            launch_sol_finish(w + p.scratch, w + p.kciP, w + p.kcs, w + p.thr,
                w + p.cen8, w + p.cens, kmean_next, nullptr,
                1, T, H, p.NTB, p.NPAD, p.NQ, tau, sl, stream);
        } else if (stage == 1) {
            launch_sol_route(w + p.cen8, w + p.cens, w + p.kciP, w + p.kcs,
                w + p.vcT, w + p.vsc, w + p.thr, w + p.idx, w + p.cnt,
                w + p.oPart, w + p.mPart, w + p.lPart,
                n_tok ? w + p.tokRef : nullptr, n_tok ? w + p.tokBits : nullptr,
                nullptr, 1, 1, T, H, p.NTB, p.NPAD, p.NQ,
                sb0, sb1, sq0, sq1, sl, stream);
        } else if (stage == 2) {
            if (n_tok) launch_sol_token(w + p.qmean, w + p.tokRef, w + p.tokBits,
                w + p.gCen8, w + p.gCens, w + p.gRef, w + p.gBits,
                w + p.kiP, w + p.ksb, w + p.vTi, w + p.tokTiles, w + p.tokTileCnt,
                w + p.tokHist, w + p.tokIdx, w + p.tokCnt,
                w + p.tokPartO, w + p.tokPartM, w + p.tokPartL,
                w + p.oPart, w + p.mPart, w + p.lPart,
                1, p.Tp, H, p.NQ, p.NPAD, p.NTB, n_tok, 1, sq0, sq1, sl, stream);
        } else if (stage >= 20 && stage <= 25) {
            if (n_tok) sol_token_probe_stage(stage - 20,
                w + p.qmean, w + p.tokRef, w + p.tokBits,
                w + p.gCen8, w + p.gCens, w + p.gRef, w + p.gBits,
                w + p.kiP, w + p.ksb, w + p.vTi, w + p.tokTiles, w + p.tokTileCnt,
                w + p.tokHist, w + p.tokIdx, w + p.tokCnt,
                w + p.tokPartO, w + p.tokPartM, w + p.tokPartL,
                w + p.oPart, w + p.mPart, w + p.lPart,
                p.Tp, H, p.NQ, p.NPAD, p.NTB, n_tok, sq0, sq1, sl, stream);
        } else if (stage == 3) {
            launch_sol_exact(w + p.qiP, w + p.qs, w + p.kiP, w + p.ksb,
                w + p.vTi, w + p.vsc, w + p.idx, w + p.cnt,
                w + p.oPart, w + p.mPart, w + p.lPart,
                w + p.vRow, w + p.tokIdx, w + p.tokCnt, n_tok, out,
                1, T, p.Tp, H, p.NQ, p.NTB, sl, sol::SOL_BF16, stream);
            cudaMemcpyAsync(vamax_out, w + p.statsV, (size_t)H * 128 * sizeof(float),
                cudaMemcpyDeviceToDevice, stream);
        } else throw std::runtime_error("unknown diagnostic stage");
        const cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
        return 0;
    } catch (const std::exception& e) { probe_error = e.what(); return -1; }
}
