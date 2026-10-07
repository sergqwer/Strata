// src/kernels/cuda/native_mmvq_vt.cuh - strata-gemma (moe-glue), included by native_mmvq.cu inside its anonymous
// namespace after the formats' traits: the warp-per-row layout of the single-column products and the multi-matrix
// launch (q / k / v, gate / up). Bitwise the per-row arithmetic of the 128-thread kernels; see tools/gemma/glue.
#pragma once

// ---- strata-gemma (moe-glue): the single-column products of the 32-value formats (Q4_0, Q5_0, Q8_0, IQ4_NL) with
// the per-row arithmetic of the 128-thread kernels above (native_small_mmvq_kernel, native_mmvq_id_kernel and the
// exact multi-column layout, at any ROWS), laid out one WARP per row: lane l does the work of the four threads
// l, 32 + l, 64 + l, 96 + l of the 128-thread block - their blocks kbx = tid / T + j BPI and sub-index kqs(tid), each
// summed in its own accumulator in ascending kbx exactly as that thread sums it - and then adds the four partial sums
// in the order the block's shared-memory pass adds them, ((s0 + s1) + s2) + s3, before the same xor tree. So every
// output is bitwise the 128-thread kernel's, with no shared memory, no __syncthreads and four independent load
// streams per lane (the 128-thread layout leaves 84 of 128 threads idle on K = 704, the experts' down projection).
// NJ = ceil(blocks_per_row / BPI): the iterations of the 128-thread kernel's loop; NVL: the virtual threads (of 4)
// with any block in the last iteration (the others are idle there for every lane). Every remaining (iteration,
// virtual thread, row) slot loads from a clamped block - all loads are straight-line code ahead of the arithmetic -
// and a slot past the row's end (or a row past n_out) leaves its accumulator untouched, as the 128-thread kernel
// never adds it.
template<typename F, int R, int NJ, int NVL>
__device__ __forceinline__ void vt_rows(const typename F::Block* __restrict__ w, int rows_left, int blocks_per_row,
                                        const Q81Block* __restrict__ x, int lane, float* __restrict__ out) {
    float acc[R][WARPS];
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
        for (int v = 0; v < WARPS; ++v) acc[r][v] = 0.0f;
#pragma unroll
    for (int j = 0; j < NJ; ++j) {
#pragma unroll
        for (int v = 0; v < (j == NJ - 1 ? NVL : WARPS); ++v) {
            const int tid = WARP * v + lane;
            const int kbx = tid / F::T + j * F::BPI;
            const bool valid = kbx < blocks_per_row;
            const int kb = valid ? kbx : blocks_per_row - 1;
            const int kqs = F::kqs(tid);
#pragma unroll
            for (int r = 0; r < R; ++r) {
                const int rr = r < rows_left ? r : 0;
                const float t = acc[r][v] + F::apply(F::load(w + std::size_t(rr) * blocks_per_row + kb, kqs), x + kb * F::KBY, kqs);
                acc[r][v] = valid && r < rows_left ? t : acc[r][v];
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; ++r) {
        float s = acc[r][0];
#pragma unroll
        for (int v = 1; v < WARPS; ++v) s += acc[r][v];
        out[r] = warp_sum(s);
    }
}

// one launch over up to 4 weight matrices that read the same activation (q / k / v, gate / up): the matrices' rows
// back to back, a warp per R rows (each matrix's row count a multiple of R)
struct VtJobs {
    const void* w[4];
    float* y[4];
    int n_out[4];
    int n = 0;
};
constexpr int VT_WARPS = 8;   // warps per block

// the 128-thread layout (native_small_mmvq_kernel<Weight, Qi, false>, one row per block, its code unchanged) over up
// to 4 matrices in one launch: block b is row b of the matrices taken back to back
template<typename Weight, int Qi>
__launch_bounds__(WARPS * WARP, 1)
__global__ void multi_w_small_kernel(const VtJobs jobs, const Q81Block* __restrict__ x, int n_in) {
    int row0 = int(blockIdx.x);
    const void* wv = jobs.w[0];
    float* y = jobs.y[0];
    int n_out = jobs.n_out[0];
#pragma unroll
    for (int j = 1; j < 4; ++j)
        if (j < jobs.n && row0 >= n_out) {
            row0 -= n_out;
            wv = jobs.w[j];
            y = jobs.y[j];
            n_out = jobs.n_out[j];
        }
    const Weight* __restrict__ w = static_cast<const Weight*>(wv);
    constexpr int ROWS = 1;
    constexpr int BLOCKS_PER_ITER = 2 * WARPS * WARP / Qi;
    const int tid = WARP * int(threadIdx.y) + int(threadIdx.x);
    const int blocks_per_row = n_in / 32;
    float tmp[ROWS] = {};
    for (int kbx = tid / (Qi / 2); kbx < blocks_per_row; kbx += BLOCKS_PER_ITER) {
        const int kqs = 2 * (tid % (Qi / 2));
#pragma unroll
        for (int i = 0; i < ROWS; ++i) {
            if (row0 + i < n_out) {
                const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                tmp[i] += small_q8_dot(w + block, x + kbx, kqs);
            }
        }
    }
    __shared__ float partial[WARPS - 1][ROWS][WARP];
    if (threadIdx.y > 0) {
#pragma unroll
        for (int i = 0; i < ROWS; ++i) partial[threadIdx.y - 1][i][threadIdx.x] = tmp[i];
    }
    __syncthreads();
    if (threadIdx.y > 0) return;
#pragma unroll
    for (int i = 0; i < ROWS; ++i) {
#pragma unroll
        for (int l = 0; l < WARPS - 1; ++l) tmp[i] += partial[l][i][threadIdx.x];
        tmp[i] = warp_sum(tmp[i]);
        if (threadIdx.x == i && row0 + i < n_out) y[row0 + i] = tmp[i];
    }
}

// the warp-per-row layout pays where the 128-thread loop's last pass is less than half full (K = 704 and 2816 of
// Q4_0, K = 2112 of Q8_0: -24%, -15%, -8% a call on the A4000); elsewhere the 128-thread kernels are as fast or faster
template<typename F> bool vt_pays(int n_in) {
    const int rem = (n_in / F::DIV) % F::BPI;
    return rem != 0 && rem < F::BPI / 2;
}

template<typename F, int R, int NJ, int NVL>
__launch_bounds__(VT_WARPS * WARP)
__global__ void vt_mmvq_kernel(const VtJobs jobs, const Q81Block* __restrict__ x, int n_in) {
    const int lane = int(threadIdx.x) % WARP;
    int row = (int(blockIdx.x) * VT_WARPS + int(threadIdx.x) / WARP) * R;
    // the matrix of this row, without indexing the parameter arrays by a variable (which copies them to the stack)
    const void* w = jobs.w[0];
    float* y = jobs.y[0];
    int n_out = jobs.n_out[0];
#pragma unroll
    for (int j = 1; j < 4; ++j)
        if (j < jobs.n && row >= n_out) {
            row -= n_out;
            w = jobs.w[j];
            y = jobs.y[j];
            n_out = jobs.n_out[j];
        }
    if (row >= n_out) return;
    const int bpr = n_in / F::DIV;
    float out[R];
    vt_rows<F, R, NJ, NVL>(static_cast<const typename F::Block*>(w) + std::size_t(row) * bpr, n_out - row, bpr, x, lane, out);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (row + r < n_out) y[row + r] = out[r];
    }
}

// native_mmvq_id's products: blockIdx.y = pair p (expert ids[p], activation column p / x_div), a warp per R rows
template<typename F, int R, int NJ, int NVL>
__launch_bounds__(VT_WARPS * WARP)
__global__ void vt_mmvq_id_kernel(const uint8_t* __restrict__ w_base, std::size_t expert_bytes,
                                  const int32_t* __restrict__ ids, const Q81Block* __restrict__ x, int x_div,
                                  float* __restrict__ y, int n_in, int n_out) {
    const int lane = int(threadIdx.x) % WARP;
    const int row = (int(blockIdx.x) * VT_WARPS + int(threadIdx.x) / WARP) * R;
    if (row >= n_out) return;
    const int p = int(blockIdx.y);
    const int bpr = n_in / F::DIV;
    const auto* w = reinterpret_cast<const typename F::Block*>(w_base + std::size_t(ids[p]) * expert_bytes);
    float out[R];
    vt_rows<F, R, NJ, NVL>(w + std::size_t(row) * bpr, n_out - row, bpr, x + std::size_t(p / x_div) * (n_in / Q8K), lane, out);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; ++r)
            if (row + r < n_out) y[std::size_t(p) * n_out + row + r] = out[r];
    }
}

// STRATA_OLD_MMVQ=1: the 128-thread kernels for every single-column product (the vt layout is the default)
bool vt_on() {
    static const bool on = !(std::getenv("STRATA_OLD_MMVQ") && std::string(std::getenv("STRATA_OLD_MMVQ")) == "1") &&
                           !(std::getenv("STRATA_OLD_GLUE") && std::string(std::getenv("STRATA_OLD_GLUE")) == "1");
    return on;
}
int g_vt_rows = 0;   // 0: the measured default per shape; 1, 2, 4: forced (the harness)

// the iteration count of the 128-thread loop and the virtual threads with work in its last pass, if instantiated
// (nj = 0: not - the caller takes the old kernels)
template<typename F> void vt_shape(int n_in, int& nj, int& nvl) {
    const int bpr = n_in / F::DIV;
    nj = (bpr + F::BPI - 1) / F::BPI;
    const int last = bpr - (nj - 1) * F::BPI;   // blocks of the last pass, 1 .. BPI
    constexpr int per_v = WARP / F::T;            // blocks one virtual thread's warp covers per pass
    nvl = (last + per_v - 1) / per_v;
    if (nj > 3 || nvl > 2) nj = 0;   // instantiated: the shapes where it pays (vt_pays: the last pass under half full)
}

template<typename F, int R, int NJ, int NVL>
void launch_vt_r(const VtJobs& jobs, int rows, const Q81Block* x, int n_in, cudaStream_t s) {
    const unsigned blocks = unsigned(((rows + R - 1) / R + VT_WARPS - 1) / VT_WARPS);
    vt_mmvq_kernel<F, R, NJ, NVL><<<blocks, VT_WARPS * WARP, 0, s>>>(jobs, x, n_in);
}
template<typename F, int NJ, int NVL>
void launch_vt_nv(const VtJobs& jobs, int rows, int R, const Q81Block* x, int n_in, cudaStream_t s) {
    if (R == 2) launch_vt_r<F, 2, NJ, NVL>(jobs, rows, x, n_in, s);
    else if (R == 4) launch_vt_r<F, 4, NJ, NVL>(jobs, rows, x, n_in, s);
    else launch_vt_r<F, 1, NJ, NVL>(jobs, rows, x, n_in, s);
}
template<typename F, int NJ>
void launch_vt_nj(const VtJobs& jobs, int rows, int R, int nvl, const Q81Block* x, int n_in, cudaStream_t s) {
    if (nvl == 1) launch_vt_nv<F, NJ, 1>(jobs, rows, R, x, n_in, s);
    else launch_vt_nv<F, NJ, 2>(jobs, rows, R, x, n_in, s);
}

// false: no kernel for this shape (the caller takes the old path)
template<typename F>
bool launch_vt(const VtJobs& jobs, const void* x_q8_1, int n_in, cudaStream_t s) {
    int nj, nvl;
    vt_shape<F>(n_in, nj, nvl);
    if (!nj) return false;
    int rows = 0;
    int R = g_vt_rows ? g_vt_rows : 1;
    for (int j = 0; j < jobs.n; ++j) {
        rows += jobs.n_out[j];
        if (jobs.n_out[j] % R) R = 1;   // a warp's rows never straddle two matrices
    }
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    switch (nj) {
    case 1: launch_vt_nj<F, 1>(jobs, rows, R, nvl, x, n_in, s); break;
    case 2: launch_vt_nj<F, 2>(jobs, rows, R, nvl, x, n_in, s); break;
    default: launch_vt_nj<F, 3>(jobs, rows, R, nvl, x, n_in, s); break;
    }
    return true;
}

template<typename F, int R, int NJ, int NVL>
void launch_vt_id_r(const uint8_t* w, std::size_t expert_bytes, const int32_t* ids, int n_pairs, const Q81Block* x,
                    int x_div, float* y, int n_in, int n_out, cudaStream_t s) {
    const int warps = (n_out + R - 1) / R;
    const dim3 grid(unsigned((warps + VT_WARPS - 1) / VT_WARPS), unsigned(n_pairs));
    vt_mmvq_id_kernel<F, R, NJ, NVL><<<grid, VT_WARPS * WARP, 0, s>>>(w, expert_bytes, ids, x, x_div, y, n_in, n_out);
}
template<typename F, int NJ, int NVL>
void launch_vt_id_nv(int R, const uint8_t* w, std::size_t expert_bytes, const int32_t* ids, int n_pairs,
                     const Q81Block* x, int x_div, float* y, int n_in, int n_out, cudaStream_t s) {
    if (R == 2) launch_vt_id_r<F, 2, NJ, NVL>(w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s);
    else if (R == 4) launch_vt_id_r<F, 4, NJ, NVL>(w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s);
    else launch_vt_id_r<F, 1, NJ, NVL>(w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s);
}
template<typename F, int NJ>
void launch_vt_id_nj(int R, int nvl, const uint8_t* w, std::size_t expert_bytes, const int32_t* ids, int n_pairs,
                     const Q81Block* x, int x_div, float* y, int n_in, int n_out, cudaStream_t s) {
    if (nvl == 1) launch_vt_id_nv<F, NJ, 1>(R, w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s);
    else launch_vt_id_nv<F, NJ, 2>(R, w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s);
}

template<typename F>
bool launch_vt_id(const void* w_base, std::size_t expert_bytes, const int32_t* ids, int n_pairs, const void* x_q8_1,
                  int x_div, float* y, int n_in, int n_out, cudaStream_t s) {
    int nj, nvl;
    vt_shape<F>(n_in, nj, nvl);
    if (!nj) return false;
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    const auto* w = static_cast<const uint8_t*>(w_base);
    const int R = g_vt_rows ? g_vt_rows : (nj == 1 ? 2 : 1);   // measured: K = 704 2 rows a warp, K = 2816 1
    switch (nj) {
    case 1: launch_vt_id_nj<F, 1>(R, nvl, w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s); break;
    case 2: launch_vt_id_nj<F, 2>(R, nvl, w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s); break;
    default: launch_vt_id_nj<F, 3>(R, nvl, w, expert_bytes, ids, n_pairs, x, x_div, y, n_in, n_out, s); break;
    }
    return true;
}
