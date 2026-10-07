// standalone harness: the production vit_fa2_kernel vs new attention kernels, accuracy vs an FP32 (double-sum)
// reference and timing (min over reps). Build: nvcc -O3 -arch=sm_86 -use_fast_math -std=c++17 harness.cu -o harness
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <random>
#include <vector>
#include <string>
#include <algorithm>
#include <cstring>

using bf16 = __nv_bfloat16;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d: %s\n", cudaGetErrorString(e_), __FILE__, __LINE__, #x); exit(1); } } while (0)

#include "cur.cuh"
#include "fa3.cuh"
#include "fa4.cuh"
#include "prod.cuh"

constexpr int H = 16, HD = 72, HDP = 80;

// reference: one block per (subset row, head, image); scores in f32 from the same f16 inputs, softmax / sums in double
__global__ void ref_kernel(const __half* Q, const __half* K, const __half* V, float* R, int n, int rs, int nsub) {
    extern __shared__ double sh[];   // [n] scores, then probabilities
    __shared__ float q[HD];
    __shared__ double red[128];
    const int si = blockIdx.x, h = blockIdx.y, z = blockIdx.z, row = si * rs;
    const __half* Qh = Q + ((size_t) z * H + h) * n * HDP;
    const __half* Kh = K + ((size_t) z * H + h) * n * HDP;
    const __half* Vh = V + ((size_t) z * H + h) * n * HDP;
    if (threadIdx.x < HD) q[threadIdx.x] = __half2float(Qh[(size_t) row * HDP + threadIdx.x]);
    __syncthreads();
    float mx = -INFINITY;
    for (int k = threadIdx.x; k < n; k += blockDim.x) {
        double s = 0;
        for (int d = 0; d < HD; ++d) s += (double) q[d] * (double) __half2float(Kh[(size_t) k * HDP + d]);
        sh[k] = s;
        mx = fmaxf(mx, (float) s);
    }
    red[threadIdx.x] = mx;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] = fmax(red[threadIdx.x], red[threadIdx.x + o]);
        __syncthreads();
    }
    const double m = red[0];   // (max in float precision: only a shift)
    __syncthreads();
    double sum = 0;
    for (int k = threadIdx.x; k < n; k += blockDim.x) { sh[k] = exp(sh[k] - m); sum += sh[k]; }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int o = 64; o > 0; o >>= 1) {
        if (threadIdx.x < o) red[threadIdx.x] += red[threadIdx.x + o];
        __syncthreads();
    }
    const double l = red[0];
    if (threadIdx.x < HD) {
        double acc = 0;
        for (int k = 0; k < n; ++k) acc += sh[k] * (double) __half2float(Vh[(size_t) k * HDP + threadIdx.x]);
        R[(((size_t) z * nsub + si) * H + h) * HD + threadIdx.x] = (float) (acc / l);
    }
}

__global__ void widen_kernel(const bf16* a, float* b, size_t n) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) b[i] = __bfloat162float(a[i]);
}
struct Cfg {
    std::string name;
    int qb, threads, smem;
    void (*launch)(const __half*, const __half*, const __half*, void*, int, int, bool f32, cudaStream_t);
};

template <int NW, int MT, int BK, int ST, int MINB>
void launch_fa3(const __half* q, const __half* k, const __half* v, void* o, int n, int B, bool f32, cudaStream_t s) {
    constexpr int QB = NW * 16 * MT, SM = ST * 2 * BK * 72 * 2;
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(xp::fa3_kernel<NW, MT, BK, ST, MINB, bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        CK(cudaFuncSetAttribute(xp::fa3_kernel<NW, MT, BK, ST, MINB, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        init = true;
    }
    dim3 grid((n + QB - 1) / QB, H, B);
    if (f32) xp::fa3_kernel<NW, MT, BK, ST, MINB, float><<<grid, NW * 32, SM, s>>>(q, k, v, (float*) o, n, H);
    else xp::fa3_kernel<NW, MT, BK, ST, MINB, bf16><<<grid, NW * 32, SM, s>>>(q, k, v, (bf16*) o, n, H);
}
template <int NW, int MT, int BK, int ST, int MINB>
void launch_fa4(const __half* q, const __half* k, const __half* v, void* o, int n, int B, bool f32, cudaStream_t s) {
    constexpr int QB = NW * 16 * MT, SM = ST * 2 * BK * 72 * 2;
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(xp::fa4_kernel<NW, MT, BK, ST, MINB, bf16>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        CK(cudaFuncSetAttribute(xp::fa4_kernel<NW, MT, BK, ST, MINB, float>, cudaFuncAttributeMaxDynamicSharedMemorySize, SM));
        init = true;
    }
    dim3 grid((n + QB - 1) / QB, H, B);
    if (f32) xp::fa4_kernel<NW, MT, BK, ST, MINB, float><<<grid, NW * 32, SM, s>>>(q, k, v, (float*) o, n, H);
    else xp::fa4_kernel<NW, MT, BK, ST, MINB, bf16><<<grid, NW * 32, SM, s>>>(q, k, v, (bf16*) o, n, H);
}
void launch_prod(const __half* q, const __half* k, const __half* v, void* o, int n, int B, bool f32, cudaStream_t s) {
    // the vision.cu kernel writes bf16 only: the f32 slot gets the bf16 result widened (compare its bf16 numbers)
    static bf16* tmp = nullptr;
    static size_t cap = 0;
    const size_t no = (size_t) B * n * H * HD;
    bf16* dst = (bf16*) o;
    if (f32) {
        if (cap < no) { if (tmp) cudaFree(tmp); CK(cudaMalloc(&tmp, no * 2)); cap = no; }
        dst = tmp;
    }
    prod::vit_fa3_kernel<HDP><<<dim3((n + prod::FA3_BQ - 1) / prod::FA3_BQ, H, B), prod::FA3_THREADS, 0, s>>>(q, k, v, dst, n, H);
    if (f32) widen_kernel<<<(unsigned) ((no + 255) / 256), 256, 0, s>>>(tmp, (float*) o, no);
}
void launch_cur(const __half* q, const __half* k, const __half* v, void* o, int n, int B, bool f32, cudaStream_t s) {
    dim3 grid((n + 63) / 64, H, B);
    if (f32) cur_fa2_kernel<HDP, float><<<grid, 128, 0, s>>>(q, k, v, (float*) o, n, H, HD);
    else cur_fa2_kernel<HDP, bf16><<<grid, 128, 0, s>>>(q, k, v, (bf16*) o, n, H, HD);
}

#define FA4(NW, MT, BK, ST, MINB) Cfg{"fa4 w" #NW " m" #MT " bk" #BK " st" #ST " b" #MINB, NW * 16 * MT, NW * 32, ST * 2 * BK * 144, launch_fa4<NW, MT, BK, ST, MINB>}
#define FA3(NW, MT, BK, ST, MINB) Cfg{"fa3 w" #NW " m" #MT " bk" #BK " st" #ST " b" #MINB, NW * 16 * MT, NW * 32, ST * 2 * BK * 144, launch_fa3<NW, MT, BK, ST, MINB>}

static float bf2f(bf16 x) { return __bfloat162float(x); }

int main(int argc, char** argv) {
    const float sig = argc > 1 ? atof(argv[1]) : 1.0f;   // std of q / k entries
    const int reps = argc > 2 ? atoi(argv[2]) : 25;
    const std::string only = argc > 3 ? argv[3] : "";
    std::vector<Cfg> cfgs = {
        {"current fa2", 64, 128, 0, launch_cur},
        FA3(4, 2, 64, 2, 2),
        {"vision.cu vit_fa3", 128, 128, 0, launch_prod},
    };
    struct Shape { int B, n, rs; };
    std::vector<Shape> shapes = {{1, 4959, 13}, {9, 630, 5}};
    if (argc > 4) {   // "B:n,B:n,..." (every row checked against the reference)
        shapes.clear();
        for (char* tok = strtok(argv[4], ","); tok; tok = strtok(nullptr, ",")) {
            int b, nn;
            if (sscanf(tok, "%d:%d", &b, &nn) == 2) shapes.push_back({b, nn, 1});
        }
    }
    cudaStream_t st;
    CK(cudaStreamCreate(&st));
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    std::mt19937 rng(1234);
    for (const Shape& sh : shapes) {
        const int B = sh.B, n = sh.n, nsub = (n + sh.rs - 1) / sh.rs;
        const size_t ne = (size_t) B * H * n * HDP, no = (size_t) B * n * H * HD;
        std::vector<__half> hq(ne), hk(ne), hv(ne);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (size_t i = 0; i < ne; ++i) {
            const bool pad = (i % HDP) >= HD;
            hq[i] = __float2half(pad ? 0.f : sig * nd(rng));
            hk[i] = __float2half(pad ? 0.f : sig * nd(rng));
            hv[i] = __float2half(pad ? 0.f : nd(rng));
        }
        __half *dq, *dk, *dv;
        void *of_cur, *of_new, *ob_cur, *ob_new;
        float* dref;
        CK(cudaMalloc(&dq, ne * 2)); CK(cudaMalloc(&dk, ne * 2)); CK(cudaMalloc(&dv, ne * 2));
        CK(cudaMalloc(&of_cur, no * 4)); CK(cudaMalloc(&of_new, no * 4));
        CK(cudaMalloc(&ob_cur, no * 2)); CK(cudaMalloc(&ob_new, no * 2));
        CK(cudaMalloc(&dref, (size_t) B * nsub * H * HD * 4));
        CK(cudaMemcpy(dq, hq.data(), ne * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dk, hk.data(), ne * 2, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dv, hv.data(), ne * 2, cudaMemcpyHostToDevice));
        CK(cudaFuncSetAttribute(ref_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, n * 8));
        ref_kernel<<<dim3(nsub, H, B), 128, n * 8, st>>>(dq, dk, dv, dref, n, sh.rs, nsub);
        CK(cudaStreamSynchronize(st));
        std::vector<float> ref((size_t) B * nsub * H * HD);
        CK(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost));
        double rr = 0;
        for (float x : ref) rr += (double) x * x;
        rr = sqrt(rr / ref.size());
        const double flop = 4.0 * n * (double) n * HD * H * B;
        printf("=== B=%d n=%d H=%d hd=%d sigma_qk=%.2f  (%.1f GFLOP, ref rms %.4f) ===\n", B, n, H, HD, sig, flop / 1e9, rr);

        auto vs_ref_f = [&](const std::vector<float>& o, double& mx, double& rms) {
            mx = 0; rms = 0;
            for (int z = 0; z < B; ++z)
                for (int si = 0; si < nsub; ++si)
                    for (int hh = 0; hh < H; ++hh)
                        for (int d = 0; d < HD; ++d) {
                            const double a = o[(((size_t) z * n + si * sh.rs) * H + hh) * HD + d];
                            const double r = ref[(((size_t) z * nsub + si) * H + hh) * HD + d];
                            const double e = fabs(a - r);
                            if (!(e <= mx)) mx = e;   // NaN-propagating
                            rms += e * e;
                        }
            rms = sqrt(rms / ref.size());
        };
        std::vector<float> fcur(no), fnew(no);
        std::vector<bf16> bcur(no), bnew(no);
        cfgs[0].launch(dq, dk, dv, of_cur, n, B, true, st);
        cfgs[0].launch(dq, dk, dv, ob_cur, n, B, false, st);
        CK(cudaStreamSynchronize(st));
        CK(cudaGetLastError());
        CK(cudaMemcpy(fcur.data(), of_cur, no * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(bcur.data(), ob_cur, no * 2, cudaMemcpyDeviceToHost));
        std::vector<float> tmp(no);
        for (size_t i = 0; i < no; ++i) tmp[i] = bf2f(bcur[i]);
        double mxf, rmsf, mxb, rmsb;
        vs_ref_f(fcur, mxf, rmsf);
        vs_ref_f(tmp, mxb, rmsb);
        printf("  %-26s vs fp32 ref: f32-out max %.3e rms %.3e | bf16-out max %.3e rms %.3e\n", cfgs[0].name.c_str(), mxf, rmsf, mxb, rmsb);
        for (size_t c = 1; c < cfgs.size(); ++c) {
            if (!only.empty() && cfgs[c].name.find(only) == std::string::npos) continue;
            CK(cudaMemset(of_new, 0xff, no * 4));
            CK(cudaMemset(ob_new, 0xff, no * 2));
            cfgs[c].launch(dq, dk, dv, of_new, n, B, true, st);
            cfgs[c].launch(dq, dk, dv, ob_new, n, B, false, st);
            CK(cudaStreamSynchronize(st));
            CK(cudaGetLastError());
            CK(cudaMemcpy(fnew.data(), of_new, no * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpy(bnew.data(), ob_new, no * 2, cudaMemcpyDeviceToHost));
            for (size_t i = 0; i < no; ++i) tmp[i] = bf2f(bnew[i]);
            vs_ref_f(fnew, mxf, rmsf);
            vs_ref_f(tmp, mxb, rmsb);
            double dmax = 0, dmaxb = 0, drel = 0;
            size_t nbad = 0;
            for (size_t i = 0; i < no; ++i) {
                const double a = fnew[i], b = fcur[i];
                const double e = fabs(a - b);
                if (!(e <= dmax)) dmax = e;
                drel = std::max(drel, e / (fabs(b) + 1e-2));
                const double eb = fabs((double) bf2f(bnew[i]) - bf2f(bcur[i]));
                if (!(eb <= dmaxb)) dmaxb = eb;
                if (!std::isfinite(a)) ++nbad;
            }
            printf("  %-26s vs fp32 ref: f32-out max %.3e rms %.3e | bf16-out max %.3e rms %.3e | vs current: f32 max %.3e rel %.3e, bf16 max %.3e, nonfinite %zu\n",
                   cfgs[c].name.c_str(), mxf, rmsf, mxb, rmsb, dmax, drel, dmaxb, nbad);
        }
        // timing, interleaved, bf16 output (production)
        std::vector<float> best(cfgs.size(), 1e9f);
        for (int r = 0; r < reps; ++r)
            for (size_t c = 0; c < cfgs.size(); ++c) {
                if (c && !only.empty() && cfgs[c].name.find(only) == std::string::npos) continue;
                CK(cudaEventRecord(e0, st));
                cfgs[c].launch(dq, dk, dv, ob_new, n, B, false, st);
                CK(cudaEventRecord(e1, st));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best[c] = std::min(best[c], ms);
            }
        for (size_t c = 0; c < cfgs.size(); ++c) {
            if (best[c] > 1e8f) continue;
            printf("  %-26s %8.3f ms  %5.1f TFLOPS  (x%.2f)\n", cfgs[c].name.c_str(), best[c], flop / best[c] / 1e9, best[0] / best[c]);
        }
        cudaFree(dq); cudaFree(dk); cudaFree(dv); cudaFree(of_cur); cudaFree(of_new); cudaFree(ob_cur); cudaFree(ob_new); cudaFree(dref);
    }
    return 0;
}
