// tools/gemma/glue/old_mmvq.cu - the baseline native_mmvq (baseline/native_mmvq.cu = src/kernels/cuda/native_mmvq.cu
// of strata-gemma gemma 8c7651c, byte for byte) in namespace strata::old_kernels.
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

#define kernels old_kernels
#include "baseline/native_mmvq.cu"
namespace strata::kernels {   // (old_kernels) the i-quants are not used by the harness
void iq_mmvq(int, const void*, const void*, float*, int, int, int, void*) { throw std::runtime_error("old iq_mmvq"); }
size_t iq_row_bytes(int, int64_t) noexcept { return 0; }
}  // namespace strata::kernels
#undef kernels
