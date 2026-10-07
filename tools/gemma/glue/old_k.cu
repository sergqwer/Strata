// tools/gemma/glue/old_k.cu - the baseline Gemma kernels (baseline/kernels.cu = src/gemma/kernels.cu of strata-gemma
// gemma 8c7651c, byte for byte) compiled in namespace strata::gemma::old_k, so the harness runs the old and the new
// code side by side on the same inputs. Built with -I baseline/include ahead of -I include (the baseline header).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>

#define k old_k
#include "baseline/kernels.cu"
#undef k
