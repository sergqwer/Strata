// tools/gemma/glue/old.hpp - declarations of the baseline kernels (old_k.cu, old_mmvq.cu): the baseline headers under
// the renamed namespaces.
#pragma once
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#define k old_k
#include "baseline/include/strata/gemma/kernels.hpp"
#undef k
#define kernels old_kernels
#include "baseline/include/strata/kernels/native_mmvq.hpp"
#undef kernels
