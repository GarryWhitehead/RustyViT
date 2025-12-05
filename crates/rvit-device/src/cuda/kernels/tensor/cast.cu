#include "cuda.h"
#include "cuda_fp16.h"

#define CAST_OP(KERNEL_NAME, SRC_TYPE, DST_TYPE, CAST_FUNC) \
extern "C" __global__ void KERNEL_NAME(const SRC_TYPE* src, DST_TYPE* dst, int size) \
{ \
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; \
 \
    for (unsigned int i = x; i < size; i += blockDim.x * gridDim.x) \
    { \
        dst[i] = CAST_FUNC(src[i]); \
    } \
}

CAST_OP(cast_f32_f16_kernel, float, __half, __float2half)
CAST_OP(cast_f16_f32_kernel, __half, float, __half2float)

