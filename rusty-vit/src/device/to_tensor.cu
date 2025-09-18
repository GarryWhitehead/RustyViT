#include "cuda.h"

#define TO_TENSOR_OP(IMAGE_TYPE, TENSOR_TYPE, KERNEL_NAME) \
__global__ void KERNEL_NAME(const IMAGE_TYPE* src, int width, int height, TENSOR_TYPE mean, TENSOR_TYPE stdDev, TENSOR_TYPE* dst) \
{ \
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x; \
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y; \
 \
    if (x >= width || y >= height) { \
        return; \
    } \
 \
    auto v = static_cast<TENSOR_TYPE>(src[x + width * y]); \
    dst[x + width * y] = v - mean / stdDev; \
}

TO_TENSOR_OP(uint8_t, float, to_tensor_u8_f32_kernel)
TO_TENSOR_OP(uint8_t, double, to_tensor_u8_f64_kernel)

TO_TENSOR_OP(uint16_t, float, to_tensor_u16_f32_kernel)
TO_TENSOR_OP(uint16_t, double, to_tensor_u16_f64_kernel)

TO_TENSOR_OP(float, float, to_tensor_f32_f32_kernel)
TO_TENSOR_OP(float, double, to_tensor_f32_f64_kernel)