#include "cuda.h"

#define CROP_OP(TYPE, KERNEL_NAME) \
extern "C" __global__ void KERNEL_NAME(const TYPE* src, int srcWidth, int dstWidth, int dstHeight, int xOffset, int yOffset, TYPE* dst)\
{ \
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; \
 \
    if (x >= dstWidth || y >= dstHeight) \
    { \
        return; \
    } \
 \
    const TYPE p = src[x + xOffset + srcWidth * y + yOffset]; \
    dst[x + dstWidth * y] = p; \
} 

CROP_OP(uint8_t, crop_kernel_u8)
CROP_OP(uint16_t, crop_kernel_u16)
CROP_OP(float, crop_kernel_f32)