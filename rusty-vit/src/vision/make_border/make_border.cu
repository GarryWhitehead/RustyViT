#include "cuda.h"

__device__ __forceinline__ bool constantBorder(int pos, int srcDim, int padding, int* outIdx)
{
    return false;
}

__device__ __forceinline__ bool clampToEdgeBorder(int pos, int srcDim, int padding, int* outIdx)
{
    if (pos - padding < 0)
    {
        *outIdx = 0;
        return true;
    }
    else if (pos >= srcDim)
    {
        *outIdx = srcDim - 1;
        return true;
    }
    return false;
}

__device__ __forceinline__ bool mirrorBorder(int pos, int srcDim, int padding, int* outIdx)
{
    if (pos - padding < 0)
    {
        *outIdx = padding - abs(pos);
        return true;
    }
    else if (pos >= srcDim)
    {
        *outIdx = srcDim - (pos - srcDim) - 1;
        return true;
    }
    return false;
}

#define MAKE_BORDER_OP(TYPE, KERNEL_NAME, BORDER_OP) \
__global__ void KERNEL_NAME(const TYPE* src, int srcWidth, int srcHeight, int padding, TYPE* dst) \
{ \
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; \
 \
    if (x >= srcWidth || y >= srcHeight) \
    {\
        return; \
    } \
 \
    int newWidth = srcWidth + 2 * padding; \
    int newHeight = srcHeight + 2 * padding; \
 \
    int bPos; \
    bool res = BORDER_OP(x, srcWidth, padding, &bPos); \
    if (res) \
    { \
        dst[x + newWidth * y] = src[bPos + srcWidth * y]; \
    } \
    res = BORDER_OP(y, srcHeight, padding, &bPos); \
    if (res) \
    { \
        dst[x + newWidth * y] = src[x + srcWidth * bPos]; \
    } \
 \
    dst[x + padding + newWidth * y + padding] = src[x + srcWidth * y]; \
}

MAKE_BORDER_OP(uint8_t, make_constant_border_kernel_u8, constantBorder)
MAKE_BORDER_OP(uint8_t, make_clamped_border_kernel_u8, clampToEdgeBorder)
MAKE_BORDER_OP(uint8_t, make_mirror_border_kernel_u8, mirrorBorder)

MAKE_BORDER_OP(uint16_t, make_constant_border_kernel_u16, constantBorder)
MAKE_BORDER_OP(uint16_t, make_clamped_border_kernel_u16, clampToEdgeBorder)
MAKE_BORDER_OP(uint16_t, make_mirror_border_kernel_u16, mirrorBorder)

MAKE_BORDER_OP(float, make_constant_border_kernel_f32, constantBorder)
MAKE_BORDER_OP(float, make_clamped_border_kernel_f32, clampToEdgeBorder)
MAKE_BORDER_OP(float, make_mirror_border_kernel_f32, mirrorBorder)