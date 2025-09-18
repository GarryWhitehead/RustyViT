#include "cuda.h"

__device__ __forceinline__ float lerp(float a, float b, float fract) 
{
    return fma(fract, b, fma(-fract, a, a));
}

template <typename TYPE>
__device__ float getPixel(const TYPE *src, float x, float y, int width, int height, int stride) 
{
    // If the xy co-ords are outside the bounds of the src image, return a
    // constant value.
    if (x < 0.0 || x >= width || y < 0.0 || y >= height) 
    {
        return 0.0;
    }
    return static_cast<float>(src[static_cast<int>(x) + stride * static_cast<int>(y)]);
}

template <typename TYPE>
__device__ __forceinline__ TYPE biLinearOp(const TYPE *src, int srcWidth, int srcHeight, float x, float y, float2 scale) 
{
    const float scaledX = (x + 0.5) * scale.x - 0.5;
    const float scaledY = (y + 0.5) * scale.y - 0.5;

    // Floor the coordinates to get to the nearest valid pixel.
    const int px = __float2int_rd(scaledX);
    const int py = __float2int_rd(scaledY);

    // Set weights of pixels according to distance from the actual location.
    const float fractX = scaledX - px;
    const float fractY = scaledY - py;

    auto p0 = getPixel(src, px, py, srcWidth, srcHeight, srcWidth);
    auto p1 = getPixel(src, px + 1, py, srcWidth, srcHeight, srcWidth);
    auto p2 = getPixel(src, px, py + 1, srcWidth, srcHeight, srcWidth);
    auto p3 = getPixel(src, px + 1, py + 1, srcWidth, srcHeight, srcWidth);

    return static_cast<TYPE>(
        lerp(lerp(p0, p1, fractX), lerp(p2, p3, fractX), fractY));
}

#define RESIZE_OP(TYPE, FILTER_OP, KERNEL_NAME) \
__global__ void KERNEL_NAME(const TYPE *src, int srcWidth, int srcHeight, int dstWidth, int dstHeight, float scaleX, float scaleY, TYPE *dst) \
{ \
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;\
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;\
\
    if (x >= dstWidth || y >= dstHeight) \
    {\
        return;\
    }\
\
    const TYPE p = FILTER_OP(src, srcWidth, srcHeight, x, y, make_float2(scaleX, scaleY));\
    dst[x + dstWidth * y] = p;\
}

RESIZE_OP(uint8_t, biLinearOp, bilinear_resize_kernel_u8)
RESIZE_OP(uint16_t, biLinearOp, bilinear_resize_kernel_u16)
RESIZE_OP(float, biLinearOp, bilinear_resize_kernel_f32)