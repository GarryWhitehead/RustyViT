#include "cuda.h"

template <typename TYPE>
__device__ __forceinline__ TYPE roundNearest(float value)
{
    return value;
}

template <>
__device__ __forceinline__ uint8_t roundNearest(float value)
{
    return __float2uint_rn(value);
}

template <>
__device__ __forceinline__ uint16_t roundNearest(float value)
{
    return __float2uint_rn(value);
}

__device__ __forceinline__ float lerp(float a, float b, float fract) 
{
    return fma(fract, b, fma(-fract, a, a));
}

template <typename TYPE>
__device__ float getPixel(const TYPE *src, float x, float y, int width, int height, int stride) 
{
    // If the xy co-ords are outside the bounds of the src image, then clamp to border.
    const float xPos = x < 0.0 ? 0.0 : x >= width ? width - 1 : x;
    const float yPos = y < 0.0 ? 0.0 : y >= height ? height - 1 : y;
    return static_cast<float>(src[__float2int_rd(xPos) + stride * __float2int_rd(yPos)]);
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

    return roundNearest<TYPE>(
        lerp(lerp(p0, p1, fractX), lerp(p2, p3, fractX), fractY));
}

#define RESIZE_OP(TYPE, FILTER_OP, KERNEL_NAME) \
extern "C" __global__ void KERNEL_NAME(const TYPE *src, int srcWidth, int srcHeight, int dstWidth, int dstHeight, float scaleX, float scaleY, TYPE *dst) \
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