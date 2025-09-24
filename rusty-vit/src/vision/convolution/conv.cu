#include <cuda.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

/// The tile width for the horizontal pass.
#define HORIZ_TILE_WIDTH 16
/// The tile height for the horizontal pass.
#define HORIZ_TILE_HEIGHT 4
#define HORIZ_HALO_STEPS 1
#define HORIZ_RESULT_STEPS 8 //1
/// The tile width for the vertical pass.
#define VERT_TILE_WIDTH 16
/// The tile height for the vertical pass.
#define VERT_TILE_HEIGHT 8
#define VERT_HALO_STEPS 1
#define VERT_RESULT_STEPS 8 //2

// TODO: roundNearest and getPixel are duplicated - move to a common cuh file.
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

template <typename Type>
__device__ void applyHorizKernel(
    const Type* input,
    Type* output,
    unsigned int imageWidth,
    unsigned int imageHeight,
    int kernelDim,
    const float* kernelX)
{
    cg::thread_block threadBlock = cg::this_thread_block();
    __shared__ Type
        sData[HORIZ_TILE_HEIGHT][(HORIZ_RESULT_STEPS + 2 * HORIZ_HALO_STEPS) * HORIZ_TILE_WIDTH];

    // Offset to the left halo edge.
    const int baseX = (blockIdx.x * HORIZ_RESULT_STEPS - HORIZ_HALO_STEPS) * HORIZ_TILE_WIDTH + threadIdx.x;
    const int baseY = blockIdx.y * HORIZ_TILE_HEIGHT + threadIdx.y;

    const int offset = baseX + imageWidth * baseY;

    // Load the main image data and apron boundary values into shared memory.
#pragma unroll
    for (int i = HORIZ_HALO_STEPS; i < HORIZ_RESULT_STEPS + HORIZ_HALO_STEPS; ++i)
    {
        sData[threadIdx.y][threadIdx.x + i * HORIZ_TILE_WIDTH] = input[offset + (i * HORIZ_TILE_WIDTH)];
    }
    #pragma unroll
    for (int i = 0; i < HORIZ_HALO_STEPS; ++i) {
        sData[threadIdx.y][threadIdx.x + i * HORIZ_TILE_WIDTH] =
            (baseX >= -i * HORIZ_TILE_WIDTH) ? input[offset + (i * HORIZ_TILE_WIDTH)] : 0;
    }
    #pragma unroll
    for (int i = HORIZ_HALO_STEPS + HORIZ_RESULT_STEPS; i < HORIZ_RESULT_STEPS + 2 * HORIZ_HALO_STEPS; ++i)
    {
        sData[threadIdx.y][threadIdx.x + i * HORIZ_TILE_WIDTH] =
            (imageWidth - baseX > i * HORIZ_TILE_WIDTH) ? input[offset + (i * HORIZ_TILE_WIDTH)] : 0;
    }
    cg::sync(threadBlock);

#pragma unroll
    for (int i = HORIZ_HALO_STEPS; i < HORIZ_HALO_STEPS + HORIZ_RESULT_STEPS; i++)
    {
        float sum = 0.0f;

#pragma unroll
        for (int j = 0; j < kernelDim; j++)
        {
            sum += kernelX[j] * static_cast<float>(sData[threadIdx.y][threadIdx.x + i * HORIZ_TILE_WIDTH 
                + (j - (kernelDim >> 1))]);
        }
        output[offset + (i * HORIZ_TILE_WIDTH)] = roundNearest<Type>(sum);
    }
}

template <typename Type>
__device__ void applyVertKernel(
    const Type* input,
    Type* output,
    unsigned int imageWidth,
    unsigned int imageHeight,
    int kernelDim,
    float* kernelY)
{
    cg::thread_block threadBlock = cg::this_thread_block();
    __shared__ Type
        sData[VERT_TILE_WIDTH][(VERT_RESULT_STEPS + 2 * VERT_HALO_STEPS) * VERT_TILE_HEIGHT + 1];

    // Offset to the upper halo edge.
    const int baseX = blockIdx.x * VERT_TILE_WIDTH + threadIdx.x;
    const int baseY = (blockIdx.y * VERT_RESULT_STEPS - VERT_HALO_STEPS) * VERT_TILE_HEIGHT + threadIdx.y;

    const int offset = baseX + imageWidth * baseY;

    // Load the main image data and surrounding apron for this tile into shared memory.
    #pragma unroll
    for (int i = VERT_HALO_STEPS; i < VERT_RESULT_STEPS + VERT_HALO_STEPS; ++i)
    {
        sData[threadIdx.x][threadIdx.y + i * VERT_TILE_HEIGHT] =
            input[offset + (i * VERT_TILE_HEIGHT * imageWidth)];
    }
    #pragma unroll
    for (int i = 0; i < VERT_HALO_STEPS; ++i) {
        sData[threadIdx.x][threadIdx.y + i * VERT_TILE_HEIGHT] =
            (baseY >= -i * VERT_TILE_HEIGHT) ? input[offset + (i * VERT_TILE_HEIGHT * imageWidth)] : 0;
    }
    #pragma unroll
    for (int i = VERT_HALO_STEPS + VERT_RESULT_STEPS; i < VERT_RESULT_STEPS + 2 * VERT_HALO_STEPS; ++i)
    {
        sData[threadIdx.x][threadIdx.y + i * VERT_TILE_HEIGHT] =
            (imageHeight - baseY > i * VERT_TILE_HEIGHT) ? input[offset + (i * VERT_TILE_HEIGHT * imageWidth)] : 0;
    }
    cg::sync(threadBlock);

#pragma unroll
    for (int i = VERT_HALO_STEPS; i < VERT_HALO_STEPS + VERT_RESULT_STEPS; i++)
    {
        float sum = 0.0f;

#pragma unroll
        for (int j = 0; j < kernelDim; j++)
        {
            sum +=
                kernelY[j] * static_cast<float>(sData[threadIdx.x][threadIdx.y + i * VERT_TILE_HEIGHT 
                    + (j - (kernelDim >> 1))]);
        }
        output[offset + (i * VERT_TILE_HEIGHT * imageWidth)] = roundNearest<Type>(sum);
    }
}

#define CONV_HORIZ_KERNEL_OP(TYPE, KERNEL_NAME) \
extern "C" __global__ void KERNEL_NAME( \
    const TYPE* input, \
    TYPE* output, \
    unsigned int imageWidth, \
    unsigned int imageHeight, \
    int kernelDim, \
    float* kernelX) \
{ \
    applyHorizKernel(input, output, imageWidth, imageHeight, kernelDim, kernelX); \
} 

#define CONV_VERT_KERNEL_OP(TYPE, KERNEL_NAME) \
extern "C" __global__ void KERNEL_NAME( \
    const TYPE* input, \
    TYPE* output, \
    unsigned int imageWidth, \
    unsigned int imageHeight, \
    int kernelDim, \
    float* kernelY) \
{ \
    applyVertKernel(input, output, imageWidth, imageHeight, kernelDim, kernelY); \
} 

CONV_HORIZ_KERNEL_OP(uint8_t, conv_horiz_kernel_u8)
CONV_HORIZ_KERNEL_OP(uint16_t, conv_horiz_kernel_u16)
CONV_HORIZ_KERNEL_OP(float, conv_horiz_kernel_f32)

CONV_VERT_KERNEL_OP(uint8_t, conv_vert_kernel_u8)
CONV_VERT_KERNEL_OP(uint16_t, conv_vert_kernel_u16)
CONV_VERT_KERNEL_OP(float, conv_vert_kernel_f32)

