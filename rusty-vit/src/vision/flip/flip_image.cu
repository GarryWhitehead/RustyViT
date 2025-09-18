#include "cuda.h"

#define HORIZ_FLIP_OP(TYPE, KERNEL_NAME) \
extern "C" __global__ void KERNEL_NAME(TYPE* src, int width, int height) \
{ \
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; \
\
    int halfHeight = height >> 1;\
\
    if (x >= width || y >= halfHeight)\
    {\
        return;\
    }\
\
    int yIdxBottom = height - y - 1;\
\
    TYPE tmp = src[x + width * yIdxBottom];\
    src[x + width * yIdxBottom] = src[x + width * y];\
    src[x + width * y] = tmp;\
}

HORIZ_FLIP_OP(uint8_t, flip_horiz_u8_kernel)
HORIZ_FLIP_OP(uint16_t, flip_horiz_u16_kernel)
HORIZ_FLIP_OP(float, flip_horiz_f32_kernel)

#define VERT_FLIP_OP(TYPE, KERNEL_NAME) \
__global__ void flip_vert_kernel(TYPE* src, int width, int height) \
{ \
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; \
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y; \
\
    int halfWidth = width >> 1; \
\
    if (x >= halfWidth || y >= height) \
    { \
        return; \
    } \
\
    int leftIdx = y * width + x; \
    int rightIdx = (width - x - 1) + y * width; \
\
    TYPE tmp = src[rightIdx]; \
    src[rightIdx] = src[leftIdx]; \
    src[leftIdx] = tmp; \
}

VERT_FLIP_OP(uint8_t, flip_vert_u8_kernel)
VERT_FLIP_OP(uint16_t, flip_vert_u16_kernel)
VERT_FLIP_OP(float, flip_vert_f32_kernel)