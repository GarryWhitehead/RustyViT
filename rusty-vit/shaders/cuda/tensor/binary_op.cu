#include "cuda.h"
#include "cuda_fp16.h"

__device__ __forceinline__ float addOp(const float a, const float b)
{
    return a + b;
}

__device__ __forceinline__ float subOp(const float a, const float b)
{
    return a - b;
}

__device__ __forceinline__ float mulOp(const float a, const float b)
{
    return a * b;
}

__device__ __forceinline__ float divOp(const float a, const float b)
{
    return a / b;
}

#define BINARY_OP(TYPE, KERNEL_NAME, OP) \
extern "C" __global__ void KERNEL_NAME(const TYPE* a, const TYPE* b, TYPE* out, int xsize) \
{ \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= xsize) \
    { \
        return; \
    } \
    out[idx] = (TYPE)OP((float)a[idx], (float)b[idx]); \
}

BINARY_OP(float, binary_op_add_f32, addOp)
BINARY_OP(__half, binary_op_add_f16, addOp)

BINARY_OP(float, binary_op_sub_f32, subOp)
BINARY_OP(__half, binary_op_sub_f16, subOp)

BINARY_OP(float, binary_op_mul_f32, mulOp)
BINARY_OP(__half, binary_op_mul_f16, mulOp)

BINARY_OP(float, binary_op_div_f32, divOp)
BINARY_OP(__half, binary_op_div_f16, divOp)