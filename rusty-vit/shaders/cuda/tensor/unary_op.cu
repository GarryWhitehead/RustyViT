#include "cuda.h"
#include "cuda_fp16.h"

__device__ __forceinline__ float sqrOp(const float x)
{
    return x * x;
}

__device__ __forceinline__ float sqrtOp(const float x)
{
    return sqrtf(x);
}

__device__ __forceinline__ float expOp(const float x)
{
    return expf(x);
}

__device__ __forceinline__ float tanhOp(const float x)
{
    return tanhf(x);
}

__device__ __forceinline__ float cosOp(const float x)
{
    return cosf(x);
}

__device__ __forceinline__ float sinOp(const float x)
{
    return sinf(x);
}

__device__ __forceinline__ float absOp(const float x)
{
    return fabsf(x);
}

__device__ __forceinline__ float geluOp(const float x)
{
    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
    return 0.5 * x * (1.0 + tanhf(SQRT_2_OVER_PI * (x + 0.044715 * powf(x, 3.0))));
}

__device__ __forceinline__ float reluOp(const float x)
{
    return fmaxf(x, 0);
}

__device__ __forceinline__ float logOp(const float x)
{
    return logf(x);
}

__device__ __forceinline__ float floorOp(const float x)
{
    return floorf(x);
}

__device__ __forceinline__ float ceilOp(const float x)
{
    return ceilf(x);
}

#define UNARY_OP(TYPE, KERNEL_NAME, OP) \
extern "C" __global__ void KERNEL_NAME(const TYPE* a, TYPE* out, int xsize) \
{ \
    const int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    if (idx >= xsize) \
    { \
        return; \
    } \
    out[idx] = (TYPE)OP((float)a[idx]); \
}

UNARY_OP(float, unary_op_sqr_f32, sqrOp)
UNARY_OP(__half, unary_op_sqr_f16, sqrOp)

UNARY_OP(float, unary_op_sqrt_f32, sqrtOp)
UNARY_OP(__half, unary_op_sqrt_f16, sqrtOp)

UNARY_OP(float, unary_op_exp_f32, expOp)
UNARY_OP(__half, unary_op_exp_f16, expOp)

UNARY_OP(float, unary_op_tanh_f32, tanhOp)
UNARY_OP(__half, unary_op_tanh_f16, tanhOp)

UNARY_OP(float, unary_op_cos_f32, cosOp)
UNARY_OP(__half, unary_op_cos_f16, cosOp)

UNARY_OP(float, unary_op_sin_f32, sinOp)
UNARY_OP(__half, unary_op_sin_f16, sinOp)

UNARY_OP(float, unary_op_abs_f32, absOp)
UNARY_OP(__half, unary_op_abs_f16, absOp)

UNARY_OP(float, unary_op_gelu_f32, geluOp)
UNARY_OP(__half, unary_op_gelu_f16, geluOp)

UNARY_OP(float, unary_op_relu_f32, reluOp)
UNARY_OP(__half, unary_op_relu_f16, reluOp)

UNARY_OP(float, unary_op_log_f32, logOp)
UNARY_OP(__half, unary_op_log_f16, logOp)

UNARY_OP(float, unary_op_floor_f32, floorOp)
UNARY_OP(__half, unary_op_floor_f16, floorOp)

UNARY_OP(float, unary_op_ceil_f32, ceilOp)
UNARY_OP(__half, unary_op_ceil_f16, ceilOp)