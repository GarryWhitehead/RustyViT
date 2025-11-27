#include "cuda.h"
#include "cuda_fp16.h"

template <typename TYPE>
__device__ void nchw_to_nhwc(const TYPE* input, int xsize, int batch_size, int channels, int height, int width, TYPE* out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= xsize)
    {
        return;
    }
    const int x = idx % width;
    const int y = idx / width;

    const int batch = blockIdx.z;
    const int ic = blockIdx.y;

    const int ib_offset = batch * channels * height * width;
    const int ob_offset = batch * height * width * channels;
    const int ic_offset = ib_offset + ic * height * width;
    const int oc_offset = ob_offset + ic;

    const int i_row = ic_offset + y * width;
    const int o_row = oc_offset + y * width * channels;

     out[o_row + x * channels] = input[i_row + x];
}

#define NCHW_TO_NHWC_OP(KERNEL_NAME, TYPE) \
extern "C" __global__ void KERNEL_NAME(const TYPE* input, int xsize, int batch_size, int channels, int height, int width, TYPE* out) \
{ \
    nchw_to_nhwc(input, xsize, batch_size, channels, height, width, out); \
}

NCHW_TO_NHWC_OP(nchw_to_nhwc_f32_kernel, float)
NCHW_TO_NHWC_OP(nchw_to_nhwc_f16_kernel, __half)

template <typename TYPE>
__device__ void nhwc_to_nchw(const TYPE* input, int xsize, int batch_size, int channels, int height, int width, TYPE* out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= xsize)
    {
        return;
    }
    const int x = idx % width;
    const int y = idx / width;

    const int batch = blockIdx.z;
    const int ic = blockIdx.y;

    const int ib_offset = batch * height * width * channels;
    const int ob_offset = batch * channels * height * width;

    const int i_row = ib_offset + y * width * channels;
    const int o_row = ob_offset + y * width;

    const int i_col = i_row + x * channels;
    const int o_col = o_row + x;

    out[o_col + ic * width * height] = input[i_col + ic];
}

#define NHWC_TO_NCHW_OP(KERNEL_NAME, TYPE) \
extern "C" __global__ void KERNEL_NAME(const TYPE* input, int xsize, int batch_size, int channels, int height, int width, TYPE* out) \
{ \
    nhwc_to_nchw(input, xsize, batch_size, channels, height, width, out); \
}

NHWC_TO_NCHW_OP(nhwc_to_nchw_f32_kernel, float)
NHWC_TO_NCHW_OP(nhwc_to_nchw_f16_kernel, __half)