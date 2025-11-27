#include "cuda.h"
#include "cuda_fp16.h"

template <typename TYPE>
__device__ void im2col(const TYPE* input, int xsize, int batch_size, int in_channels,
    int ksize, int out_h, int out_w, int in_h, int in_w, int stride, int padding, TYPE* out)
{
    // = KW*KH*OW
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= xsize)
    {
        return;
    }
    int idx = i;
    const int ow = idx % out_w;
    idx /= out_w;
    const int kh = idx % ksize;
    idx /= ksize;
    const int kw = idx % ksize;

    // = OH
    const int oh = blockIdx.y;

    // = B*C
    const int batch = blockIdx.z / in_channels;
    const int ic = blockIdx.z % in_channels;

    const int row = oh * stride + kh - padding;
    const int col = ow * stride + kw - padding;

    const int kstep = ksize * ksize;
    const int ostep = kstep * out_h * out_w;
    const int ib_offset = batch * in_channels * in_h * in_w;    
    const int ob_offset = batch * in_channels * ostep;
    const int out_idx = ic * ostep + (oh * out_w * kstep) + (ow * kstep) + (kh * ksize + kw);

    if (row < 0 || col < 0 || row >= in_h || col >= in_w)
    {
        out[ob_offset + out_idx] = static_cast<TYPE>(0.0);
    }   
    else
    {  
        const int in_idx = (ic * in_w * in_h) + row * in_w + col;
        out[ob_offset + out_idx] = input[ib_offset + in_idx];
    }
}

#define IM2COL_OP(KERNEL_NAME, TYPE) \
extern "C" __global__ void KERNEL_NAME(const TYPE* input, int xsize, int batch_size, int in_channels, \
    int ksize, int out_h, int out_w, int in_h, int in_w, int stride, int padding, TYPE* out) \
{ \
    im2col(input, xsize, batch_size, in_channels, ksize, out_h, out_w, in_h, in_w, stride, padding, out); \
}

IM2COL_OP(im2col_f32_kernel, float)
IM2COL_OP(im2col_f16_kernel, __half)
