use crate::device::DeviceStorage;
use crate::device::cpu::Cpu;
use crate::tensor::Tensor;
use crate::tensor::convolution::{ConvInput, ConvShape};
use crate::tensor::matmul::matmul_cpu::MatMul;
use crate::type_traits::FloatType;

impl Cpu {
    fn im2col_nchw<T: FloatType>(
        &self,
        batch_size: usize,
        p: &ConvInput<T, Cpu>,
        in_data: &[T],
        buffer: &mut [T],
    ) {
        let mut out_idx = 0;
        for b in 0..batch_size {
            let in_offset = b * p.in_channels * p.in_width * p.in_height;
            let ksize = p.filters.shape[2];
            for c in 0..p.in_channels {
                for oh in 0..p.out_height {
                    for ow in 0..p.out_width {
                        for kh in 0..ksize {
                            for kw in 0..ksize {
                                let row = (oh * p.stride + kh).wrapping_sub(p.padding);
                                let col = (ow * p.stride + kw).wrapping_sub(p.padding);
                                if row < p.in_height && col < p.in_width {
                                    let in_idx =
                                        c * (p.in_width * p.in_height) + row * p.in_width + col;
                                    buffer[out_idx] = in_data[in_offset + in_idx];
                                }
                                out_idx += 1;
                            }
                        }
                    }
                }
            }
        }
    }

    fn im2col_nhwc<T: FloatType>(
        &self,
        batch_size: usize,
        p: &ConvInput<T, Cpu>,
        in_data: &[T],
        buffer: &mut [T],
    ) {
        let ksize = p.filters.shape[2];
        let out_channels = p.filters.shape[0];
        for b in 0..batch_size {
            let in_offset = b * p.in_channels * p.in_width * p.in_height;
            let out_offset = b * out_channels * p.out_width * p.out_height;
            for oh in 0..p.out_height {
                for ow in 0..p.out_width {
                    let row = oh * p.out_width + ow;
                    for ky in 0..ksize {
                        let iy = (p.stride * oh + ky).wrapping_sub(p.padding);
                        if iy >= p.in_height {
                            continue;
                        }
                        for kx in 0..ksize {
                            let ix = (p.stride * ow + kx).wrapping_sub(p.padding);
                            if ix >= p.in_width {
                                continue;
                            }
                            for c in 0..p.in_channels {
                                let col = c * ksize * ksize + ky * ksize + kx;
                                buffer[out_offset + row * p.in_channels * ksize * ksize + col] =
                                    in_data[in_offset + (iy * p.in_width + ix) * p.in_channels + c];
                            }
                        }
                    }
                }
            }
        }
    }

    fn nchw_to_nhwc<T: FloatType>(&mut self, in_data: &[T], shape: &[usize], out_data: &mut [T]) {
        let (batch_size, channels, height, width) = (shape[0], shape[1], shape[2], shape[3]);
        for b in 0..batch_size {
            let in_offset = b * channels * height * width;
            let out_offset = b * height * width * channels;
            for c in 0..channels {
                let ic_offset = in_offset + c * height * width;
                let oc_offset = out_offset + c;
                for h in 0..height {
                    let i_row = ic_offset + h * width;
                    let o_row = oc_offset + h * width * channels;
                    for w in 0..width {
                        let i_idx = i_row + w;
                        let o_idx = o_row + w * channels;
                        out_data[o_idx] = in_data[i_idx];
                    }
                }
            }
        }
    }

    fn nhwc_to_nchw<T: FloatType>(&mut self, in_data: &[T], shape: &[usize], out_data: &mut [T]) {
        let (batch_size, height, width, channels) = (shape[0], shape[1], shape[2], shape[3]);
        for b in 0..batch_size {
            let in_offset = b * height * width * channels;
            let out_offset = b * channels * height * width;
            for h in 0..height {
                let i_row = in_offset + h * width * channels;
                let o_row = out_offset + h * width;
                for w in 0..width {
                    let i_col = i_row + w * channels;
                    let o_col = o_row + w;
                    for c in 0..channels {
                        let i_idx = i_col + c;
                        let o_idx = o_col + c * width * height;
                        out_data[o_idx] = in_data[i_idx];
                    }
                }
            }
        }
    }

    fn im2col_gemm_conv2d<T: FloatType>(
        &self,
        p: &ConvInput<T, Cpu>,
        in_data: &[T],
        filters: &[T],
        buffer: &mut [T],
        out: &mut [T],
    ) where
        Self: MatMul<T>,
    {
        let out_channels = p.filters.shape[0];
        let ksize = p.filters.shape[2];

        match p.conv_shape {
            super::ConvShape::Nchw => self.im2col_nchw(1, p, in_data, buffer),
            super::ConvShape::Nhwc => self.im2col_nhwc(1, p, in_data, buffer),
        };

        let ic_per_group = p.in_channels / p.groups;
        let oc_per_group = out_channels / p.groups;

        let m = oc_per_group;
        let k = ic_per_group * ksize * ksize;
        let n = p.out_width * p.out_height;

        for g in 0..p.groups {
            Self::gemm_matmul(
                m,
                k,
                n,
                &filters[g * m * k..],
                [k, 1],
                &buffer[g * k * n..],
                [1, k],
                &mut out[g * m * n..],
                [n, 1],
            );
        }
    }
}

impl<T: FloatType> super::ConvKernel<T> for Cpu
where
    Self: MatMul<T>,
{
    fn conv2d_fwd(
        &mut self,
        p: &ConvInput<T, Cpu>,
        mut tensor: Tensor<T, Self>,
    ) -> Tensor<T, Self> {
        assert_eq!(tensor.shape.len(), 4);

        let (out_channel, ksize) = (p.filters.shape[0], p.filters.shape[2]);

        let out_shape = [p.batch_size, out_channel, p.out_height, p.out_width];
        let mut out = Tensor::try_new(&out_shape, self).unwrap();

        let sz = p.in_channels * ksize * ksize * p.out_width * p.out_height;
        let mut buffer = self.try_alloc(sz).unwrap();

        if p.conv_shape == ConvShape::Nhwc {
            tensor = super::permute_nchw_to_nhwc(tensor);
        }

        for b in 0..p.batch_size {
            let t_base = b * tensor.strides[0];
            let o_base = b * out.strides[0];
            let channel_slice = &tensor.data[t_base..t_base + tensor.strides[1]];
            let out_slice = &mut out.data[o_base..o_base + out.strides[1]];
            self.im2col_gemm_conv2d(
                p,
                channel_slice,
                p.filters.data.as_slice(),
                &mut buffer,
                out_slice,
            );
        }

        if p.conv_shape == ConvShape::Nhwc {
            out = super::permute_nhwc_to_nchw(out);
        }
        out
    }

    fn im2col(&mut self, p: &ConvInput<T, Self>, tensor: &Tensor<T, Self>) -> Tensor<T, Self> {
        let (out_channels, ksize) = (p.filters.shape[0], p.filters.shape[2]);
        let shape = [
            p.batch_size,
            out_channels,
            ksize,
            ksize,
            p.out_width,
            p.out_height,
        ];
        let mut out = Tensor::try_new(&shape, self).unwrap();
        match p.conv_shape {
            ConvShape::Nchw => self.im2col_nchw(p.batch_size, p, &tensor.data, &mut out.data),
            ConvShape::Nhwc => self.im2col_nhwc(p.batch_size, p, &tensor.data, &mut out.data),
        };
        out
    }

    fn nchw_to_nhwc(&mut self, tensor: &Tensor<T, Self>) -> Tensor<T, Self> {
        let mut out = Tensor::try_new(&tensor.shape, self).unwrap();
        self.nchw_to_nhwc(&tensor.data, &tensor.shape, &mut out.data);
        out
    }
    fn nhwc_to_nchw(&mut self, tensor: &Tensor<T, Self>) -> Tensor<T, Self> {
        let mut out = Tensor::try_new(&tensor.shape, self).unwrap();
        self.nhwc_to_nchw(&tensor.data, &tensor.shape, &mut out.data);
        out
    }
}
