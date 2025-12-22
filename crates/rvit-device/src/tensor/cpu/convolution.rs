use crate::cpu::device::Cpu;
use crate::tensor::cpu::matmul::MatMul;
use crate::tensor::op_traits::{Conv2dKernel, ConvConvertKernel};
use crate::{DAlloc, Device, Runtime};
use rvit_core::element_traits::{DataElem, Elem, FloatElem, IntElem};
use rvit_core::memory::storage::DeviceStorage;
use rvit_core::tensor::*;

impl Runtime {
    fn im2col_nchw<T: Copy>(
        &self,
        batch_size: usize,
        in_channels: usize,
        in_height: usize,
        in_width: usize,
        out_height: usize,
        out_width: usize,
        ksize: usize,
        stride: usize,
        padding: usize,
        in_data: &[T],
        buffer: &mut [T],
    ) {
        let mut out_idx = 0;
        for b in 0..batch_size {
            let in_offset = b * in_channels * in_width * in_height;
            for c in 0..in_channels {
                for oh in 0..out_height {
                    for ow in 0..out_width {
                        for kh in 0..ksize {
                            for kw in 0..ksize {
                                let row = (oh * stride + kh).wrapping_sub(padding);
                                let col = (ow * stride + kw).wrapping_sub(padding);
                                if row < in_height && col < in_width {
                                    let in_idx = c * (in_width * in_height) + row * in_width + col;
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

    fn im2col_nhwc<T: Copy>(
        &self,
        batch_size: usize,
        out_channels: usize,
        in_channels: usize,
        in_height: usize,
        in_width: usize,
        out_height: usize,
        out_width: usize,
        ksize: usize,
        stride: usize,
        padding: usize,
        in_data: &[T],
        buffer: &mut [T],
    ) {
        for b in 0..batch_size {
            let in_offset = b * in_channels * in_width * in_height;
            let out_offset = b * out_channels * out_width * out_height;
            for oh in 0..out_height {
                for ow in 0..out_width {
                    let row = oh * out_width + ow;
                    for ky in 0..ksize {
                        let iy = (stride * oh + ky).wrapping_sub(padding);
                        if iy >= in_height {
                            continue;
                        }
                        for kx in 0..ksize {
                            let ix = (stride * ow + kx).wrapping_sub(padding);
                            if ix >= in_width {
                                continue;
                            }
                            for c in 0..in_channels {
                                let col = c * ksize * ksize + ky * ksize + kx;
                                buffer[out_offset + row * in_channels * ksize * ksize + col] =
                                    in_data[in_offset + (iy * in_width + ix) * in_channels + c];
                            }
                        }
                    }
                }
            }
        }
    }

    fn nchw_to_nhwc<T: Copy>(&mut self, in_data: &[T], shape: &[usize], out_data: &mut [T]) {
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

    fn nhwc_to_nchw<T: Copy>(&mut self, in_data: &[T], shape: &[usize], out_data: &mut [T]) {
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

    fn im2col_gemm_conv2d<T: Copy>(
        &self,
        in_data: &[T],
        in_shape: &[usize],
        filters: &[T],
        f_shape: &[usize],
        buffer: &mut [T],
        out: &mut [T],
        out_height: usize,
        out_width: usize,
        groups: usize,
        stride: usize,
        padding: usize,
        is_nhwc: bool,
    ) where
        Self: MatMul<T>,
    {
        let (out_channels, ksize) = (f_shape[0], f_shape[2]);
        let (in_channels, in_height, in_width) = (in_shape[1], in_shape[2], in_shape[3]);

        if is_nhwc {
            self.im2col_nhwc(
                1,
                out_channels,
                in_channels,
                in_height,
                in_width,
                out_height,
                out_width,
                ksize,
                stride,
                padding,
                in_data,
                buffer,
            );
        } else {
            self.im2col_nchw(
                1,
                out_channels,
                in_channels,
                in_height,
                in_width,
                out_height,
                out_width,
                ksize,
                stride,
                in_data,
                buffer,
            );
        }

        let ic_per_group = in_channels / groups;
        let oc_per_group = out_channels / groups;

        let m = oc_per_group;
        let k = ic_per_group * ksize * ksize;
        let n = out_width * out_height;

        for g in 0..groups {
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

impl<E: DataElem> Conv2dKernel<E> for Runtime
where
    Self: MatMul<E>,
{
    fn conv2d_fwd(
        &mut self,
        x: &DAlloc<Self>,
        shape: &[usize],
        strides: &[usize],
        filters: &DAlloc<Self>,
        f_shape: &[usize],
        out_shape: &[usize],
        stride: usize,
        padding: usize,
        groups: usize,
        to_nhwc: bool,
    ) -> DAlloc<Self> {
        assert_eq!(shape.len(), 4);
        assert_eq!(out_shape.len(), 4);

        let ksize = f_shape[2];

        let out_sz = tensor_size(out_shape);
        let mut out = self.storage.try_alloc(out_sz, x.dtype()).unwrap();
        let out_strides = compute_strides(out_shape);

        let (batch_size, in_channels) = (shape[0], shape[1]);
        let (out_height, out_width) = (out_shape[2], out_shape[3]);
        let sz = in_channels * ksize * ksize * out_width * out_height;
        let mut buffer = self.storage.try_alloc(sz, x.dtype()).unwrap();

        let x_slice = x.as_slice().unwrap();
        let mut out_slice = out.as_mut_slice().unwrap();

        for b in 0..batch_size {
            let t_base = b * strides[0];
            let o_base = b * out_strides[0];
            let channel_slice = &x_slice[t_base..t_base + strides[1]];
            let out_slice = &mut out_slice[o_base..o_base + out_strides[1]];
            self.im2col_gemm_conv2d(
                channel_slice,
                shape,
                filters.as_slice().unwrap(),
                f_shape,
                &mut buffer.as_mut_slice().unwrap(),
                out_slice,
                out_height,
                out_width,
                groups,
                stride,
                padding,
                to_nhwc,
            );
        }

        out
    }
}

impl<E: DataElem> ConvConvertKernel<E> for Runtime {
    fn im2col(
        &mut self,
        x: &DAlloc<Self>,
        x_shape: &[usize],
        f_shape: &[usize],
        batch_size: usize,
        out_width: usize,
        out_height: usize,
        stride: usize,
        padding: usize,
        is_nhwc: bool,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>) {
        let (out_channels, ksize) = (f_shape[0], f_shape[2]);
        let (in_channels, in_height, in_width) = (x_shape[1], x_shape[2], x_shape[3]);
        let out_shape = [
            batch_size,
            out_channels,
            ksize,
            ksize,
            out_width,
            out_height,
        ];
        let out_strides = compute_strides(&out_shape);

        let out_sz = tensor_size(&out_shape);
        let mut out = self.storage.try_alloc(out_sz, x.dtype()).unwrap();
        if is_nhwc {
            self.im2col_nhwc(
                1,
                out_channels,
                in_channels,
                in_height,
                in_width,
                out_height,
                out_width,
                ksize,
                stride,
                padding,
                x.as_slice::<E>().unwrap(),
                &mut out.as_mut_slice().unwrap(),
            );
        } else {
            self.im2col_nchw(
                1,
                out_channels,
                in_channels,
                in_height,
                in_width,
                out_height,
                out_width,
                ksize,
                stride,
                x.as_slice::<E>().unwrap(),
                &mut out.as_mut_slice().unwrap(),
            );
        }
        (out, out_shape.to_vec(), out_strides)
    }

    fn nhwc_to_nchw(&mut self, x: &DAlloc<Self>, shape: &[usize]) -> DAlloc<Self> {
        let sz = tensor_size(shape);
        let mut out = self.storage.try_alloc(sz, x.dtype()).unwrap();
        self.nhwc_to_nchw(
            &x.as_slice::<E>().unwrap(),
            &shape,
            &mut out.as_mut_slice().unwrap(),
        );
        out
    }

    fn nchw_to_nhwc(&mut self, x: &DAlloc<Self>, shape: &[usize]) -> DAlloc<Self> {
        let sz = tensor_size(shape);
        let mut out = self.storage.try_alloc(sz, x.dtype()).unwrap();
        self.nchw_to_nhwc(
            &x.as_slice::<E>().unwrap(),
            &shape,
            &mut out.as_mut_slice().unwrap(),
        );
        out
    }
}
