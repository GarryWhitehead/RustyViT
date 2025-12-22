use crate::cuda::device::Cuda;
use crate::cuda::tensor::matmul::BatchedMatMul;
use crate::cuda::utils::*;
use crate::op_traits::{Conv2dKernel, ConvConvertKernel};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};
use rvit_core::element_traits::DataElem;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::tensor_size;

const IM2COL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/im2col.ptx"));
const CONVERT_SHAPE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/shape_convert.ptx"));

pub trait KernelOp<T: DataElem> {
    const IM2COL_KERNEL_NAME: &'static str;
    const TO_NHWC_KERNEL_NAME: &'static str;
    const TO_NCHW_KERNEL_NAME: &'static str;
}
impl KernelOp<f32> for Cuda {
    const IM2COL_KERNEL_NAME: &'static str = "im2col_f32_kernel";
    const TO_NHWC_KERNEL_NAME: &'static str = "nchw_to_nhwc_f32_kernel";
    const TO_NCHW_KERNEL_NAME: &'static str = "nhwc_to_nchw_f32_kernel";
}
impl KernelOp<half::f16> for Cuda {
    const IM2COL_KERNEL_NAME: &'static str = "im2col_f16_kernel";
    const TO_NHWC_KERNEL_NAME: &'static str = "nchw_to_nhwc_f16_kernel";
    const TO_NCHW_KERNEL_NAME: &'static str = "nhwc_to_nchw_f16_kernel";
}

impl Cuda {
    fn im2col_nchw<T: DataElem>(
        &mut self,
        batch_size: i32,
        ksize: i32,
        stride: usize,
        padding: usize,
        in_data: &CudaSlice<T>,
        in_shape: &[usize],
        buffer: &CudaSlice<T>,
        out_height: usize,
        out_width: usize,
        is_nhwc: bool,
    ) where
        Self: KernelOp<T>,
    {
        let k_func = self.register_kernel(IM2COL_PTX, Self::IM2COL_KERNEL_NAME);

        let (in_channels, in_height, in_width) = (in_shape[1], in_shape[0], in_shape[1]);
        let thread_size = ksize * ksize * out_width as i32;
        let block_dim = (256, 1, 1);
        let grid_dim = (
            div_up(thread_size as u32, block_dim.0),
            out_height as u32,
            (in_channels * batch_size as usize) as u32,
        );

        let ic = in_channels as i32;
        let oh = out_height as i32;
        let ow = out_width as i32;
        let ih = in_height as i32;
        let iw = in_width as i32;
        let stride = stride as i32;
        let padding = padding as i32;

        let mut builder = self.stream0.launch_builder(&k_func);
        builder
            .arg(in_data)
            .arg(&thread_size)
            .arg(&batch_size)
            .arg(&ic)
            .arg(&ksize)
            .arg(&oh)
            .arg(&ow)
            .arg(&ih)
            .arg(&iw)
            .arg(&stride)
            .arg(&padding)
            .arg(buffer);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
    }

    fn convert_shape<T: DataElem>(
        &mut self,
        kname: &str,
        in_data: &CudaSlice<T>,
        batch_size: i32,
        channels: i32,
        height: i32,
        width: i32,
        out_data: &CudaSlice<T>,
    ) {
        let k_func = self.register_kernel(CONVERT_SHAPE_PTX, kname);

        let thread_size = height * width;
        let block_dim = (256, 1, 1);
        let grid_dim = (
            div_up(thread_size as u32, block_dim.0),
            channels as u32,
            batch_size as u32,
        );

        let mut builder = self.stream0.launch_builder(&k_func);
        builder
            .arg(in_data)
            .arg(&thread_size)
            .arg(&batch_size)
            .arg(&channels)
            .arg(&height)
            .arg(&width)
            .arg(out_data);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
    }

    fn im2col_gemm_conv2d<T: DataElem>(
        &mut self,
        batch_size: usize,
        in_data: &CudaSlice<T>,
        in_shape: &[usize],
        filters: &CudaSlice<T>,
        f_shape: &[usize],
        buffer: &CudaSlice<T>,
        out: &mut CudaSlice<T>,
        out_height: usize,
        out_width: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        is_nhwc: bool,
    ) where
        Self: BatchedMatMul<T>,
        Self: KernelOp<T>,
    {
        let (out_channels, ksize) = (f_shape[0], f_shape[2]);
        let (in_channels, in_height, in_width) = (in_shape[1], in_shape[2], in_shape[3]);

        if is_nhwc {
            todo!()
        } else {
            self.im2col_nchw(
                batch_size as i32,
                ksize as i32,
                stride,
                padding,
                &in_data,
                in_shape,
                &buffer,
                out_height,
                out_width,
                is_nhwc,
            );
        }

        let ic_per_group = in_channels / groups;
        let oc_per_group = out_channels / groups;

        let m = oc_per_group;
        let k = ic_per_group * ksize * ksize;
        let n = out_width * out_height;

        // Note - the input matrix is transposed here, thus the a and b inputs are switched
        // along with the m and n dimensions.
        if groups == 1 {
            self.batched_gemm(
                n,
                m,
                k,
                batch_size,
                k * n,
                0,
                m * n,
                &buffer.as_view(),
                &filters.as_view(),
                &mut out.as_view_mut(),
            );
        } else {
            // Less efficient grouped version.
            for b in 0..batch_size {
                let buffer_offset = b * groups * k * n;
                let out_offset = b * groups * m * n;
                self.batched_gemm(
                    n,
                    m,
                    k,
                    groups,
                    k * n,
                    m * k,
                    m * n,
                    &buffer.slice(buffer_offset..),
                    &filters.as_view(),
                    &mut out.slice_mut(out_offset..),
                );
            }
        }
    }
}

impl<T: DataElem> Conv2dKernel<T> for Cuda
where
    Self: BatchedMatMul<T>,
    Self: KernelOp<T>,
{
    fn forward(
        &mut self,
        x: &Self::Vec,
        shape: &[usize],
        _strides: &[usize],
        filters: &Self::Vec,
        f_shape: &[usize],
        out_shape: &[usize],
        stride: usize,
        padding: usize,
        groups: usize,
        to_nhwc: bool,
    ) -> Self::Vec {
        let (out_channels, ksize) = (f_shape[0], f_shape[2]);
        let (batch_size, in_channels, in_height, in_width) =
            (shape[0], shape[2], shape[3], shape[0]);
        let (out_height, out_width) = (out_shape[2], out_shape[3]);

        let out_sz = tensor_size(&out_shape);
        let mut out = self.try_alloc(out_sz).unwrap();

        let sz = batch_size * in_channels * ksize * ksize * out_width * out_height;
        let buffer = self.try_alloc(sz).unwrap();

        self.im2col_gemm_conv2d(
            batch_size, &x, shape, filters, f_shape, &buffer, &mut out, out_height, out_width,
            stride, padding, groups, to_nhwc,
        );

        out
    }
}

impl<T: DataElem> ConvConvertKernel<T> for Cuda {
    fn im2col(
        &mut self,
        x: &Self::Vec,
        x_shape: &[usize],
        f_shape: &[usize],
        batch_size: usize,
        out_width: usize,
        out_height: usize,
        stride: usize,
        padding: usize,
        is_nhwc: bool,
    ) -> Self::Vec {
        let (out_channels, ksize) = (f_shape[0], f_shape[2]);
        let out_shape = [
            batch_size,
            out_channels,
            out_width,
            out_height,
            ksize,
            ksize,
        ];

        let out_sz = tensor_size(&out_shape);
        let out = self.try_alloc(out_sz).unwrap();
        if is_nhwc {
            todo!()
        } else {
            self.im2col_nchw(
                batch_size as i32,
                ksize as i32,
                stride,
                padding,
                &x,
                x_shape,
                &out,
                out_height,
                out_width,
                is_nhwc,
            );
        }
        out
    }

    fn nchw_to_nhwc(&mut self, x: &Self::Vec, shape: &[usize]) -> Self::Vec
    where
        Self: KernelOp<T>,
    {
        let (batch_size, in_channels, in_height, in_width) =
            (shape[0], shape[1], shape[2], shape[3]);
        let out_sz = tensor_size(&shape);
        let mut out = self.try_alloc(out_sz).unwrap();

        self.convert_shape(
            Self::TO_NHWC_KERNEL_NAME,
            &x,
            batch_size as i32,
            in_channels as i32,
            in_height as i32,
            in_width as i32,
            &mut out,
        );
        out
    }

    fn nhwc_to_nchw(&mut self, x: &Self::Vec, shape: &[usize]) -> Self::Vec
    where
        Self: KernelOp<T>,
    {
        let (batch_size, in_channels, in_height, in_width) =
            (shape[0], shape[1], shape[2], shape[3]);
        let out_sz = tensor_size(&shape);
        let mut out = self.try_alloc(out_sz).unwrap();

        self.convert_shape(
            Self::TO_NCHW_KERNEL_NAME,
            &x,
            batch_size as i32,
            in_channels as i32,
            in_height as i32,
            in_width as i32,
            &mut out,
        );
        out
    }
}
