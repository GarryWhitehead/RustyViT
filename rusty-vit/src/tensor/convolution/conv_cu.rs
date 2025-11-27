use crate::device::DeviceStorage;
use crate::device::cu_utils::div_up;
use crate::device::cuda::Cuda;
use crate::tensor::Tensor;
use crate::tensor::convolution::{ConvInput, ConvShape};
use crate::tensor::matmul::matmul_cu::BatchedMatMul;
use crate::type_traits::FloatType;
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

const IM2COL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/im2col.ptx"));
const CONVERT_SHAPE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/shape_convert.ptx"));

pub trait KernelOp<T: FloatType> {
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
    fn im2col_nchw<T: FloatType>(
        &mut self,
        batch_size: i32,
        ksize: i32,
        p: &ConvInput<T, Self>,
        in_data: &CudaSlice<T>,
        buffer: &CudaSlice<T>,
    ) where
        Self: KernelOp<T>,
    {
        let k_func = self.register_kernel(IM2COL_PTX, Self::IM2COL_KERNEL_NAME);

        let thread_size = ksize * ksize * p.out_width as i32;
        let block_dim = (256, 1, 1);
        let grid_dim = (
            div_up(thread_size as u32, block_dim.0),
            p.out_height as u32,
            (p.in_channels * batch_size as usize) as u32,
        );

        let ic = p.in_channels as i32;
        let oh = p.out_height as i32;
        let ow = p.out_width as i32;
        let ih = p.in_height as i32;
        let iw = p.in_width as i32;
        let stride = p.stride as i32;
        let padding = p.padding as i32;

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

    fn convert_shape<T: FloatType>(
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

    fn im2col_gemm_conv2d<T: FloatType>(
        &mut self,
        p: &ConvInput<T, Self>,
        in_data: &CudaSlice<T>,
        filters: &CudaSlice<T>,
        buffer: &CudaSlice<T>,
        out: &mut CudaSlice<T>,
    ) where
        Self: BatchedMatMul<T>,
        Self: KernelOp<T>,
    {
        let (out_channels, ksize) = (p.filters.shape[0], p.filters.shape[2]);

        match p.conv_shape {
            ConvShape::Nchw => {
                self.im2col_nchw(p.batch_size as i32, ksize as i32, &p, &in_data, &buffer)
            }
            ConvShape::Nhwc => todo!(),
        }

        let ic_per_group = p.in_channels / p.groups;
        let oc_per_group = out_channels / p.groups;

        let m = oc_per_group;
        let k = ic_per_group * ksize * ksize;
        let n = p.out_width * p.out_height;

        // Note - the input matrix is transposed here, thus the a and b inputs are switched
        // along with the m and n dimensions.
        if p.groups == 1 {
            self.batched_gemm(
                n,
                m,
                k,
                p.batch_size,
                k * n,
                0,
                m * n,
                &buffer.as_view(),
                &filters.as_view(),
                &mut out.as_view_mut(),
            );
        } else {
            // Less efficient grouped version.
            for b in 0..p.batch_size {
                let buffer_offset = b * p.groups * k * n;
                let out_offset = b * p.groups * m * n;
                self.batched_gemm(
                    n,
                    m,
                    k,
                    p.groups,
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

impl<T: FloatType> super::ConvKernel<T> for Cuda
where
    Self: BatchedMatMul<T>,
    Self: KernelOp<T>,
{
    fn conv2d_fwd(&mut self, p: &ConvInput<T, Self>, tensor: Tensor<T, Self>) -> Tensor<T, Self> {
        let (out_channels, ksize) = (p.filters.shape[0], p.filters.shape[2]);

        let out_shape = [p.batch_size, out_channels, p.out_height, p.out_width];
        let mut out = Tensor::try_new(&out_shape, self).unwrap();

        let sz = p.batch_size * p.in_channels * ksize * ksize * p.out_width * p.out_height;
        let buffer = self.try_alloc(sz).unwrap();

        self.im2col_gemm_conv2d(p, &tensor.data, &p.filters.data, &buffer, &mut out.data);

        out
    }

    fn im2col(&mut self, p: &ConvInput<T, Self>, tensor: &Tensor<T, Self>) -> Tensor<T, Self> {
        let (out_channels, ksize) = (p.filters.shape[0], p.filters.shape[2]);
        let out_shape = [
            p.batch_size,
            out_channels,
            p.out_width,
            p.out_height,
            ksize,
            ksize,
        ];
        let out = Tensor::try_new(&out_shape, self).unwrap();
        match p.conv_shape {
            ConvShape::Nchw => self.im2col_nchw(
                p.batch_size as i32,
                ksize as i32,
                p,
                &tensor.data,
                &out.data,
            ),
            ConvShape::Nhwc => todo!(), //self.im2col_nhwc(p.batch_size, ksize, p, &tensor.data, &out),
        };
        out
    }

    fn nchw_to_nhwc(&mut self, tensor: &Tensor<T, Self>) -> Tensor<T, Self>
    where
        Self: KernelOp<T>,
    {
        let mut out = Tensor::try_new(&tensor.shape, self).unwrap();
        self.convert_shape(
            Self::TO_NHWC_KERNEL_NAME,
            &tensor.data,
            tensor.shape[0] as i32,
            tensor.shape[1] as i32,
            tensor.shape[2] as i32,
            tensor.shape[3] as i32,
            &mut out.data,
        );
        out
    }

    fn nhwc_to_nchw(&mut self, tensor: &Tensor<T, Self>) -> Tensor<T, Self>
    where
        Self: KernelOp<T>,
    {
        let mut out = Tensor::try_new(&tensor.shape, self).unwrap();
        self.convert_shape(
            Self::TO_NCHW_KERNEL_NAME,
            &tensor.data,
            tensor.shape[0] as i32,
            tensor.shape[3] as i32,
            tensor.shape[1] as i32,
            tensor.shape[2] as i32,
            &mut out.data,
        );
        out
    }
}
