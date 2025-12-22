use crate::cuda::device::Cuda;
use crate::cuda::utils::*;
use crate::op_traits::ResizeKernel;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rvit_core::pixel_traits::*;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::{compute_strides, tensor_size};
use std::env;

const RESIZE_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/resize.ptx"));

trait KernelOp<T: PixelType, I: InterpMode> {
    const KERNEL_NAME: &'static str;
}
impl KernelOp<u8, Bilinear> for Cuda {
    const KERNEL_NAME: &'static str = "bilinear_resize_kernel_u8";
}
impl KernelOp<u16, Bilinear> for Cuda {
    const KERNEL_NAME: &'static str = "bilinear_resize_kernel_u16";
}
impl KernelOp<f32, Bilinear> for Cuda {
    const KERNEL_NAME: &'static str = "bilinear_resize_kernel_f32";
}

impl<T: PixelType, I: InterpMode> ResizeKernel<T, I> for Cuda
where
    Self: KernelOp<T, I>,
{
    fn resize(
        &mut self,
        src: &Self::Vec,
        in_shape: &[usize],
        in_strides: &[usize],
        dst_width: usize,
        dst_height: usize,
    ) -> Self::Vec {
        let (batch_size, channels, width, height) =
            (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let k_func = self.register_kernel(RESIZE_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(dst_width as u32, block_dim.0),
            div_up(dst_height as u32, block_dim.1),
            1,
        );

        let rz_shape = [batch_size, channels, dst_width, dst_height];
        let rz_sz = tensor_size(&rz_shape);
        let rz_strides = compute_strides(&rz_shape);
        let rz_img = self.try_alloc(rz_sz).unwrap();

        let scale_x = width as f32 / dst_width as f32;
        let scale_y = height as f32 / dst_height as f32;

        for b in 0..batch_size {
            let slice_base = b * in_strides[0];
            let dst_slice_base = b * rz_strides[0];
            for c in 0..channels {
                let slice_start = slice_base + c * in_strides[1];
                let dst_slice_start = dst_slice_base + c * rz_strides[1];
                let slice_end = slice_start + in_strides[1];
                let dst_slice_end = dst_slice_start + rz_strides[0];
                let s = src.slice(slice_start..slice_end);
                let cs = rz_img.slice(dst_slice_start..dst_slice_end);

                let mut builder = self.stream0.launch_builder(&k_func);
                builder
                    .arg(&s)
                    .arg(&width)
                    .arg(&height)
                    .arg(&dst_width)
                    .arg(&dst_height)
                    .arg(&scale_x)
                    .arg(&scale_y)
                    .arg(&cs);
                let cfg = LaunchConfig {
                    block_dim,
                    grid_dim,
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg) }.unwrap();
            }
        }
        rz_img
    }
}
