use std::env;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::device::cu_utils::*;
use crate::device::cuda::Cuda;
use crate::image::PixelType;
use super::*;

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
    fn resize(&mut self, src: &mut Image<T, Self>, dst_width: usize, dst_height: usize) -> Image<T, Self>
    {
        let k_func = self.register_kernel(RESIZE_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(dst_width as u32, block_dim.0),
            div_up(dst_height as u32, block_dim.1),
            1,
        );

        let rz_img = Image::try_new(src.batch_size, dst_width, dst_height, src.channels, self).unwrap();
        let scale_x = src.width as f32 / dst_width as f32;
        let scale_y = src.height as f32 / dst_height as f32 ;

        for b in 0..src.batch_size {
            let slice_base = b * src.strides[0];
            let dst_slice_base = b * rz_img.strides[0];
            for c in 0..src.channels {
                let slice_start = slice_base + c * src.strides[1];
                let dst_slice_start = dst_slice_base + c * rz_img.strides[1];
                let slice_end = slice_start + src.strides[1];
                let dst_slice_end = dst_slice_start + rz_img.strides[0];
                let s = src.data.slice(slice_start..slice_end);
                let cs = rz_img.data.slice(dst_slice_start..dst_slice_end);

                let mut builder = self.stream0.launch_builder(&k_func);
                builder
                    .arg(&s)
                    .arg(&src.width)
                    .arg(&src.height)
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