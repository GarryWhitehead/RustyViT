use crate::device::cu_utils::*;
use crate::device::cuda::Cuda;
use crate::image::{Image, PixelType};
use crate::vision::make_border::BorderMode;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::env;

trait KernelOp<T: PixelType, B: BorderMode> {
    const KERNEL_NAME: &'static str;
}

const MAKE_BORDER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/make_border.ptx"));

impl KernelOp<u8, super::Constant> for Cuda {
    const KERNEL_NAME: &'static str = "make_constant_border_kernel_u8";
}
impl KernelOp<u16, super::Constant> for Cuda {
    const KERNEL_NAME: &'static str = "make_constant_border_kernel_u16";
}
impl KernelOp<f32, super::Constant> for Cuda {
    const KERNEL_NAME: &'static str = "make_constant_border_kernel_f32";
}
impl KernelOp<u8, super::ClampToEdge> for Cuda {
    const KERNEL_NAME: &'static str = "make_clamp_border_kernel_u8";
}
impl KernelOp<u16, super::ClampToEdge> for Cuda {
    const KERNEL_NAME: &'static str = "make_clamp_border_kernel_u16";
}
impl KernelOp<f32, super::ClampToEdge> for Cuda {
    const KERNEL_NAME: &'static str = "make_clamp_border_kernel_f32";
}
impl KernelOp<u8, super::Mirror> for Cuda {
    const KERNEL_NAME: &'static str = "make_mirror_border_kernel_u8";
}
impl KernelOp<u16, super::Mirror> for Cuda {
    const KERNEL_NAME: &'static str = "make_mirror_border_kernel_u16";
}
impl KernelOp<f32, super::Mirror> for Cuda {
    const KERNEL_NAME: &'static str = "make_mirror_border_kernel_f32";
}

impl<T: PixelType, B: BorderMode> super::MakeBorderKernel<T, B> for Cuda
where
    Self: KernelOp<T, B>,
{
    fn make_border(&mut self, src: &Image<T, Self>, padding: usize) -> Image<T, Self> {
        let k_func = self.register_kernel(MAKE_BORDER_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(src.width as u32, block_dim.0),
            div_up(src.height as u32, block_dim.1),
            1,
        );

        let dst_width = src.width + 2 * padding;
        let dst_height = src.height + 2 * padding;
        let mb_img =
            Image::try_new(src.batch_size, dst_width, dst_height, src.channels, self).unwrap();

        for b in 0..src.batch_size {
            let slice_base = b * src.strides[0];
            let dst_slice_base = b * mb_img.strides[0];
            for c in 0..src.channels {
                let src_slice_start = slice_base + c * src.strides[1];
                let dst_slice_start = dst_slice_base + c * mb_img.strides[1];
                let src_slice_end = src_slice_start + src.strides[1];
                let dst_slice_end = dst_slice_start + mb_img.strides[1];
                let src_view = src.data.slice(src_slice_start..src_slice_end);
                let dst_view = mb_img.data.slice(dst_slice_start..dst_slice_end);

                let mut builder = self.stream0.launch_builder(&k_func);
                builder
                    .arg(&src_view)
                    .arg(&src.width)
                    .arg(&src.height)
                    .arg(&padding)
                    .arg(&dst_view);
                let cfg = LaunchConfig {
                    block_dim,
                    grid_dim,
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg) }.unwrap();
            }
        }
        mb_img
    }
}
