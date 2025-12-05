use crate::cuda::device::Cuda;
use crate::cuda::utils::*;
use crate::vision_traits::MakeBorderKernel;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rvit_core::pixel_traits::*;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::{compute_strides, tensor_size};
use std::env;

trait KernelOp<T: PixelType, B: BorderMode> {
    const KERNEL_NAME: &'static str;
}

const MAKE_BORDER_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/make_border.ptx"));

impl KernelOp<u8, Constant> for Cuda {
    const KERNEL_NAME: &'static str = "make_constant_border_kernel_u8";
}
impl KernelOp<u16, Constant> for Cuda {
    const KERNEL_NAME: &'static str = "make_constant_border_kernel_u16";
}
impl KernelOp<f32, Constant> for Cuda {
    const KERNEL_NAME: &'static str = "make_constant_border_kernel_f32";
}
impl KernelOp<u8, ClampToEdge> for Cuda {
    const KERNEL_NAME: &'static str = "make_clamp_border_kernel_u8";
}
impl KernelOp<u16, ClampToEdge> for Cuda {
    const KERNEL_NAME: &'static str = "make_clamp_border_kernel_u16";
}
impl KernelOp<f32, ClampToEdge> for Cuda {
    const KERNEL_NAME: &'static str = "make_clamp_border_kernel_f32";
}
impl KernelOp<u8, Mirror> for Cuda {
    const KERNEL_NAME: &'static str = "make_mirror_border_kernel_u8";
}
impl KernelOp<u16, Mirror> for Cuda {
    const KERNEL_NAME: &'static str = "make_mirror_border_kernel_u16";
}
impl KernelOp<f32, Mirror> for Cuda {
    const KERNEL_NAME: &'static str = "make_mirror_border_kernel_f32";
}

impl<T: PixelType, B: BorderMode> MakeBorderKernel<T, B> for Cuda
where
    Self: KernelOp<T, B>,
{
    fn make_border(
        &mut self,
        src: &Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        padding: usize,
    ) -> Self::Vec {
        let (batch_size, channels, width, height) =
            (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
        let k_func = self.register_kernel(MAKE_BORDER_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(width as u32, block_dim.0),
            div_up(height as u32, block_dim.1),
            1,
        );

        let dst_width = width + 2 * padding;
        let dst_height = height + 2 * padding;
        let dst_shape = [batch_size, channels, dst_width, dst_height];
        let dst_sz = tensor_size(&dst_shape);
        let dst_strides = compute_strides(&dst_shape);
        let mb_img = self.try_alloc(dst_sz);

        for b in 0..batch_size {
            let slice_base = b * src_strides[0];
            let dst_slice_base = b * dst_strides[0];
            for c in 0..channels {
                let src_slice_start = slice_base + c * src_strides[1];
                let dst_slice_start = dst_slice_base + c * dst_strides[1];
                let src_slice_end = src_slice_start + src_strides[1];
                let dst_slice_end = dst_slice_start + dst_strides[1];
                let src_view = src.slice(src_slice_start..src_slice_end);
                let dst_view = mb_img.slice(dst_slice_start..dst_slice_end);

                let mut builder = self.stream0.launch_builder(&k_func);
                builder
                    .arg(&src_view)
                    .arg(&width)
                    .arg(&height)
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
