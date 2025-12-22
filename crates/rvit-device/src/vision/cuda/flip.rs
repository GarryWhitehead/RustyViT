use crate::cuda::device::Cuda;
use crate::cuda::utils::*;
use crate::op_traits::HorizontalFlipKernel;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rand::Rng;
use rvit_core::pixel_traits::PixelType;
use std::env;

trait KernelOp<TYPE> {
    const KERNEL_NAME: &'static str;
}

const FLIP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/flip_image.ptx"));

impl KernelOp<u8> for Cuda {
    const KERNEL_NAME: &'static str = "flip_horiz_u8_kernel";
}
impl KernelOp<u16> for Cuda {
    const KERNEL_NAME: &'static str = "flip_horiz_u16_kernel";
}
impl KernelOp<f32> for Cuda {
    const KERNEL_NAME: &'static str = "flip_horiz_f32_kernel";
}

impl<T: PixelType> HorizontalFlipKernel<T> for Cuda
where
    Self: KernelOp<T>,
{
    fn flip_horizontal(
        &mut self,
        src: &mut Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        prob: f32,
    ) {
        let (batch_size, channels, width, height) =
            (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
        let k_func = self.register_kernel(FLIP_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(width as u32, block_dim.0),
            div_up(height as u32, block_dim.1),
            1,
        );

        let mut rng = rand::rng();
        for b in 0..batch_size {
            if rng.random_range(0.0..1.0) < prob {
                let slice_base = b * src_strides[0];
                for c in 0..channels {
                    let slice_start = slice_base + c * src_strides[1];
                    let slice_end = slice_start + src_strides[1];
                    let s = src.slice(slice_start..slice_end);
                    let mut builder = self.stream0.launch_builder(&k_func);
                    builder.arg(&s).arg(&width).arg(&height);
                    let cfg = LaunchConfig {
                        block_dim,
                        grid_dim,
                        shared_mem_bytes: 0,
                    };
                    unsafe { builder.launch(cfg) }.unwrap();
                }
            }
        }
    }
}
