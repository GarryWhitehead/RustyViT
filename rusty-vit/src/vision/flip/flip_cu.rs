use crate::device::cu_utils::*;
use crate::device::cuda::Cuda;
use crate::image::{Image, PixelType};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rand::Rng;
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

impl<T: PixelType> super::HorizFlipKernel<T> for Cuda
where
    Self: KernelOp<T>,
{
    fn flip_horizontal(&mut self, src: &mut Image<T, Self>, prob: f32) {
        let k_func = self.register_kernel(FLIP_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(src.width as u32, block_dim.0),
            div_up(src.height as u32, block_dim.1),
            1,
        );

        let mut rng = rand::rng();
        for b in 0..src.batch_size {
            if rng.random_range(0.0..1.0) < prob {
                let slice_base = b * src.strides[0];
                for c in 0..src.channels {
                    let slice_start = slice_base + c * src.strides[1];
                    let slice_end = slice_start + src.strides[1];
                    let s = src.data.slice(slice_start..slice_end);
                    let mut builder = self.stream0.launch_builder(&k_func);
                    builder.arg(&s).arg(&src.width).arg(&src.height);
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
