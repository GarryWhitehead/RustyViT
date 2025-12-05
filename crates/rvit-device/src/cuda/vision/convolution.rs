use super::*;
use crate::cuda::device::Cuda;
use crate::vision_traits::ConvKernel;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::tensor_size;

// TODO: These are duplicated also in the CU file which isn't brilliant.
// Maybe there is a way of passing this to the CU side - using the constant buffer maybe?
// The tile width for the horizontal pass.
const HORIZ_TILE_WIDTH: u32 = 16;
// The tile height for the horizontal pass.
const HORIZ_TILE_HEIGHT: u32 = 4;
const HORIZ_RESULT_STEPS: u32 = 8;
// The tile width for the vertical pass.
const VERT_TILE_WIDTH: u32 = 16;
// The tile height for the vertical pass.
const VERT_TILE_HEIGHT: u32 = 8;
const VERT_RESULT_STEPS: u32 = 8;

const CONV_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/conv.ptx"));

pub(crate) trait KernelOp<T: PixelType> {
    const HORIZ_KERNEL_NAME: &'static str;
    const VERT_KERNEL_NAME: &'static str;
}
impl KernelOp<u8> for Cuda {
    const HORIZ_KERNEL_NAME: &'static str = "conv_horiz_kernel_u8";
    const VERT_KERNEL_NAME: &'static str = "conv_vert_kernel_u8";
}
impl KernelOp<u16> for Cuda {
    const HORIZ_KERNEL_NAME: &'static str = "conv_horiz_kernel_u16";
    const VERT_KERNEL_NAME: &'static str = "conv_vert_kernel_u16";
}
impl KernelOp<f32> for Cuda {
    const HORIZ_KERNEL_NAME: &'static str = "conv_horiz_kernel_f32";
    const VERT_KERNEL_NAME: &'static str = "conv_vert_kernel_f32";
}

impl<T: PixelType> ConvKernel<T> for Cuda
where
    Self: KernelOp<T>,
{
    fn convolution(
        &mut self,
        src: &mut &Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        x_kernel: &Self::Vec,
        y_kernel: &Self::Vec,
    ) {
        let (batch_size, channels, width, height) =
            (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
        if width as u32 % HORIZ_TILE_WIDTH != 0 {
            panic!(
                "kernel width is not a multiple of the tile width {}",
                HORIZ_TILE_WIDTH
            );
        }
        if height as u32 % VERT_TILE_HEIGHT != 0 {
            panic!(
                "kernel height is not a multiple of the tile height {}",
                VERT_TILE_HEIGHT
            );
        }

        let k_horiz_func = self.register_kernel(CONV_PTX, Self::HORIZ_KERNEL_NAME);
        let k_vert_func = self.register_kernel(CONV_PTX, Self::VERT_KERNEL_NAME);

        let horiz_cfg = LaunchConfig {
            block_dim: (HORIZ_TILE_WIDTH, HORIZ_TILE_HEIGHT, 1),
            grid_dim: (
                width as u32 / (HORIZ_RESULT_STEPS * HORIZ_TILE_WIDTH),
                height as u32 / HORIZ_TILE_HEIGHT,
                1,
            ),
            shared_mem_bytes: 0,
        };
        let vert_cfg = LaunchConfig {
            block_dim: (VERT_TILE_WIDTH, VERT_TILE_HEIGHT, 1),
            grid_dim: (
                width as u32 / VERT_TILE_WIDTH,
                height as u32 / (VERT_RESULT_STEPS * VERT_TILE_HEIGHT),
                1,
            ),
            shared_mem_bytes: 0,
        };

        let tmp_shape = [1, 1, width, height];
        let tmp_sz = tensor_size(&tmp_shape);
        let tmp_img = self.try_alloc(tmp_sz).unwrap();

        for b in 0..batch_size {
            let batch_offset = b * src_strides[0];
            for c in 0..channels {
                let start_idx = batch_offset + c * src_strides[1];
                let end_idx = start_idx + src_strides[1];
                let src_slice = src.slice(start_idx..end_idx);

                // Horizontal pass. src -> tmp image
                let mut builder = self.stream0.launch_builder(&k_horiz_func);
                let x_len = x_kernel.len();
                builder
                    .arg(&src_slice)
                    .arg(&tmp_img)
                    .arg(&width)
                    .arg(&height)
                    .arg(&x_len)
                    .arg(&x_kernel);
                unsafe { builder.launch(horiz_cfg) }.unwrap();

                // Vertical pass. tmp -> src image
                let mut builder = self.stream0.launch_builder(&k_vert_func);
                let y_len = y_kernel.len();
                builder
                    .arg(&tmp_img)
                    .arg(&src_slice)
                    .arg(&width)
                    .arg(&height)
                    .arg(&y_len)
                    .arg(&y_kernel);
                unsafe { builder.launch(vert_cfg) }.unwrap();
            }
        }
    }
}
