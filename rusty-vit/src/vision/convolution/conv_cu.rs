use super::*;
use crate::device::cuda::Cuda;
use crate::image::{Image, PixelType};
use crate::type_traits::FloatType;
use cudarc::driver::{LaunchConfig, PushKernelArg};

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
impl<T: PixelType, F: FloatType> Conv<T, F> for Cuda
where
    Self: KernelOp<T>,
{
    fn convolution(
        &mut self,
        src: &mut Image<T, Self>,
        x_kernel: &Kernel<F, Self>,
        y_kernel: &Kernel<F, Self>,
    ) {
        if src.width as u32 % HORIZ_TILE_WIDTH != 0 {
            panic!(
                "kernel width is not a multiple of the tile width {}",
                HORIZ_TILE_WIDTH
            );
        }
        if src.height as u32 % VERT_TILE_HEIGHT != 0 {
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
                src.width as u32 / (HORIZ_RESULT_STEPS * HORIZ_TILE_WIDTH),
                src.height as u32 / HORIZ_TILE_HEIGHT,
                1,
            ),
            shared_mem_bytes: 0,
        };
        let vert_cfg = LaunchConfig {
            block_dim: (VERT_TILE_WIDTH, VERT_TILE_HEIGHT, 1),
            grid_dim: (
                src.width as u32 / VERT_TILE_WIDTH,
                src.height as u32 / (VERT_RESULT_STEPS * VERT_TILE_HEIGHT),
                1,
            ),
            shared_mem_bytes: 0,
        };

        let tmp_img: Image<T, _> = Image::try_new(1, src.width, src.height, 1, self).unwrap();

        for b in 0..src.batch_size {
            let batch_offset = b * src.strides[0];
            for c in 0..src.channels {
                let start_idx = batch_offset + c * src.strides[1];
                let end_idx = start_idx + src.strides[1];
                let src_slice = src.data.slice(start_idx..end_idx);

                // Horizontal pass. src -> tmp image
                let mut builder = self.stream0.launch_builder(&k_horiz_func);
                let x_len = x_kernel.data.len();
                builder
                    .arg(&src_slice)
                    .arg(&tmp_img.data)
                    .arg(&src.width)
                    .arg(&src.height)
                    .arg(&x_len)
                    .arg(&x_kernel.data);
                unsafe { builder.launch(horiz_cfg) }.unwrap();

                // Vertical pass.
                let mut builder = self.stream0.launch_builder(&k_vert_func);
                let y_len = y_kernel.data.len();
                builder
                    .arg(&tmp_img.data)
                    .arg(&src_slice)
                    .arg(&tmp_img.width)
                    .arg(&tmp_img.height)
                    .arg(&y_len)
                    .arg(&y_kernel.data);
                unsafe { builder.launch(vert_cfg) }.unwrap();
            }
        }
    }
}
