use crate::device::DeviceStorage;
use crate::device::cu_utils::*;
use crate::device::cuda::Cuda;
use crate::image::{Image, PixelType};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use std::env;

const CROP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/crop.ptx"));

trait KernelOp<T: PixelType> {
    const KERNEL_NAME: &'static str;
}

impl KernelOp<u8> for Cuda {
    const KERNEL_NAME: &'static str = "crop_kernel_u8";
}
impl KernelOp<u16> for Cuda {
    const KERNEL_NAME: &'static str = "crop_kernel_u16";
}
impl KernelOp<f32> for Cuda {
    const KERNEL_NAME: &'static str = "crop_kernel_f32";
}

impl<T: PixelType> super::CropKernel<T> for Cuda
where
    Self: KernelOp<T>,
{
    fn crop(
        &mut self,
        src: &mut Image<T, Self>,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> Image<T, Self> {
        let k_func = self.register_kernel(CROP_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(crop_width as u32, block_dim.0),
            div_up(crop_height as u32, block_dim.1),
            1,
        );

        let crop_img =
            Image::try_new(src.batch_size, src.channels, crop_width, crop_height, self).unwrap();
        for b in 0..src.batch_size {
            let slice_base = b * src.strides[0];
            let crop_slice_base = b * crop_img.strides[0];
            for c in 0..src.channels {
                let slice_start = slice_base + c * src.strides[1];
                let crop_slice_start = crop_slice_base + c * crop_img.strides[1];
                let s = src.data.slice(slice_start..slice_start + src.strides[1]);
                let cs = crop_img
                    .data
                    .slice(crop_slice_start..crop_slice_start + crop_img.strides[1]);
                let mut builder = self.stream0.launch_builder(&k_func);
                builder
                    .arg(&s)
                    .arg(&src.width)
                    .arg(&crop_width)
                    .arg(&crop_width)
                    .arg(&x)
                    .arg(&y)
                    .arg(&cs);
                let cfg = LaunchConfig {
                    block_dim,
                    grid_dim,
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg) }.unwrap();
            }
        }
        crop_img
    }
}
