use crate::cuda::device::Cuda;
use crate::cuda::utils::*;
use crate::vision_traits::CropKernel;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::{compute_strides, tensor_size};
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

impl<T: PixelType> CropKernel<T> for Cuda
where
    Self: KernelOp<T>,
{
    fn crop(
        &mut self,
        src: &mut Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> Self::Vec {
        let (batch_size, channels, src_width, src_height) =
            (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
        let k_func = self.register_kernel(CROP_PTX, Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(crop_width as u32, block_dim.0),
            div_up(crop_height as u32, block_dim.1),
            1,
        );

        let crop_shape = [batch_size, channels, crop_width, crop_height];
        let crop_sz = tensor_size(&crop_shape);
        let crop_strides = compute_strides(&crop_shape);
        let crop_img = self.try_alloc(crop_sz).unwrap();

        for b in 0..batch_size {
            let slice_base = b * src_strides[0];
            let crop_slice_base = b * crop_strides[0];
            for c in 0..channels {
                let slice_start = slice_base + c * src_strides[1];
                let crop_slice_start = crop_slice_base + c * crop_strides[1];
                let s = src.slice(slice_start..slice_start + src_strides[1]);
                let cs = crop_img.slice(crop_slice_start..crop_slice_start + crop_strides[1]);
                let mut builder = self.stream0.launch_builder(&k_func);
                builder
                    .arg(&s)
                    .arg(&src_width)
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
