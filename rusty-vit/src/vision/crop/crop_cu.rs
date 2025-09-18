use crate::device::DeviceStorage;
use crate::device::cuda::Cuda;
use crate::image::{Image, PixelType};
use std::env;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::device::cu_utils::*;

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
    Self: KernelOp<T> {
    fn crop(
        &mut self,
        src: &mut Image<T, Self>,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> Self::Vec
    {
        let k_path = format!(
            "{}/{}",
            env::current_dir().unwrap().to_str().unwrap(),
            "crop.cu"
        );
        let k_func = self.register_kernel(k_path.as_str(), Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(crop_width as u32, block_dim.0),
            div_up(crop_height as u32, block_dim.1),
            1,
        );

        let crop_dmem = src
            .device
            .try_alloc(src.batch_size * src.channels * crop_width * crop_height)
            .unwrap();
        for b in 0..src.batch_size {
            let slice_base = b * src.channels * src.width * src.height;
            let crop_slice_base = b * src.channels * crop_width * crop_height;
            for c in 0..src.channels {
                let slice_end = slice_base + c * src.width * src.height;
                let crop_slice_end = crop_slice_base + c * crop_width * crop_height;
                let s = src.data.slice(slice_base..slice_end);
                let cs = crop_dmem.slice(crop_slice_base..crop_slice_end);
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
        crop_dmem
    }
}
