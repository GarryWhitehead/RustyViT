use std::env;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use crate::device::cu_utils::*;
use crate::device::cuda::Cuda;
use crate::device::DeviceStorage;
use crate::image::{Image, PixelType};
use crate::vision::make_border::BorderMode;

trait KernelOp<T: PixelType, B: BorderMode> {
    const KERNEL_NAME: &'static str;
}

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
    fn make_border(&mut self, src: &Image<T, Self>, padding: usize, fill_value: T) -> Self::Vec
    {
        let k_path = format!(
            "{}/{}",
            env::current_dir().unwrap().to_str().unwrap(),
            "crop.cu"
        );
        let k_func = self.register_kernel(k_path.as_str(), Self::KERNEL_NAME);

        let block_dim = (32, 8, 1);
        let grid_dim = (
            div_up(src.width as u32, block_dim.0),
            div_up(src.height as u32, block_dim.1),
            1,
        );
        
        let dst_width = src.width + 2 * padding;
        let dst_height = src.height + 2 * padding;
        let dst_dmem = src
            .device
            .try_alloc(src.batch_size * src.channels * dst_width * dst_height)
            .unwrap();
        for b in 0..src.batch_size {
            let slice_base = b * src.width * src.height;
            let dst_slice_base = b * dst_width * dst_height;
            for c in 0..src.channels {
                let slice_end = slice_base + c * src.width * src.height;
                let crop_slice_end = dst_slice_base + c * dst_width * dst_height;
                let src_view = src.data.slice(slice_base..slice_end);
                let dst_view = dst_dmem.slice(dst_slice_base..crop_slice_end);
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
        dst_dmem
    }
}