use crate::device::vulkan::Vulkan;
use crate::image::{Image, PixelType};
use rusty_vk::public_types::ComputeWork;

trait KernelOp<TYPE> {
    const SPIRV_NAME: &'static str;
}

impl KernelOp<u8> for Vulkan {
    const SPIRV_NAME: &'static str = "u8_crop.spv";
}
impl KernelOp<u16> for Vulkan {
    const SPIRV_NAME: &'static str = "u16_crop.spv";
}
impl KernelOp<f32> for Vulkan {
    const SPIRV_NAME: &'static str = "f32_crop.spv";
}

impl<T: PixelType> super::CropKernel<T> for Vulkan
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
        let driver = self.driver.clone();
        // Initialise the UBO with the image parameters.
        let ubo_data = &[
            src.width as u32,
            src.height as u32,
            crop_width as u32,
            crop_height as u32,
            x as u32,
            y as u32,
        ];
        let ubo = self.alloc_ubo_from_slice(ubo_data);

        let crop_img =
            Image::try_new(src.batch_size, crop_width, crop_height, src.channels, self).unwrap();

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        for b in 0..src.batch_size {
            let src_slice_base = b * src.strides[0];
            let dst_slice_base = b * crop_img.strides[0];
            for c in 0..src.channels {
                let src_slice_start = src_slice_base + c * src.strides[1];
                let dst_slice_start = dst_slice_base + c * crop_img.strides[1];
                let src_slice_end = src_slice_start + src.strides[1];
                let src_img_slice = src.data.slice(src_slice_start..src_slice_end).unwrap();
                let dst_img_slice = crop_img
                    .data
                    .slice(dst_slice_start..dst_slice_start + crop_img.strides[1])
                    .unwrap();

                program
                    .try_bind_ssbo::<T>("src_image", &src_img_slice)
                    .unwrap();
                program
                    .try_bind_ssbo::<T>("dst_image", &dst_img_slice)
                    .unwrap();
                program.try_bind_ubo("image_info", &ubo).unwrap();

                let work_size = program.get_work_size();
                driver
                    .borrow_mut()
                    .dispatch_compute(
                        &program,
                        &ComputeWork::new(
                            Self::div_up(crop_width as u32, work_size.x),
                            Self::div_up(crop_height as u32, work_size.y),
                            1,
                        ),
                    )
                    .unwrap();
            }
            driver.borrow_mut().flush_cmds();
        }
        crop_img
    }
}
