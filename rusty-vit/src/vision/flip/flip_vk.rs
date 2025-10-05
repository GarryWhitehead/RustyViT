use crate::device::vulkan::Vulkan;
use crate::image::{Image, PixelType};
use rand::Rng;
use rusty_vk::public_types::ComputeWork;

trait KernelOp<TYPE> {
    const SPIRV_NAME: &'static str;
}
impl<'a> KernelOp<u8> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u8_flip.spv";
}
impl<'a> KernelOp<u16> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u16_flip.spv";
}
impl<'a> KernelOp<f32> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "f32_flip.spv";
}

impl<'a, T: PixelType> super::HorizFlipKernel<T> for Vulkan<'a>
where
    Self: KernelOp<T>,
{
    fn flip_horizontal(&mut self, src: &mut Image<T, Self>, prob: f32) {
        let driver = self.driver.clone();
        // Initialise the UBO with the image parameters.
        let ubo_data = &[src.width as u32, src.height as u32];
        let ubo = self.alloc_ubo_from_slice(ubo_data);

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        let mut rng = rand::rng();
        for b in 0..src.batch_size {
            if rng.random_range(0.0..1.0) < prob {
                let slice_base = b * src.strides[0];
                for c in 0..src.channels {
                    let slice_start = slice_base + c * src.strides[1];
                    let slice_end = slice_start + src.strides[1];
                    let img_slice = src.data.slice(slice_start..slice_end).unwrap();

                    program.try_bind_ssbo::<T>("src_image", img_slice).unwrap();
                    program.try_bind_ubo("image_info", &ubo).unwrap();

                    driver
                        .borrow_mut()
                        .dispatch_compute(
                            &program,
                            &ComputeWork::new(
                                (32 + src.width as u32 - 1) / 32,
                                (8 + src.height as u32 - 1) / 8,
                                1,
                            ),
                        )
                        .unwrap();
                }
                driver.borrow_mut().flush_cmds();
            }
        }
    }
}
