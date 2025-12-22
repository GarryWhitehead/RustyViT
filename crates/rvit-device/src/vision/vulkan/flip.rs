use crate::op_traits::HorizontalFlipKernel;
use crate::vulkan::device::Vulkan;
use rand::Rng;
use rvit_core::pixel_traits::PixelType;
use rvit_vk_backend::public_types::ComputeWork;

trait KernelOp<TYPE> {
    const SPIRV_NAME: &'static str;
}
impl KernelOp<u8> for Vulkan {
    const SPIRV_NAME: &'static str = "u8_flip.spv";
}
impl KernelOp<u16> for Vulkan {
    const SPIRV_NAME: &'static str = "u16_flip.spv";
}
impl KernelOp<f32> for Vulkan {
    const SPIRV_NAME: &'static str = "f32_flip.spv";
}

impl<T: PixelType> HorizontalFlipKernel<T> for Vulkan
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

        let driver = self.driver.clone();
        // Initialise the UBO with the image parameters.
        let ubo_data = &[width as u32, height as u32];
        let ubo = self.alloc_ubo_from_slice(ubo_data);

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        let mut rng = rand::rng();
        for b in 0..batch_size {
            if rng.random_range(0.0..1.0) < prob {
                let slice_base = b * src_strides[0];
                for c in 0..channels {
                    let slice_start = slice_base + c * src_strides[1];
                    let slice_end = slice_start + src_strides[1];
                    let img_slice = src.slice(slice_start..slice_end).unwrap();

                    program.try_bind_ssbo::<T>("src_image", &img_slice).unwrap();
                    program.try_bind_ubo("image_info", &ubo).unwrap();

                    let work_size = program.get_work_size();
                    driver
                        .borrow_mut()
                        .dispatch_compute(
                            &program,
                            &ComputeWork::new(
                                Self::div_up(width as u32, work_size.x),
                                Self::div_up(height as u32, work_size.y),
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
