use crate::op_traits::MakeBorderKernel;
use crate::vulkan::device::Vulkan;
use rvit_core::pixel_traits::*;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::{compute_strides, tensor_size};
use rvit_vk_backend::public_types::ComputeWork;

// Border type ids correspond to those found in the GLSL shader.
const CONSTANT_BORDER: u32 = 0;
const CLAMP_TO_EDGE_BORDER: u32 = 1;
const MIRROR_BORDER: u32 = 2;

trait KernelOp<T: PixelType, B: BorderMode> {
    const SPIRV_NAME: &'static str;
    const BORDER_ID: u32;
}

impl KernelOp<u8, Constant> for Vulkan {
    const SPIRV_NAME: &'static str = "u8_make_border.spv";
    const BORDER_ID: u32 = CONSTANT_BORDER;
}
impl KernelOp<u16, Constant> for Vulkan {
    const SPIRV_NAME: &'static str = "u16_make_border.spv";
    const BORDER_ID: u32 = CONSTANT_BORDER;
}
impl KernelOp<f32, Constant> for Vulkan {
    const SPIRV_NAME: &'static str = "f32_make_border.spv";
    const BORDER_ID: u32 = CONSTANT_BORDER;
}

impl KernelOp<u8, ClampToEdge> for Vulkan {
    const SPIRV_NAME: &'static str = "u8_make_border.spv";
    const BORDER_ID: u32 = CLAMP_TO_EDGE_BORDER;
}
impl KernelOp<u16, ClampToEdge> for Vulkan {
    const SPIRV_NAME: &'static str = "u16_make_border.spv";
    const BORDER_ID: u32 = CLAMP_TO_EDGE_BORDER;
}
impl KernelOp<f32, ClampToEdge> for Vulkan {
    const SPIRV_NAME: &'static str = "f32_make_border.spv";
    const BORDER_ID: u32 = CLAMP_TO_EDGE_BORDER;
}

impl KernelOp<u8, Mirror> for Vulkan {
    const SPIRV_NAME: &'static str = "u8_make_border.spv";
    const BORDER_ID: u32 = MIRROR_BORDER;
}
impl KernelOp<u16, Mirror> for Vulkan {
    const SPIRV_NAME: &'static str = "u16_make_border.spv";
    const BORDER_ID: u32 = MIRROR_BORDER;
}
impl KernelOp<f32, Mirror> for Vulkan {
    const SPIRV_NAME: &'static str = "f32_make_border.spv";
    const BORDER_ID: u32 = MIRROR_BORDER;
}

impl<T: PixelType, B: BorderMode> MakeBorderKernel<T, B> for Vulkan
where
    Self: KernelOp<T, B>,
{
    fn make_border(
        &mut self,
        src: &Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        padding: usize,
    ) -> Self::Vec {
        let (batch_size, channels, width, height) =
            (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);

        let driver = self.driver.clone();
        // Initialise the UBO with the image parameters.
        let ubo_data = &[width as u32, height as u32, padding as u32];
        let ubo = self.alloc_ubo_from_slice(ubo_data);

        let dst_width = width + 2 * padding;
        let dst_height = height + 2 * padding;
        let dst_shape = [batch_size, channels, dst_width, dst_height];
        let dst_sz = tensor_size(&dst_shape);
        let dst_strides = compute_strides(&dst_shape);
        let dst_img = self.try_alloc(dst_sz);

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        for b in 0..batch_size {
            let src_slice_base = b * src_strides[0];
            let dst_slice_base = b * dst_strides[0];
            for c in 0..channels {
                let src_slice_start = src_slice_base + c * src_strides[1];
                let dst_slice_start = dst_slice_base + c * dst_strides[1];
                let src_slice_end = src_slice_start + src_strides[1];
                let src_img_slice = src.slice(src_slice_start..src_slice_end).unwrap();
                let dst_img_slice = dst_img
                    .slice(dst_slice_start..dst_slice_start + dst_strides[1])
                    .unwrap();

                // Set the specialization border mode constant - compile time evaluation of the
                // if statement means that the branching will be optimised out.
                program.bind_spec_constant(0, &[Self::BORDER_ID]);
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
                            Self::div_up(width as u32, work_size.x),
                            Self::div_up(height as u32, work_size.y),
                            1,
                        ),
                    )
                    .unwrap();
            }
            driver.borrow_mut().flush_cmds();
        }
        dst_img
    }
}
