use crate::device::vulkan::Vulkan;
use crate::image::{Image, PixelType};
use crate::vision::make_border::{BorderMode, ClampToEdge, Constant, Mirror};
use rusty_vk::public_types::ComputeWork;

const CONSTANT_BORDER: u32 = 0;
const CLAMP_TO_EDGE_BORDER: u32 = 1;
const MIRROR_BORDER: u32 = 2;

trait KernelOp<T: PixelType, B: BorderMode> {
    const SPIRV_NAME: &'static str;
    const BORDER_ID: u32;
}

impl<'a> KernelOp<u8, Constant> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u8_make_border.spv";
    const BORDER_ID: u32 = CONSTANT_BORDER;
}
impl<'a> KernelOp<u16, Constant> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u16_make_border.spv";
    const BORDER_ID: u32 = CONSTANT_BORDER;
}
impl<'a> KernelOp<f32, Constant> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "f32_make_border.spv";
    const BORDER_ID: u32 = CONSTANT_BORDER;
}

impl<'a> KernelOp<u8, ClampToEdge> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u8_make_border.spv";
    const BORDER_ID: u32 = CLAMP_TO_EDGE_BORDER;
}
impl<'a> KernelOp<u16, ClampToEdge> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u16_make_border.spv";
    const BORDER_ID: u32 = CLAMP_TO_EDGE_BORDER;
}
impl<'a> KernelOp<f32, ClampToEdge> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "f32_make_border.spv";
    const BORDER_ID: u32 = CLAMP_TO_EDGE_BORDER;
}

impl<'a> KernelOp<u8, Mirror> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u8_make_border.spv";
    const BORDER_ID: u32 = MIRROR_BORDER;
}
impl<'a> KernelOp<u16, Mirror> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "u16_make_border.spv";
    const BORDER_ID: u32 = MIRROR_BORDER;
}
impl<'a> KernelOp<f32, Mirror> for Vulkan<'a> {
    const SPIRV_NAME: &'static str = "f32_make_border.spv";
    const BORDER_ID: u32 = MIRROR_BORDER;
}

impl<T: PixelType, B: BorderMode> super::MakeBorderKernel<T, B> for Vulkan<'_>
where
    Self: KernelOp<T, B>,
{
    fn make_border(
        &mut self,
        src: &Image<T, Self>,
        padding: usize,
        fill_value: T,
    ) -> Image<T, Self> {
        let driver = self.driver.clone();
        // Initialise the UBO with the image parameters.
        let ubo_data = &[src.width as u32, src.height as u32, padding as u32];
        let ubo = self.alloc_ubo_from_slice(ubo_data);

        let dst_width = src.width + 2 * padding;
        let dst_height = src.height + 2 * padding;
        let dst_img =
            Image::try_new(src.batch_size, dst_width, dst_height, src.channels, self).unwrap();

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        for b in 0..src.batch_size {
            let src_slice_base = b * src.strides[0];
            let dst_slice_base = b * dst_img.strides[0];
            for c in 0..src.channels {
                let src_slice_start = src_slice_base + c * src.strides[1];
                let dst_slice_start = dst_slice_base + c * dst_img.strides[1];
                let src_slice_end = src_slice_start + src.strides[1];
                let src_img_slice = src.data.slice(src_slice_start..src_slice_end).unwrap();
                let dst_img_slice = dst_img
                    .data
                    .slice(dst_slice_start..dst_slice_start + dst_img.strides[1])
                    .unwrap();

                // Set the specialization border mode constant - compile time evaluation of the
                // if statement means that the branching will be optimised out.
                program.bind_spec_constant(0, &Self::BORDER_ID);
                program
                    .try_bind_ssbo::<T>("src_image", src_img_slice)
                    .unwrap();
                program
                    .try_bind_ssbo::<T>("dst_image", dst_img_slice)
                    .unwrap();
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
        dst_img
    }
}
