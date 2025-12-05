use crate::vision_traits::ResizeKernel;
use crate::vulkan::device::Vulkan;
use rvit_core::pixel_traits::*;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::{compute_strides, tensor_size};
use rvit_vk_backend::public_types::ComputeWork;

// Filter type ids correspond to those found in the GLSL shader.
const FILTER_OP_BILINEAR: u32 = 0;

trait KernelOp<T: PixelType, I: InterpMode> {
    const SPIRV_NAME: &'static str;
    const FILTER_OP: u32;
}

impl KernelOp<u8, Bilinear> for Vulkan {
    const SPIRV_NAME: &'static str = "u8_resize.spv";
    const FILTER_OP: u32 = FILTER_OP_BILINEAR;
}
impl KernelOp<u16, Bilinear> for Vulkan {
    const SPIRV_NAME: &'static str = "u16_resize.spv";
    const FILTER_OP: u32 = FILTER_OP_BILINEAR;
}
impl KernelOp<f32, Bilinear> for Vulkan {
    const SPIRV_NAME: &'static str = "f32_resize.spv";
    const FILTER_OP: u32 = FILTER_OP_BILINEAR;
}

#[allow(dead_code)]
struct ResizeUbo {
    src_width: u32,
    src_height: u32,
    dst_width: u32,
    dst_height: u32,
    scale_x: f32,
    scale_y: f32,
}

impl<T: PixelType, I: InterpMode> ResizeKernel<T, I> for Vulkan
where
    Self: KernelOp<T, I>,
{
    fn resize(
        &mut self,
        src: &Self::Vec,
        in_shape: &[usize],
        in_strides: &[usize],
        dst_width: usize,
        dst_height: usize,
    ) -> Self::Vec {
        let (batch_size, channels, width, height) =
            (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);

        let driver = self.driver.clone();

        // Initialise the UBO with the image parameters.
        let ubo_data = ResizeUbo {
            src_width: width as u32,
            src_height: height as u32,
            dst_width: dst_width as u32,
            dst_height: dst_height as u32,
            scale_x: width as f32 / dst_width as f32,
            scale_y: height as f32 / dst_height as f32,
        };
        let ubo = self.alloc_ubo_from_slice(&[ubo_data]);

        let dst_shape = [batch_size, channels, dst_width, dst_height];
        let dst_sz = tensor_size(&dst_shape);
        let dst_strides = compute_strides(&dst_shape);
        let dst_img = self.try_alloc(dst_sz).unwrap();

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        for b in 0..batch_size {
            let src_slice_base = b * in_strides[0];
            let dst_slice_base = b * dst_strides[0];
            for c in 0..channels {
                let src_slice_start = src_slice_base + c * in_strides[1];
                let dst_slice_start = dst_slice_base + c * dst_strides[1];
                let src_slice_end = src_slice_start + in_strides[1];
                let src_img_slice = src.slice(src_slice_start..src_slice_end).unwrap();
                let dst_img_slice = dst_img
                    .slice(dst_slice_start..dst_slice_start + dst_strides[1])
                    .unwrap();

                // Set the specialization border mode constant - compile time evaluation of the
                // if statement means that the branching will be optimised out.
                program.bind_spec_constant(0, &[Self::FILTER_OP]);
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
