use crate::device::vulkan::Vulkan;
use crate::image::{Image, PixelType};
use crate::type_traits::FloatType;
use crate::vision::convolution::{Conv, Kernel};
use rusty_vk::public_types::ComputeWork;

const HORIZ_TILE_WIDTH: i32 = 16;
// The tile height for the horizontal pass.
const HORIZ_TILE_HEIGHT: i32 = 4;
const HORIZ_RESULT_STEPS: i32 = 8;
// The tile width for the vertical pass.
const VERT_TILE_WIDTH: i32 = 16;
// The tile height for the vertical pass.
const VERT_TILE_HEIGHT: i32 = 8;
const VERT_RESULT_STEPS: i32 = 8;

pub(crate) trait KernelOp<T: PixelType> {
    const HORIZ_SPIRV_NAME: &'static str;
    const VERT_SPIRV_NAME: &'static str;
}

impl KernelOp<u8> for Vulkan {
    const HORIZ_SPIRV_NAME: &'static str = "u8_conv_horiz.spv";
    const VERT_SPIRV_NAME: &'static str = "u8_conv_vert.spv";
}
impl KernelOp<u16> for Vulkan {
    const HORIZ_SPIRV_NAME: &'static str = "u16_conv_horiz.spv";
    const VERT_SPIRV_NAME: &'static str = "u16_conv_vert.spv";
}
impl KernelOp<f32> for Vulkan {
    const HORIZ_SPIRV_NAME: &'static str = "f32_conv_horiz.spv";
    const VERT_SPIRV_NAME: &'static str = "f32_conv_vert.spv";
}

impl<T: PixelType, F: FloatType> Conv<T, F> for Vulkan
where
    Self: KernelOp<T>,
{
    fn convolution(
        &mut self,
        src: &mut Image<T, Self>,
        x_kernel: &Kernel<F, Self>,
        y_kernel: &Kernel<F, Self>,
    ) {
        if src.width as u32 % HORIZ_TILE_WIDTH as u32 != 0 {
            panic!(
                "kernel width is not a multiple of the tile width {}",
                HORIZ_TILE_WIDTH
            );
        }
        if src.height as u32 % VERT_TILE_HEIGHT as u32 != 0 {
            panic!(
                "kernel height is not a multiple of the tile height {}",
                VERT_TILE_HEIGHT
            );
        }

        let driver = self.driver.clone();
        // Initialise the UBO with the image parameters.
        let h_ubo_data = &[
            src.width as i32,
            src.height as i32,
            x_kernel.data.element_count() as i32,
        ];
        let h_ubo = self.alloc_ubo_from_slice(h_ubo_data);

        let v_ubo_data = &[
            src.width as i32,
            src.height as i32,
            y_kernel.data.element_count() as i32,
        ];
        let v_ubo = self.alloc_ubo_from_slice(v_ubo_data);

        let tmp_img: Image<T, _> = Image::try_new(1, src.width, src.height, 1, self).unwrap();

        let mut h_program = self
            .try_get_module(env!("OUT_DIR"), Self::HORIZ_SPIRV_NAME)
            .unwrap();
        let mut v_program = self
            .try_get_module(env!("OUT_DIR"), Self::VERT_SPIRV_NAME)
            .unwrap();

        let h_work = ComputeWork::new(
            src.width as u32 / (HORIZ_RESULT_STEPS * HORIZ_TILE_WIDTH) as u32,
            src.height as u32 / HORIZ_TILE_HEIGHT as u32,
            1,
        );
        let v_work = ComputeWork::new(
            src.width as u32 / VERT_TILE_WIDTH as u32,
            src.height as u32 / (VERT_RESULT_STEPS * VERT_TILE_HEIGHT) as u32,
            1,
        );

        for b in 0..src.batch_size {
            let slice_base = b * src.strides[0];
            for c in 0..src.channels {
                let slice_start = slice_base + c * src.strides[1];
                let slice_end = slice_start + src.strides[1];
                let src_img_slice = src.data.slice(slice_start..slice_end).unwrap();

                // Horizontal pass.
                h_program.bind_spec_constant(0, &[HORIZ_TILE_WIDTH]);
                h_program.bind_spec_constant(1, &[HORIZ_TILE_HEIGHT]);
                h_program.bind_spec_constant(3, &[HORIZ_RESULT_STEPS]);
                h_program
                    .try_bind_ssbo::<T>("src_image", &src_img_slice)
                    .unwrap();
                h_program
                    .try_bind_ssbo::<T>("dst_image", &tmp_img.data.slice(..).unwrap())
                    .unwrap();
                h_program
                    .try_bind_ssbo::<F>("kernel", &x_kernel.data.slice(..).unwrap())
                    .unwrap();
                h_program.try_bind_ubo("image_info", &h_ubo).unwrap();

                driver
                    .borrow_mut()
                    .dispatch_compute(&h_program, &h_work)
                    .unwrap();
                driver.borrow_mut().write_read_barrier();

                // Vertical pass.
                v_program.bind_spec_constant(0, &[VERT_TILE_WIDTH]);
                v_program.bind_spec_constant(1, &[VERT_TILE_HEIGHT]);
                v_program.bind_spec_constant(3, &[VERT_RESULT_STEPS]);
                v_program
                    .try_bind_ssbo::<T>("src_image", &tmp_img.data.slice(..).unwrap())
                    .unwrap();
                v_program
                    .try_bind_ssbo::<T>("dst_image", &src_img_slice)
                    .unwrap();
                v_program
                    .try_bind_ssbo::<F>("kernel", &y_kernel.data.slice(..).unwrap())
                    .unwrap();
                v_program.try_bind_ubo("image_info", &v_ubo).unwrap();

                driver
                    .borrow_mut()
                    .dispatch_compute(&v_program, &v_work)
                    .unwrap();
                driver.borrow_mut().write_read_barrier();
            }
            driver.borrow_mut().flush_cmds();
        }
    }
}
