use crate::op_traits::CastKernel;
use crate::vulkan::device::Vulkan;
use rvit_core::element_traits::DataElem;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::*;
use rvit_vk_backend::public_types::ComputeWork;

trait KernelOp<T: DataElem, O: DataElem> {
    const SPIRV_NAME: &'static str;
}
impl KernelOp<f32, half::f16> for Vulkan {
    const SPIRV_NAME: &'static str = "cast_fp32_to_fp16.spv";
}
impl KernelOp<half::f16, f32> for Vulkan {
    const SPIRV_NAME: &'static str = "cast_fp16_to_fp32.spv";
}

impl<T: DataElem, O: DataElem> CastKernel<T, O> for Vulkan
where
    Self: KernelOp<T, O>,
{
    fn forward(&self, x: &Self::Vec, shape: &[usize]) -> Self::Vec {
        let driver = self.driver.clone();
        let sz = tensor_size(&shape);
        let out = Self::try_alloc(self, sz).unwrap();

        let ubo = self.alloc_ubo_from_slice(&[sz]);

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        program.try_bind_ubo("params", &ubo).unwrap();
        program
            .try_bind_ssbo::<T>("in_matrix", &x.slice(..).unwrap())
            .unwrap();
        program
            .try_bind_ssbo::<O>("out_matrix", &out.slice(..).unwrap())
            .unwrap();

        let work_size = program.get_work_size();
        driver
            .borrow_mut()
            .dispatch_compute(
                &program,
                &ComputeWork::new(Self::div_up(sz as u32, work_size.x), 1, 1),
            )
            .unwrap();
        driver.borrow_mut().flush_cmds();

        out
    }
}
