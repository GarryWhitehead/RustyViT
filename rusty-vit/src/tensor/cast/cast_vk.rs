use crate::device::vulkan::Vulkan;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
use rusty_vk::public_types::ComputeWork;

trait KernelOp<T: FloatType, O: FloatType> {
    const SPIRV_NAME: &'static str;
}
impl KernelOp<f32, half::f16> for Vulkan {
    const SPIRV_NAME: &'static str = "cast_fp32_to_fp16.spv";
}
impl KernelOp<half::f16, f32> for Vulkan {
    const SPIRV_NAME: &'static str = "cast_fp16_to_fp32.spv";
}

impl<T: FloatType, O: FloatType> super::CastKernel<T, O> for Vulkan
where
    Self: KernelOp<T, O>,
{
    fn cast(&mut self, lhs: &Tensor<T, Self>) -> Tensor<O, Self> {
        let driver = self.driver.clone();
        let out: Tensor<O, _> = Tensor::try_new(&*lhs.shape, self).unwrap();

        let sz = lhs.total_size() as u32;
        let ubo = self.alloc_ubo_from_slice(&[sz]);

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        program.try_bind_ubo("params", &ubo).unwrap();
        program
            .try_bind_ssbo::<T>("in_matrix", &lhs.data.slice(..).unwrap())
            .unwrap();
        program
            .try_bind_ssbo::<O>("out_matrix", &out.data.slice(..).unwrap())
            .unwrap();

        let work_size = program.get_work_size();
        driver
            .borrow_mut()
            .dispatch_compute(
                &program,
                &ComputeWork::new(Self::div_up(sz, work_size.x), 1, 1),
            )
            .unwrap();
        driver.borrow_mut().flush_cmds();

        out
    }
}
