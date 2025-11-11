use crate::device::vulkan::Vulkan;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
use rusty_vk::public_types::ComputeWork;

#[allow(dead_code)]
struct MatrixUbo<T: FloatType> {
    m: u32,
    k: u32,
    n: u32,
    lda: u32,
    ldb: u32,
    ldc: u32,
    alpha: T,
    beta: T,
}

pub(crate) trait KernelOp<T: FloatType> {
    const SPIRV_NAME: &'static str;
}
impl KernelOp<f32> for Vulkan {
    const SPIRV_NAME: &'static str = "f32_matmul.spv";
}
impl KernelOp<half::f16> for Vulkan {
    const SPIRV_NAME: &'static str = "f16_matmul.spv";
}

impl<T: FloatType> super::MatMulKernel<T> for Vulkan
where
    Self: KernelOp<T>,
{
    fn matmul(&mut self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>) -> Tensor<T, Self> {
        let driver = self.driver.clone();

        let (m, k, n) = super::inner_shape(&lhs.shape, &rhs.shape);
        let out_shape = super::compute_shape(&lhs.shape, m, n);
        let out = Tensor::try_new(&out_shape, self).unwrap();
        let matrix_c: Tensor<T, _> = Tensor::try_new(&out_shape, self).unwrap();

        let ubo_data = MatrixUbo {
            m: m as u32,
            n: n as u32,
            k: k as u32,
            lda: m as u32,
            ldb: k as u32,
            ldc: m as u32,
            alpha: T::one(),
            beta: T::zero(),
        };
        let ubo = self.alloc_ubo_from_slice(&[ubo_data]);

        let mut program = self
            .try_get_module(env!("OUT_DIR"), Self::SPIRV_NAME)
            .unwrap();

        let tile_m = if m < 16 { 1 } else { 16 };
        let tile_n = if n < 16 { 1 } else { 16 };

        program.try_bind_ubo("m_params", &ubo).unwrap();
        program
            .try_bind_ssbo::<T>("m_a", &lhs.data.slice(..).unwrap())
            .unwrap();
        program
            .try_bind_ssbo::<T>("m_b", &rhs.data.slice(..).unwrap())
            .unwrap();
        program
            .try_bind_ssbo::<T>("m_c", &matrix_c.data.slice(..).unwrap())
            .unwrap();
        program
            .try_bind_ssbo::<T>("m_d", &out.data.slice(..).unwrap())
            .unwrap();
        program.bind_spec_constant(0, &[tile_m as u32]);
        program.bind_spec_constant(2, &[tile_n as u32]);

        driver
            .borrow_mut()
            .dispatch_compute(
                &program,
                &ComputeWork::new((m / tile_m) as u32, (n / tile_n) as u32, 1),
            )
            .unwrap();
        driver.borrow_mut().flush_cmds();
        out
    }
}
