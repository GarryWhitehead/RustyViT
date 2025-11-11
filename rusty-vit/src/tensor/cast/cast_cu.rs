use crate::device::cu_utils::div_up;
use crate::device::cuda::Cuda;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
use cudarc::driver::{LaunchConfig, PushKernelArg};

const CAST_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cast.ptx"));

trait KernelOp<T: FloatType, O: FloatType> {
    const KERNEL_NAME: &'static str;
}
impl KernelOp<f32, half::f16> for Cuda {
    const KERNEL_NAME: &'static str = "cast_f32_f16_kernel";
}
impl KernelOp<half::f16, f32> for Cuda {
    const KERNEL_NAME: &'static str = "cast_f16_f32_kernel";
}

impl<T: FloatType, O: FloatType> super::CastKernel<T, O> for Cuda
where
    Self: KernelOp<T, O>,
{
    fn cast(&mut self, lhs: &Tensor<T, Self>) -> Tensor<O, Self> {
        let k_func = self.register_kernel(CAST_PTX, Self::KERNEL_NAME);
        let out = Tensor::try_new(&*lhs.shape, self).unwrap();

        let block_dim = (256, 1, 1);
        let grid_dim = (div_up(out.data.len() as u32, block_dim.0), 1, 1);
        let n = lhs.data.len();

        let mut builder = self.stream0.launch_builder(&k_func);
        builder.arg(&lhs.data).arg(&out.data).arg(&n);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
        out
    }
}
