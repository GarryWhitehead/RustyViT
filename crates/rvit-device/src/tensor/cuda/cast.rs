use crate::cuda::device::Cuda;
use crate::cuda::utils::*;
use crate::op_traits::CastKernel;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use rvit_core::element_traits::DataElem;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::*;

const CAST_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/cast.ptx"));

trait KernelOp<T: DataElem, O: DataElem> {
    const KERNEL_NAME: &'static str;
}
impl KernelOp<f32, half::f16> for Cuda {
    const KERNEL_NAME: &'static str = "cast_f32_f16_kernel";
}
impl KernelOp<half::f16, f32> for Cuda {
    const KERNEL_NAME: &'static str = "cast_f16_f32_kernel";
}

impl<T: DataElem, O: DataElem> CastKernel<T, O> for Cuda {
    fn forward(&self, x: &Self::Vec, shape: &[usize]) -> Self::Vec {
        let k_func = self.register_kernel(CAST_PTX, Self::KERNEL_NAME);
        let sz = tensor_size(&shape);
        let out = Self::try_alloc(self, sz).unwrap();

        let block_dim = (256, 1, 1);
        let grid_dim = (div_up(out.len() as u32, block_dim.0), 1, 1);
        let n = x.len();

        let mut builder = self.stream0.launch_builder(&k_func);
        builder.arg(&x).arg(&out).arg(&n);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
        out
    }
}
