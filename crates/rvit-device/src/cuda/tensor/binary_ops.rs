use crate::cuda::device::Cuda;
use crate::cuda::utils::*;
use crate::op_traits::{BinaryAddOp, BinaryDivOp, BinaryMulOp, BinaryOpKernel, BinarySubOp};
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;
use rvit_core::type_traits::FloatType;

const BINARY_OP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/binary_op.ptx"));

trait BinaryOp<T: FloatType> {
    const KERNEL_NAME: &'static str;
}

impl BinaryOp<f32> for BinaryAddOp {
    const KERNEL_NAME: &'static str = "binary_op_add_f32";
}
impl BinaryOp<f16> for BinaryAddOp {
    const KERNEL_NAME: &'static str = "binary_op_add_f16";
}

impl BinaryOp<f32> for BinarySubOp {
    const KERNEL_NAME: &'static str = "binary_op_sub_f32";
}
impl BinaryOp<f16> for BinarySubOp {
    const KERNEL_NAME: &'static str = "binary_op_sub_f16";
}

impl BinaryOp<f32> for BinaryMulOp {
    const KERNEL_NAME: &'static str = "binary_op_mul_f32";
}
impl BinaryOp<f16> for BinaryMulOp {
    const KERNEL_NAME: &'static str = "binary_op_mul_f16";
}

impl BinaryOp<f32> for BinaryDivOp {
    const KERNEL_NAME: &'static str = "binary_op_div_f32";
}
impl BinaryOp<f16> for BinaryDivOp {
    const KERNEL_NAME: &'static str = "binary_op_div_f16";
}

impl<T: FloatType, Op: BinaryOp<T>> BinaryOpKernel<T, Op> for Cuda {
    fn forward(&mut self, lhs: &mut Self::Vec, rhs: &Self::Vec, op: Op) -> Self::Vec {
        let k_func = self.register_kernel(BINARY_OP_PTX, Op::KERNEL_NAME);

        let w_size = lhs.len() as i32;
        let block_dim = (256, 1, 1);
        let grid_dim = (div_up(w_size as u32, block_dim.0), 1, 1);

        let mut builder = self.stream0.launch_builder(&k_func);
        builder.arg(&lhs).arg(&rhs).arg(&lhs).arg(&w_size);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
        lhs.clone()
    }
}
