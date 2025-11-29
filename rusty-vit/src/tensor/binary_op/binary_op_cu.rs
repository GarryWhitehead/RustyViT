use crate::device::cu_utils::div_up;
use crate::device::cuda::Cuda;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;

const BINARY_OP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/binary_op.ptx"));

trait BinaryOp<T: FloatType> {
    const KERNEL_NAME: &'static str;
}

impl BinaryOp<f32> for super::BinaryAddOp {
    const KERNEL_NAME: &'static str = "binary_op_add_f32";
}
impl BinaryOp<f16> for super::BinaryAddOp {
    const KERNEL_NAME: &'static str = "binary_op_add_f16";
}

impl BinaryOp<f32> for super::BinarySubOp {
    const KERNEL_NAME: &'static str = "binary_op_sub_f32";
}
impl BinaryOp<f16> for super::BinarySubOp {
    const KERNEL_NAME: &'static str = "binary_op_sub_f16";
}

impl BinaryOp<f32> for super::BinaryMulOp {
    const KERNEL_NAME: &'static str = "binary_op_mul_f32";
}
impl BinaryOp<f16> for super::BinaryMulOp {
    const KERNEL_NAME: &'static str = "binary_op_mul_f16";
}

impl BinaryOp<f32> for super::BinaryDivOp {
    const KERNEL_NAME: &'static str = "binary_op_div_f32";
}
impl BinaryOp<f16> for super::BinaryDivOp {
    const KERNEL_NAME: &'static str = "binary_op_div_f16";
}

impl<T: FloatType, Op: BinaryOp<T>> super::BinaryOpExecutor<T, Op> for Cuda {
    fn forward(
        &mut self,
        lhs: &Tensor<T, Self>,
        rhs: &Tensor<T, Self>,
        _op: Op,
    ) -> Tensor<T, Self> {
        let k_func = self.register_kernel(BINARY_OP_PTX, Op::KERNEL_NAME);

        let w_size = lhs.data.len() as i32;
        let block_dim = (256, 1, 1);
        let grid_dim = (div_up(w_size as u32, block_dim.0), 1, 1);

        let mut builder = self.stream0.launch_builder(&k_func);
        builder
            .arg(&lhs.data)
            .arg(&rhs.data)
            .arg(&lhs.data)
            .arg(&w_size);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
        lhs.clone()
    }
}
