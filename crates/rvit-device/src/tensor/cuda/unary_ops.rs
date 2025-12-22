use crate::cuda::device::Cuda;
use crate::cuda::utils::*;
use crate::op_traits::*;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;
use rvit_core::element_traits::DataElem;

const UNARY_OP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/unary_op.ptx"));

trait UnaryOp<T: DataElem> {
    const KERNEL_NAME: &'static str;
}

impl UnaryOp<f32> for UnarySqrOp {
    const KERNEL_NAME: &'static str = "unary_op_sqr_f32";
}
impl UnaryOp<f16> for UnarySqrOp {
    const KERNEL_NAME: &'static str = "unary_op_sqr_f16";
}

impl UnaryOp<f32> for UnarySqrtOp {
    const KERNEL_NAME: &'static str = "unary_op_sqrt_f32";
}
impl UnaryOp<f16> for UnarySqrtOp {
    const KERNEL_NAME: &'static str = "unary_op_sqrt_f16";
}

impl UnaryOp<f32> for UnaryExpOp {
    const KERNEL_NAME: &'static str = "unary_op_exp_f32";
}
impl UnaryOp<f16> for UnaryExpOp {
    const KERNEL_NAME: &'static str = "unary_op_exp_f16";
}

impl UnaryOp<f32> for UnaryTanhOp {
    const KERNEL_NAME: &'static str = "unary_op_tanh_f32";
}
impl UnaryOp<f16> for UnaryTanhOp {
    const KERNEL_NAME: &'static str = "unary_op_tanh_f16";
}

impl UnaryOp<f32> for UnaryCosOp {
    const KERNEL_NAME: &'static str = "unary_op_cos_f32";
}
impl UnaryOp<f16> for UnaryCosOp {
    const KERNEL_NAME: &'static str = "unary_op_cos_f16";
}

impl UnaryOp<f32> for UnarySinOp {
    const KERNEL_NAME: &'static str = "unary_op_sin_f32";
}
impl UnaryOp<f16> for UnarySinOp {
    const KERNEL_NAME: &'static str = "unary_op_sin_f16";
}

impl UnaryOp<f32> for UnaryAbsOp {
    const KERNEL_NAME: &'static str = "unary_op_abs_f32";
}
impl UnaryOp<f16> for UnaryAbsOp {
    const KERNEL_NAME: &'static str = "unary_op_abs_f16";
}

impl UnaryOp<f32> for UnaryGeluOp {
    const KERNEL_NAME: &'static str = "unary_op_gelu_f32";
}
impl UnaryOp<f16> for UnaryGeluOp {
    const KERNEL_NAME: &'static str = "unary_op_gelu_f16";
}

impl UnaryOp<f32> for UnaryReluOp {
    const KERNEL_NAME: &'static str = "unary_op_relu_f32";
}
impl UnaryOp<f16> for UnaryReluOp {
    const KERNEL_NAME: &'static str = "unary_op_relu_f16";
}

impl UnaryOp<f32> for UnaryLogOp {
    const KERNEL_NAME: &'static str = "unary_op_log_f32";
}
impl UnaryOp<f16> for UnaryLogOp {
    const KERNEL_NAME: &'static str = "unary_op_log_f16";
}

impl UnaryOp<f32> for UnaryFloorOp {
    const KERNEL_NAME: &'static str = "unary_op_floor_f32";
}
impl UnaryOp<f16> for UnaryFloorOp {
    const KERNEL_NAME: &'static str = "unary_op_floor_f16";
}

impl UnaryOp<f32> for UnaryCeilOp {
    const KERNEL_NAME: &'static str = "unary_op_ceil_f32";
}
impl UnaryOp<f16> for UnaryCeilOp {
    const KERNEL_NAME: &'static str = "unary_op_ceil_f16";
}

impl<T: DataElem, Op: UnaryOp<T>> UnaryOpKernel<T, Op> for Cuda {
    fn forward(&mut self, x: &Self::Vec, _op: Op) -> Self::Vec {
        let k_func = self.register_kernel(UNARY_OP_PTX, Op::KERNEL_NAME);

        let w_size = x.len() as i32;
        let block_dim = (256, 1, 1);
        let grid_dim = (div_up(w_size as u32, block_dim.0), 1, 1);

        let mut builder = self.stream0.launch_builder(&k_func);
        builder.arg(&x).arg(&x).arg(&w_size);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
        *x
    }
}
