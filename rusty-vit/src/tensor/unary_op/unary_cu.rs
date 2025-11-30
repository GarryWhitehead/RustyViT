use crate::device::cu_utils::div_up;
use crate::device::cuda::Cuda;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
use cudarc::driver::{LaunchConfig, PushKernelArg};
use half::f16;

const UNARY_OP_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/unary_op.ptx"));

trait UnaryOp<T: FloatType> {
    const KERNEL_NAME: &'static str;
}

impl UnaryOp<f32> for super::UnarySqrOp {
    const KERNEL_NAME: &'static str = "unary_op_sqr_f32";
}
impl UnaryOp<f16> for super::UnarySqrOp {
    const KERNEL_NAME: &'static str = "unary_op_sqr_f16";
}

impl UnaryOp<f32> for super::UnarySqrtOp {
    const KERNEL_NAME: &'static str = "unary_op_sqrt_f32";
}
impl UnaryOp<f16> for super::UnarySqrtOp {
    const KERNEL_NAME: &'static str = "unary_op_sqrt_f16";
}

impl UnaryOp<f32> for super::UnaryExpOp {
    const KERNEL_NAME: &'static str = "unary_op_exp_f32";
}
impl UnaryOp<f16> for super::UnaryExpOp {
    const KERNEL_NAME: &'static str = "unary_op_exp_f16";
}

impl UnaryOp<f32> for super::UnaryTanhOp {
    const KERNEL_NAME: &'static str = "unary_op_tanh_f32";
}
impl UnaryOp<f16> for super::UnaryTanhOp {
    const KERNEL_NAME: &'static str = "unary_op_tanh_f16";
}

impl UnaryOp<f32> for super::UnaryCosOp {
    const KERNEL_NAME: &'static str = "unary_op_cos_f32";
}
impl UnaryOp<f16> for super::UnaryCosOp {
    const KERNEL_NAME: &'static str = "unary_op_cos_f16";
}

impl UnaryOp<f32> for super::UnarySinOp {
    const KERNEL_NAME: &'static str = "unary_op_sin_f32";
}
impl UnaryOp<f16> for super::UnarySinOp {
    const KERNEL_NAME: &'static str = "unary_op_sin_f16";
}

impl UnaryOp<f32> for super::UnaryAbsOp {
    const KERNEL_NAME: &'static str = "unary_op_abs_f32";
}
impl UnaryOp<f16> for super::UnaryAbsOp {
    const KERNEL_NAME: &'static str = "unary_op_abs_f16";
}

impl UnaryOp<f32> for super::UnaryGeluOp {
    const KERNEL_NAME: &'static str = "unary_op_gelu_f32";
}
impl UnaryOp<f16> for super::UnaryGeluOp {
    const KERNEL_NAME: &'static str = "unary_op_gelu_f16";
}

impl UnaryOp<f32> for super::UnaryReluOp {
    const KERNEL_NAME: &'static str = "unary_op_relu_f32";
}
impl UnaryOp<f16> for super::UnaryReluOp {
    const KERNEL_NAME: &'static str = "unary_op_relu_f16";
}

impl UnaryOp<f32> for super::UnaryLogOp {
    const KERNEL_NAME: &'static str = "unary_op_log_f32";
}
impl UnaryOp<f16> for super::UnaryLogOp {
    const KERNEL_NAME: &'static str = "unary_op_log_f16";
}

impl UnaryOp<f32> for super::UnaryFloorOp {
    const KERNEL_NAME: &'static str = "unary_op_floor_f32";
}
impl UnaryOp<f16> for super::UnaryFloorOp {
    const KERNEL_NAME: &'static str = "unary_op_floor_f16";
}

impl UnaryOp<f32> for super::UnaryCeilOp {
    const KERNEL_NAME: &'static str = "unary_op_ceil_f32";
}
impl UnaryOp<f16> for super::UnaryCeilOp {
    const KERNEL_NAME: &'static str = "unary_op_ceil_f16";
}

impl<T: FloatType, Op: UnaryOp<T>> super::UnaryOpExecutor<T, Op> for Cuda {
    fn forward(&mut self, x: &Tensor<T, Self>, _op: Op) -> Tensor<T, Self> {
        let k_func = self.register_kernel(UNARY_OP_PTX, Op::KERNEL_NAME);

        let w_size = x.data.len() as i32;
        let block_dim = (256, 1, 1);
        let grid_dim = (div_up(w_size as u32, block_dim.0), 1, 1);

        let mut builder = self.stream0.launch_builder(&k_func);
        builder.arg(&x.data).arg(&x.data).arg(&w_size);

        let cfg = LaunchConfig {
            block_dim,
            grid_dim,
            shared_mem_bytes: 0,
        };
        unsafe { builder.launch(cfg) }.unwrap();
        x.clone()
    }
}
