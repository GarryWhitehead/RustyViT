use crate::device::{DAlloc, Device};

use crate::element_traits::{DataElem, FloatElem};
use crate::memory::storage::DeviceStorage;

pub trait MatMulKernel<T>: Device {
    fn matmul_fwd(
        &mut self,
        lhs: &DAlloc<Self>,
        lhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs: &DAlloc<Self>,
        rhs_shape: &[usize],
        rhs_strides: &[usize],
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>);
}

pub trait CastKernel<O: DataElem>: Device {
    fn cast_fwd(&mut self, x: &DAlloc<Self>) -> DAlloc<Self>;
}

pub trait Conv2dKernel<T>: Device {
    fn conv2d_fwd(
        &mut self,
        x: &DAlloc<Self>,
        shape: &[usize],
        strides: &[usize],
        filters: &DAlloc<Self>,
        f_shape: &[usize],
        out_shape: &[usize],
        stride: usize,
        padding: usize,
        groups: usize,
        to_nhwc: bool,
    ) -> DAlloc<Self>;
}

pub trait ConvConvertKernel<T>: Device {
    fn im2col(
        &mut self,
        x: &DAlloc<Self>,
        x_shape: &[usize],
        f_shape: &[usize],
        batch_size: usize,
        out_width: usize,
        out_height: usize,
        stride: usize,
        padding: usize,
        is_nhwc: bool,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>);
    fn nhwc_to_nchw(&mut self, x: &DAlloc<Self>, shape: &[usize]) -> DAlloc<Self>;
    fn nchw_to_nhwc(&mut self, x: &DAlloc<Self>, shape: &[usize]) -> DAlloc<Self>;
}

pub struct BinaryAddOp;
pub struct BinarySubOp;
pub struct BinaryMulOp;
pub struct BinaryDivOp;

pub trait BinaryOpKernel<T, Op>: Device {
    fn binary_op_fwd(&mut self, lhs: &DAlloc<Self>, rhs: &DAlloc<Self>, op: Op)
                     -> DAlloc<Self>;
}

pub struct UnarySqrOp;
pub struct UnarySqrtOp;
pub struct UnaryExpOp;
pub struct UnaryTanhOp;
pub struct UnaryCosOp;
pub struct UnarySinOp;
pub struct UnaryAbsOp;
pub struct UnaryReluOp;
pub struct UnaryGeluOp;
pub struct UnaryLogOp;
pub struct UnaryFloorOp;
pub struct UnaryCeilOp;

pub trait UnaryOpKernel<T: FloatElem, Op>: Device {
    fn unary_op_fwd(&mut self, x: &DAlloc<Self<>>, op: Op) -> DAlloc<Self>;
}
