use crate::cpu::device::VecPool;
use rvit_core::storage::DeviceStorage;
use rvit_core::type_traits::FloatType;

pub trait MatMulKernel<T: FloatType>: DeviceStorage<T> {
    fn forward(
        &self,
        lhs: &Self::Vec,
        lhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs: &Self::Vec,
        rhs_shape: &[usize],
        rhs_strides: &[usize],
    ) -> (Self::Vec, Vec<usize>, Vec<usize>);
}

pub trait CastKernel<T: FloatType, O: FloatType>: DeviceStorage<T> + DeviceStorage<O> {
    fn forward(
        &self,
        x: &<Self as DeviceStorage<T>>::Vec,
        shape: &[usize],
    ) -> <Self as DeviceStorage<O>>::Vec;
}

pub trait Conv2dKernel<T: FloatType>: DeviceStorage<T> {
    fn forward(
        &self,
        x: &Self::Vec,
        shape: &[usize],
        strides: &[usize],
        filters: &Self::Vec,
        f_shape: &[usize],
        out_shape: &[usize],
        stride: usize,
        padding: usize,
        groups: usize,
        to_nhwc: bool,
    ) -> Self::Vec;
}

pub trait ConvConvertKernel<T: FloatType>: DeviceStorage<T> {
    fn im2col(
        &self,
        x: &Self::Vec,
        x_shape: &[usize],
        f_shape: &[usize],
        batch_size: usize,
        out_width: usize,
        out_height: usize,
        stride: usize,
        padding: usize,
        is_nhwc: bool,
    ) -> (Self::Vec, Vec<usize>, Vec<usize>);
    fn nhwc_to_nchw(&mut self, x: &Self::Vec, shape: &[usize]) -> Self::Vec;
    fn nchw_to_nhwc(&mut self, x: &Self::Vec, shape: &[usize]) -> Self::Vec;
}

pub struct BinaryAddOp;
pub struct BinarySubOp;
pub struct BinaryMulOp;
pub struct BinaryDivOp;

pub trait BinaryOpKernel<T: FloatType, Op>: DeviceStorage<T> {
    fn forward(&mut self, lhs: &mut Self::Vec, rhs: &Self::Vec, op: Op) -> Self::Vec;
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

pub trait UnaryOpKernel<T: FloatType, Op>: DeviceStorage<T> {
    fn forward(&mut self, x: &mut Self::Vec, op: Op) -> Self::Vec;
}
