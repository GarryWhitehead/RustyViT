use crate::device::{DAlloc, Device};
use crate::element_traits::{DataElem, FloatElem};
use crate::kernel_ops::tensor::*;

pub trait TensorFloatOps<D: Device> {
    type Elem: FloatElem;

    fn float_add(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinaryAddOp>;

    fn float_sub(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinarySubOp>;

    fn float_mul(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinaryMulOp>;

    fn float_div(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinaryDivOp>;

    fn float_sqr(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnarySqrOp>;

    fn float_sqrt(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnarySqrtOp>;

    fn float_exp(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryExpOp>;

    fn float_tanh(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryTanhOp>;

    fn float_cos(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryCosOp>;

    fn float_sin(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnarySinOp>;

    fn float_abs(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryAbsOp>;

    fn float_log(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryLogOp>;

    fn float_floor(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryFloorOp>;

    fn float_ceil(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryCeilOp>;

    fn float_relu(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryReluOp>;

    fn float_gelu(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryGeluOp>;

    fn float_matmul(lhs: Tensor<Float, D>, rhs: &mut Tensor<Float, D>) -> TensorPrimitive<D>
    where
        D: MatMulKernel<Self::Elem>;

    fn float_cast<O: DataElem>(x: &mut Tensor<Float, D>) -> TensorPrimitive<D>
    where
        D: CastKernel<O>;
}
