use crate::ops::TensorPrimitive;
use crate::tensor::{Float, Tensor};
use rvit_core::element_traits::{DataElem, FloatElem};
use rvit_core::device::{DAlloc, Device};
use rvit_core::tensor_ops::float_ops::TensorFloatOps;
use rvit_core::kernel_ops::tensor::*;

impl<D: Device<Storage = D> + rvit_core::memory::storage::DeviceStorage> TensorFloatOps<D>
    for Float
{
    type Elem = D::FloatElem;

    fn float_add(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinaryAddOp>,
    {
        D::binary_op_fwd(dev, lhs_data, rhs_data, BinaryAddOp);
    }

    fn float_sub(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinarySubOp>,
    {
        D::binary_op_fwd(dev, lhs_data, rhs_data, BinarySubOp);
    }

    fn float_mul(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinaryMulOp>,
    {
        D::binary_op_fwd(dev, lhs_data, rhs_data, BinaryMulOp);
    }

    fn float_div(lhs_data: &DAlloc<D>, rhs_data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: BinaryOpKernel<Self::Elem, BinaryDivOp>,
    {
        D::binary_op_fwd(dev, lhs_data, rhs_data, BinaryDivOp);
    }

    fn float_sqr(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnarySqrOp>,
    {
        D::unary_op_fwd(dev, data, UnarySqrOp)
    }

    fn float_sqrt(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnarySqrtOp>,
    {   
        D::unary_op_fwd(dev, data, UnarySqrtOp)
    }

    fn float_exp(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryExpOp>,
    {
        D::unary_op_fwd(dev, data, UnaryExpOp)
    }

    fn float_tanh(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryTanhOp>,
    {
        D::unary_op_fwd(dev, data, UnaryTanhOp)
    }

    fn float_cos(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryCosOp>,
    {
        D::unary_op_fwd(dev, data, UnaryCosOp)
    }

    fn float_sin(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnarySinOp>,
    {
        D::unary_op_fwd(dev, data, UnarySinOp)
    }

    fn float_abs(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryAbsOp>,
    {
        D::unary_op_fwd(dev, data, UnaryAbsOp)
    }

    fn float_log(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryLogOp>,
    {
        D::unary_op_fwd(dev, data, UnaryLogOp)
    }

    fn float_floor(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryFloorOp>,
    {
        D::unary_op_fwd(dev, data, UnaryFloorOp)
    }

    fn float_ceil(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryCeilOp>,
    {
        D::unary_op_fwd(dev, data, UnaryCeilOp)
    }

    fn float_relu(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryReluOp>,
    {
        D::unary_op_fwd(dev, data, UnaryReluOp)
    }

    fn float_gelu(data: &DAlloc<D>, dev: &mut D) -> DAlloc<D>
    where
        D: UnaryOpKernel<Self::Elem, UnaryGeluOp>,
    {
        D::unary_op_fwd(dev, data, UnaryGeluOp)
    }

    fn float_matmul(lhs: Tensor<Float, D>, rhs: &mut Tensor<Float, D>) -> TensorPrimitive<D>
    where
        D: MatMulKernel<Self::Elem>,
    {
        let (data, out_shape, out_strides) = D::matmul_fwd(
            &mut rhs.device,
            &lhs.data,
            &lhs.shape,
            &lhs.strides,
            &rhs.data,
            &rhs.shape,
            &rhs.strides,
        );
        TensorPrimitive {
            data,
            shape: out_shape,
            strides: out_strides,
        }
    }

    fn float_cast<O: DataElem>(x: &mut Tensor<Float, D>) -> TensorPrimitive<D>
    where
        D: CastKernel<O>,
    {
        TensorPrimitive {
            data: D::cast_fwd(&mut x.device, &x.data),
            shape: x.shape.clone(),
            strides: x.strides.clone(),
        }
    }
}
