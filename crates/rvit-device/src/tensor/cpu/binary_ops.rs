use crate::tensor::op_traits::{
    BinaryAddOp, BinaryDivOp, BinaryMulOp, BinaryOpKernel, BinarySubOp,
};
use crate::{DAlloc, Runtime};
use rvit_core::element_traits::DataElem;

trait BinaryOp<T: DataElem> {
    fn func(&self, lhs: T, rhs: T) -> T;
}

impl<T: DataElem> BinaryOp<T> for BinaryAddOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs + rhs
    }
}

impl<T: DataElem> BinaryOp<T> for BinarySubOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs - rhs
    }
}

impl<T: DataElem> BinaryOp<T> for BinaryMulOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs * rhs
    }
}

impl<T: DataElem> BinaryOp<T> for BinaryDivOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs / rhs
    }
}

impl<E: DataElem, Op: BinaryOp<E>> BinaryOpKernel<E, Op> for Runtime {
    fn binary_op_fwd(
        &mut self,
        lhs: &mut DAlloc<Self>,
        rhs: &DAlloc<Self>,
        op: Op,
    ) -> DAlloc<Self> {
        // As the shape of the lhs and rhs tensors must match, and the data layout is contiguous,
        // it is safe to just iterate through the data buffer.
        for (lhs_v, rhs_v) in lhs.iter_mut().zip(rhs.iter::<E>()) {
            *lhs_v = op.func(*lhs_v, rhs_v);
        }
        lhs.clone()
    }
}
