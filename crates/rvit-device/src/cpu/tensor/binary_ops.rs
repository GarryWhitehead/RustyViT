use crate::cpu::device::Cpu;
use crate::op_traits::{BinaryAddOp, BinaryDivOp, BinaryMulOp, BinaryOpKernel, BinarySubOp};
use rvit_core::type_traits::FloatType;

trait BinaryOp<T: FloatType> {
    fn func(&self, lhs: T, rhs: T) -> T;
}

impl<T: FloatType> BinaryOp<T> for BinaryAddOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs + rhs
    }
}

impl<T: FloatType> BinaryOp<T> for BinarySubOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs - rhs
    }
}

impl<T: FloatType> BinaryOp<T> for BinaryMulOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs * rhs
    }
}

impl<T: FloatType> BinaryOp<T> for BinaryDivOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs / rhs
    }
}

impl<T: FloatType, Op: BinaryOp<T>> BinaryOpKernel<T, Op> for Cpu {
    fn forward(&mut self, lhs: &mut Self::Vec, rhs: &Self::Vec, op: Op) -> Self::Vec {
        // As the shape of the lhs and rhs tensors must match, and the data layout is contiguous,
        // it is safe to just iterate through the data buffer.
        for (lhs_v, rhs_v) in lhs.iter_mut().zip(rhs.iter()) {
            *lhs_v = op.func(*lhs_v, *rhs_v);
        }
        lhs.clone()
    }
}
