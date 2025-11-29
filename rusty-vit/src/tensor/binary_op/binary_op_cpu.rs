use crate::device::cpu::Cpu;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

trait BinaryOp<T: FloatType> {
    fn func(&self, lhs: T, rhs: T) -> T;
}

impl<T: FloatType> BinaryOp<T> for super::BinaryAddOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs + rhs
    }
}

impl<T: FloatType> BinaryOp<T> for super::BinarySubOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs - rhs
    }
}

impl<T: FloatType> BinaryOp<T> for super::BinaryMulOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs * rhs
    }
}

impl<T: FloatType> BinaryOp<T> for super::BinaryDivOp {
    fn func(&self, lhs: T, rhs: T) -> T {
        lhs / rhs
    }
}

impl<T: FloatType, Op: BinaryOp<T>> super::BinaryOpExecutor<T, Op> for Cpu {
    fn forward(&mut self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>, op: Op) -> Tensor<T, Self> {
        let mut t_lhs = lhs.clone();
        // As the shape of the lhs and rhs tensors must match, and the data layout is contiguous,
        // it is safe to just iterate through the data buffer.
        for (lhs_v, rhs_v) in t_lhs.data.iter_mut().zip(rhs.data.iter()) {
            *lhs_v = op.func(*lhs_v, *rhs_v);
        }
        t_lhs
    }
}
