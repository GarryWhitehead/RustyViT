use crate::cpu::device::Cpu;
use crate::tensor::op_traits::*;
use crate::{DAlloc, Runtime};
use rvit_core::element_traits::{FloatElem, IntElem};

pub trait UnaryOp<T: FloatElem> {
    fn func(&self, x: T) -> T;
}

impl<T: FloatElem> UnaryOp<T> for UnarySqrOp {
    fn func(&self, x: T) -> T {
        x * x
    }
}

impl<T: FloatElem> UnaryOp<T> for UnarySqrtOp {
    fn func(&self, x: T) -> T {
        x.sqrt()
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryExpOp {
    fn func(&self, x: T) -> T {
        x.exp()
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryTanhOp {
    fn func(&self, x: T) -> T {
        x.tanh()
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryCosOp {
    fn func(&self, x: T) -> T {
        x.cos()
    }
}

impl<T: FloatElem> UnaryOp<T> for UnarySinOp {
    fn func(&self, x: T) -> T {
        x.sin()
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryAbsOp {
    fn func(&self, x: T) -> T {
        x.abs()
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryReluOp {
    fn func(&self, x: T) -> T {
        T::max(x, T::zero())
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryGeluOp {
    fn func(&self, x: T) -> T {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        let fx = x.to_f32().unwrap();
        let v =
            0.5f32 * fx * (1.0 + f32::tanh(SQRT_2_OVER_PI * (fx + 0.044715 * f32::powf(fx, 3.0))));
        T::from(v).unwrap_or(T::zero())
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryLogOp {
    fn func(&self, x: T) -> T {
        x.log(T::from(10).unwrap())
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryFloorOp {
    fn func(&self, x: T) -> T {
        x.floor()
    }
}

impl<T: FloatElem> UnaryOp<T> for UnaryCeilOp {
    fn func(&self, x: T) -> T {
        x.ceil()
    }
}

impl<F: FloatElem, Op: UnaryOp<F>> UnaryOpKernel<F, Op> for Runtime {
    fn unary_op_fwd(&mut self, lhs: &mut DAlloc<Self>, op: Op) -> DAlloc<Self> {
        // As the shape of the lhs and rhs tensors must match, and the data layout is contiguous,
        // it is safe to just iterate through the data buffer.
        for lhs_v in lhs.iter_mut() {
            *lhs_v = op.func(*lhs_v);
        }
        lhs.clone()
    }
}
