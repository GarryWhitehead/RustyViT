use crate::device::cpu::Cpu;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

pub trait UnaryOp<T: FloatType> {
    fn func(&self, x: T) -> T;
}

impl<T: FloatType> UnaryOp<T> for super::UnarySqrOp {
    fn func(&self, x: T) -> T {
        x * x
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnarySqrtOp {
    fn func(&self, x: T) -> T {
        x.sqrt()
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryExpOp {
    fn func(&self, x: T) -> T {
        x.exp()
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryTanhOp {
    fn func(&self, x: T) -> T {
        x.tanh()
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryCosOp {
    fn func(&self, x: T) -> T {
        x.cos()
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnarySinOp {
    fn func(&self, x: T) -> T {
        x.sin()
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryAbsOp {
    fn func(&self, x: T) -> T {
        x.abs()
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryReluOp {
    fn func(&self, x: T) -> T {
        T::max(x, T::zero())
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryGeluOp {
    fn func(&self, x: T) -> T {
        const SQRT_2_OVER_PI: f32 = 0.797_884_6;
        let fx = x.to_f32().unwrap();
        let v =
            0.5f32 * fx * (1.0 + f32::tanh(SQRT_2_OVER_PI * (fx + 0.044715 * f32::powf(fx, 3.0))));
        T::from(v).unwrap_or(T::zero())
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryLogOp {
    fn func(&self, x: T) -> T {
        x.log(T::from(10).unwrap())
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryFloorOp {
    fn func(&self, x: T) -> T {
        x.floor()
    }
}

impl<T: FloatType> UnaryOp<T> for super::UnaryCeilOp {
    fn func(&self, x: T) -> T {
        x.ceil()
    }
}

impl<T: FloatType, Op: UnaryOp<T>> super::UnaryOpExecutor<T, Op> for Cpu {
    fn forward(&mut self, lhs: &Tensor<T, Self>, op: Op) -> Tensor<T, Self> {
        let mut t_lhs = lhs.clone();
        // As the shape of the lhs and rhs tensors must match, and the data layout is contiguous,
        // it is safe to just iterate through the data buffer.
        for lhs_v in t_lhs.data.iter_mut() {
            *lhs_v = op.func(*lhs_v);
        }
        t_lhs
    }
}
