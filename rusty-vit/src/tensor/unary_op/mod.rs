use crate::device::DeviceStorage;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

mod unary_cpu;
#[cfg(feature = "cuda")]
mod unary_cu;

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

pub trait UnaryOpExecutor<T: FloatType, Op>: DeviceStorage<T> {
    fn forward(&mut self, x: &Tensor<T, Self>, op: Op) -> Tensor<T, Self>;
}

pub trait UnarySqrOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnarySqrOp>> {
    fn sqr(&mut self) -> Tensor<T, S>;
}

pub trait UnarySqrtOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnarySqrtOp>> {
    fn sqrt(&mut self) -> Tensor<T, S>;
}

pub trait UnaryExpOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryExpOp>> {
    fn exp(&mut self) -> Tensor<T, S>;
}

pub trait UnaryTanhOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryTanhOp>> {
    fn tanh(&mut self) -> Tensor<T, S>;
}

pub trait UnaryCosOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryCosOp>> {
    fn cos(&mut self) -> Tensor<T, S>;
}

pub trait UnarySinOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnarySinOp>> {
    fn sin(&mut self) -> Tensor<T, S>;
}

pub trait UnaryAbsOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryAbsOp>> {
    fn abs(&mut self) -> Tensor<T, S>;
}

pub trait UnaryReluOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryReluOp>> {
    fn relu(&mut self) -> Tensor<T, S>;
}

pub trait UnaryGeluOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryGeluOp>> {
    fn gelu(&mut self) -> Tensor<T, S>;
}

pub trait UnaryLogOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryLogOp>> {
    fn log(&mut self) -> Tensor<T, S>;
}

pub trait UnaryFloorOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryFloorOp>> {
    fn floor(&mut self) -> Tensor<T, S>;
}

pub trait UnaryCeilOpKernel<T: FloatType, S: UnaryOpExecutor<T, UnaryCeilOp>> {
    fn ceil(&mut self) -> Tensor<T, S>;
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnarySqrOp>> UnarySqrOpKernel<T, E> for Tensor<T, E> {
    fn sqr(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnarySqrOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnarySqrtOp>> UnarySqrtOpKernel<T, E> for Tensor<T, E> {
    fn sqrt(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnarySqrtOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryExpOp>> UnaryExpOpKernel<T, E> for Tensor<T, E> {
    fn exp(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryExpOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryTanhOp>> UnaryTanhOpKernel<T, E> for Tensor<T, E> {
    fn tanh(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryTanhOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryCosOp>> UnaryCosOpKernel<T, E> for Tensor<T, E> {
    fn cos(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryCosOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnarySinOp>> UnarySinOpKernel<T, E> for Tensor<T, E> {
    fn sin(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnarySinOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryAbsOp>> UnaryAbsOpKernel<T, E> for Tensor<T, E> {
    fn abs(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryAbsOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryGeluOp>> UnaryGeluOpKernel<T, E> for Tensor<T, E> {
    fn gelu(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryGeluOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryReluOp>> UnaryReluOpKernel<T, E> for Tensor<T, E> {
    fn relu(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryReluOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryLogOp>> UnaryLogOpKernel<T, E> for Tensor<T, E> {
    fn log(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryLogOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryFloorOp>> UnaryFloorOpKernel<T, E> for Tensor<T, E> {
    fn floor(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryFloorOp)
    }
}

impl<T: FloatType, E: UnaryOpExecutor<T, UnaryCeilOp>> UnaryCeilOpKernel<T, E> for Tensor<T, E> {
    fn ceil(&mut self) -> Tensor<T, E> {
        execute_unary_op(self, UnaryCeilOp)
    }
}

pub(crate) fn execute_unary_op<T: FloatType, Op, D: UnaryOpExecutor<T, Op>>(
    x: &Tensor<T, D>,
    op: Op,
) -> Tensor<T, D> {
    let mut dev = x.device.clone();
    dev.forward(x, op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestDevice;
    use approxim::assert_abs_diff_eq;

    macro_rules! test_unary_op {
        ($X_data:expr, $Shape:expr, $Exp:expr, $Op: ident) => {
            let dev = TestDevice::default();
            let mut x = Tensor::try_from_data(&$Shape, &$X_data, &dev).unwrap();
            let x = x.$Op();
            assert_abs_diff_eq!(&x.try_get_data().unwrap().as_ref(), &$Exp.as_ref());
        };
    }

    #[test]
    fn test_sqr_unary_op() {
        test_unary_op!(
            [1.222, 1.021, 0.432, 0.786],
            [2, 2],
            [1.493284, 1.042441, 0.186624, 0.617796],
            sqr
        );
    }

    #[test]
    fn test_sqrt_unary_op() {
        test_unary_op!(
            [8.4, 9.6532, 5.443, 9.888],
            [2, 2],
            [2.8982754, 3.10696, 2.3330238, 3.1445189],
            sqrt
        );
    }

    #[test]
    fn test_exp_unary_op() {
        test_unary_op!(
            [2.1, 2.8, 3.1, 1.1],
            [2, 2],
            [8.166169, 16.444647, 22.197948, 3.0041661],
            exp
        );
    }

    #[test]
    fn test_tanh_unary_op() {
        test_unary_op!(
            [0.654, -0.341, -0.112, 0.219],
            [2, 2],
            [0.5743566, -0.32836986, -0.11153404, 0.21556474],
            tanh
        );
    }

    #[test]
    fn test_cos_unary_op() {
        test_unary_op!(
            [0.5, -0.3, 0.1, -0.7],
            [2, 2],
            [0.87758256189, 0.95533648912, 0.99500416527, 0.76484218728],
            cos
        );
    }

    #[test]
    fn test_sin_unary_op() {
        test_unary_op!(
            [0.5, -0.3, 0.1, -0.7],
            [2, 2],
            [0.4794255386, -0.29552020666, 0.09983341664, -0.64421768723],
            sin
        );
    }

    #[test]
    fn test_abs_unary_op() {
        test_unary_op!(
            [-0.41, -0.987, 0.132, -1.154],
            [2, 2],
            [0.41, 0.987, 0.132, 1.154],
            abs
        );
    }

    #[test]
    fn test_log_unary_op() {
        test_unary_op!(
            [0.1, 0.9, 1.2, 2.44],
            [2, 2],
            [-1.0, -0.04575749056, 0.07918124604, 0.38738982633],
            log
        );
    }

    #[test]
    fn test_floor_unary_op() {
        test_unary_op!(
            [0.89632, 1.6523, 1.0931, 3.768],
            [2, 2],
            [0.0, 1.0, 1.0, 3.0],
            floor
        );
    }

    #[test]
    fn test_ceil_unary_op() {
        test_unary_op!(
            [0.89632, 1.6523, 1.0931, 3.768],
            [2, 2],
            [1.0, 2.0, 2.0, 4.0],
            ceil
        );
    }

    #[test]
    fn test_relu_unary_op() {
        test_unary_op!(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [1, 5],
            [0.0, 0.0, 0.0, 1.0, 2.0],
            relu
        );
    }

    #[test]
    fn test_gelu_unary_op() {
        test_unary_op!(
            [-2.0, -1.0, 0.0, 1.0, 2.0],
            [1, 5],
            [-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977],
            gelu
        );
    }
}
