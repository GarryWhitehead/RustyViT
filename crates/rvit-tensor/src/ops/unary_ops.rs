use crate::tensor::Tensor;
use rvit_core::storage::DeviceStorage;
use rvit_core::type_traits::FloatType;
use rvit_device::op_traits::*;

impl<T: FloatType, D: UnaryOpKernel<T, UnarySqrOp>> Tensor<T, D> {
    fn sqr(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnarySqrOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnarySqrtOp>> Tensor<T, D> {
    fn sqrt(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnarySqrtOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryExpOp>> Tensor<T, D> {
    fn exp(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryExpOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryTanhOp>> Tensor<T, D> {
    fn tanh(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryTanhOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryCosOp>> Tensor<T, D> {
    fn cos(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryCosOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnarySinOp>> Tensor<T, D> {
    fn sin(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnarySinOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryReluOp>> Tensor<T, D> {
    fn relu(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryReluOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryGeluOp>> Tensor<T, D> {
    fn gelu(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryGeluOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryAbsOp>> Tensor<T, D> {
    fn abs(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryAbsOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryLogOp>> Tensor<T, D> {
    fn log(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryLogOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryFloorOp>> Tensor<T, D> {
    fn floor(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryFloorOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

impl<T: FloatType, D: UnaryOpKernel<T, UnaryCeilOp>> Tensor<T, D> {
    fn ceil(&mut self) -> Tensor<T, D> {
        let data = D::forward(&mut self.device, &mut self.data, UnaryCeilOp);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_abs_diff_eq;
    use rvit_device::tests::TestDevice;

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
