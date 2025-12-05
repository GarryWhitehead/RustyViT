use crate::tensor::Tensor;
use rvit_core::storage::DeviceStorage;
use rvit_core::type_traits::FloatType;
use rvit_device::op_traits::{BinaryAddOp, BinaryDivOp, BinaryMulOp, BinaryOpKernel, BinarySubOp};

impl<T: FloatType, D: BinaryOpKernel<T, BinaryAddOp>> Tensor<T, D> {
    fn add(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D> {
        validate_op(self, rhs);
        let data = D::forward(&mut rhs.device, &mut self.data, &rhs.data, BinaryAddOp);
        Tensor::from_parts(data, &rhs.shape, &rhs.strides, &rhs.device)
    }
}

impl<T: FloatType, D: BinaryOpKernel<T, BinarySubOp>> Tensor<T, D> {
    fn sub(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D> {
        validate_op(self, rhs);
        let data = D::forward(&mut rhs.device, &mut self.data, &rhs.data, BinarySubOp);
        Tensor::from_parts(data, &rhs.shape, &rhs.strides, &rhs.device)
    }
}

impl<T: FloatType, D: BinaryOpKernel<T, BinaryMulOp>> Tensor<T, D> {
    fn mul(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D> {
        validate_op(self, rhs);
        let data = D::forward(&mut rhs.device, &mut self.data, &rhs.data, BinaryMulOp);
        Tensor::from_parts(data, &rhs.shape, &rhs.strides, &rhs.device)
    }
}

impl<T: FloatType, D: BinaryOpKernel<T, BinaryDivOp>> Tensor<T, D> {
    fn div(&mut self, rhs: &mut Tensor<T, D>) -> Tensor<T, D> {
        validate_op(self, rhs);
        let data = D::forward(&mut rhs.device, &mut self.data, &rhs.data, BinaryDivOp);
        Tensor::from_parts(data, &rhs.shape, &rhs.strides, &rhs.device)
    }
}

pub(crate) fn validate_op<T: FloatType, D: DeviceStorage<T>>(
    lhs: &Tensor<T, D>,
    rhs: &Tensor<T, D>,
) {
    if lhs.shape != rhs.shape {
        panic!("Tensor shapes must be the same shape");
    }
    if !lhs.is_contiguous() || !rhs.is_contiguous() {
        panic!("Tensors must have a contiguous memory layout");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_abs_diff_eq;
    use rvit_device::tests::TestDevice;

    macro_rules! test_binary_op {
        ($A_data:expr, $B_data:expr, $Shape:expr, $Exp:expr, $Op: ident) => {
            let dev = TestDevice::default();
            let mut a = Tensor::try_from_data(&$Shape, &$A_data, &dev).unwrap();
            let mut b = Tensor::try_from_data(&$Shape, &$B_data, &dev).unwrap();
            let a = a.$Op(&mut b);
            assert_abs_diff_eq!(&a.try_get_data().unwrap().as_ref(), &$Exp.as_ref());
        };
    }

    #[test]
    fn test_add_binary_op() {
        test_binary_op!(
            [3.457, 2.981, 4.561, 3.999],
            [0.345, 0.198, 1.301, 0.987],
            [2, 2],
            [3.802, 3.179, 5.862, 4.986],
            add
        );
    }

    #[test]
    fn test_sub_binary_op() {
        test_binary_op!(
            [1.662, 1.982, 0.651, 1.009],
            [0.022, 0.010, 0.120, 0.009],
            [2, 2],
            [1.640, 1.972, 0.531, 1.000],
            sub
        );
    }

    #[test]
    fn test_mul_binary_op() {
        test_binary_op!(
            [0.367, 0.210, 0.871, 0.5, 0.409, 0.112],
            [2.0, 1.576, 1.0, 2.0, 0.661, 0.1],
            [2, 3],
            [0.734, 0.33095998, 0.871, 1.0, 0.27034903, 0.0112],
            mul
        );
    }

    #[test]
    fn test_div_binary_op() {
        test_binary_op!(
            [2.0, 4.0, 8.0, 10.0, 6.0, 1.0],
            [2.0, 2.0, 4.0, 2.0, 2.0, 0.5],
            [2, 3],
            [1.0, 2.0, 2.0, 5.0, 3.0, 2.0],
            div
        );
    }
}
