use crate::device::DeviceStorage;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

mod binary_op_cpu;
#[cfg(feature = "cuda")]
mod binary_op_cu;

pub struct BinaryAddOp;
pub struct BinarySubOp;
pub struct BinaryMulOp;
pub struct BinaryDivOp;

pub trait BinaryOpExecutor<T: FloatType, Op>: DeviceStorage<T> {
    fn forward(&mut self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>, op: Op) -> Tensor<T, Self>;
}

pub trait BinaryAddOpKernel<T: FloatType, S: BinaryOpExecutor<T, BinaryAddOp>> {
    fn add(&mut self, other: &Tensor<T, S>) -> Tensor<T, S>;
}

pub trait BinarySubOpKernel<T: FloatType, S: BinaryOpExecutor<T, BinarySubOp>> {
    fn sub(&mut self, other: &Tensor<T, S>) -> Tensor<T, S>;
}

pub trait BinaryMulOpKernel<T: FloatType, S: BinaryOpExecutor<T, BinaryMulOp>> {
    fn mul(&mut self, other: &Tensor<T, S>) -> Tensor<T, S>;
}

pub trait BinaryDivOpKernel<T: FloatType, S: BinaryOpExecutor<T, BinaryDivOp>> {
    fn div(&mut self, other: &Tensor<T, S>) -> Tensor<T, S>;
}

impl<T: FloatType, E: BinaryOpExecutor<T, BinaryAddOp>> BinaryAddOpKernel<T, E> for Tensor<T, E> {
    fn add(&mut self, rhs: &Tensor<T, E>) -> Tensor<T, E> {
        execute_binary_op(self, rhs, BinaryAddOp)
    }
}

impl<T: FloatType, E: BinaryOpExecutor<T, BinarySubOp>> BinarySubOpKernel<T, E> for Tensor<T, E> {
    fn sub(&mut self, rhs: &Tensor<T, E>) -> Tensor<T, E> {
        execute_binary_op(self, rhs, BinarySubOp)
    }
}

impl<T: FloatType, E: BinaryOpExecutor<T, BinaryDivOp>> BinaryDivOpKernel<T, E> for Tensor<T, E> {
    fn div(&mut self, rhs: &Tensor<T, E>) -> Tensor<T, E> {
        execute_binary_op(self, rhs, BinaryDivOp)
    }
}

impl<T: FloatType, E: BinaryOpExecutor<T, BinaryMulOp>> BinaryMulOpKernel<T, E> for Tensor<T, E> {
    fn mul(&mut self, rhs: &Tensor<T, E>) -> Tensor<T, E> {
        execute_binary_op(self, rhs, BinaryMulOp)
    }
}

pub(crate) fn execute_binary_op<T: FloatType, Op, D: BinaryOpExecutor<T, Op>>(
    lhs: &Tensor<T, D>,
    rhs: &Tensor<T, D>,
    op: Op,
) -> Tensor<T, D> {
    if lhs.shape != rhs.shape {
        panic!("Tensor shapes must be the same shape");
    }
    if !lhs.is_contiguous() || !rhs.is_contiguous() {
        panic!("Tensors must have a contiguous memory layout");
    }
    let t_lhs = lhs.clone();
    let mut t_rhs = rhs.clone();
    t_rhs.device.forward(&t_lhs, rhs, op)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestDevice;
    use approxim::assert_abs_diff_eq;

    #[test]
    fn test_add_binary_op() {
        let dev = TestDevice::default();
        let data_a = [3.457, 2.981, 4.561, 3.999];
        let data_b = [0.345, 0.198, 1.301, 0.987];
        let mut a = Tensor::try_from_data(&[2, 2], &data_a, &dev).unwrap();
        let mut b = Tensor::try_from_data(&[2, 2], &data_b, &dev).unwrap();
        let a = a.add(&mut b);
        assert_abs_diff_eq!(
            &a.try_get_data().unwrap().as_ref(),
            &[3.802, 3.179, 5.862, 4.986].as_ref()
        );
    }

    #[test]
    fn test_sub_binary_op() {
        let dev = TestDevice::default();
        let data_a = [1.662, 1.982, 0.651, 1.009];
        let data_b = [0.022, 0.010, 0.120, 0.009];
        let mut a = Tensor::try_from_data(&[2, 2], &data_a, &dev).unwrap();
        let mut b = Tensor::try_from_data(&[2, 2], &data_b, &dev).unwrap();
        let a = a.sub(&mut b);
        assert_abs_diff_eq!(
            &a.try_get_data().unwrap().as_ref(),
            &[1.640, 1.972, 0.531, 1.000].as_ref()
        );
    }

    #[test]
    fn test_mul_binary_op() {
        let dev = TestDevice::default();
        let data_a = [0.367, 0.210, 0.871, 0.5, 0.409, 0.112];
        let data_b = [2.0, 1.576, 1.0, 2.0, 0.661, 0.1];
        let mut a = Tensor::try_from_data(&[2, 3], &data_a, &dev).unwrap();
        let mut b = Tensor::try_from_data(&[2, 3], &data_b, &dev).unwrap();
        let a = a.mul(&mut b);
        assert_abs_diff_eq!(
            &a.try_get_data().unwrap().as_ref(),
            &[0.734, 0.33095998, 0.871, 1.0, 0.27034903, 0.0112].as_ref()
        );
    }

    #[test]
    fn test_div_binary_op() {
        let dev = TestDevice::default();
        let data_a = [2.0, 4.0, 8.0, 10.0, 6.0, 1.0];
        let data_b = [2.0, 2.0, 4.0, 2.0, 2.0, 0.5];
        let mut a = Tensor::try_from_data(&[2, 3], &data_a, &dev).unwrap();
        let mut b = Tensor::try_from_data(&[2, 3], &data_b, &dev).unwrap();
        let a = a.div(&mut b);
        assert_abs_diff_eq!(
            &a.try_get_data().unwrap().as_ref(),
            &[1.0, 2.0, 2.0, 5.0, 3.0, 2.0].as_ref()
        );
    }
}
