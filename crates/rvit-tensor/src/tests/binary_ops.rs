//#[rvit_testgen::testgen(binary_ops)]
mod tests {
    use super::*;
    use crate::FloatTensor;
    use crate::basic_ops::*;
    use crate::tensor::{Float, Tensor};
    use crate::tensor_ops::TensorOps;
    use rvit_device::{DeviceRuntime, Runtime};

    #[test]
    fn test_add_binary_op() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let a_data = [[3.457, 2.981], [4.561, 3.999]];
        let b_data = [[0.345, 0.198], [1.301, 0.987]];
        let mut a = FloatTensor::from_data_shape2(&a_data, &mut dev.storage);
        let mut b = FloatTensor::from_data_shape2(&b_data, &mut dev.storage);
        let a = a.add(&mut b);
        approx::assert_approx_eq(
            &a.into_vec(),
            &[3.802, 3.179, 5.862, 4.986],
            Default::default(),
        );
    }

    #[test]
    fn test_sub_binary_op() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let a_data = [[1.662, 1.982], [0.651, 1.009]];
        let b_data = [[0.022, 0.010], [0.120, 0.009]];
        let mut a = FloatTensor::from_data_shape2(&a_data, &dev.storage);
        let mut b = FloatTensor::from_data_shape2(&b_data, &dev.storage);
        let a = a.sub(&mut b);
        approx::assert_approx_eq(
            &a.into_vec(),
            &[1.640, 1.972, 0.531, 1.000],
            Default::default(),
        );
    }

    #[test]
    fn test_mul_binary_op() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let a_data = [[0.367, 0.210, 0.871], [0.5, 0.409, 0.112]];
        let b_data = [[2.0, 1.576, 1.0], [2.0, 0.661, 0.1]];
        let mut a = FloatTensor::from_data_shape2(&a_data, &dev.storage);
        let mut b = FloatTensor::from_data_shape2(&b_data, &dev.storage);
        let a = a.mul(&mut b);
        approx::assert_approx_eq(
            &a.into_vec(),
            &[0.734, 0.33095998, 0.871, 1.0, 0.27034903, 0.0112],
            Default::default(),
        );
    }

    #[test]
    fn test_div_binary_op() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let a_data = [[2.0, 4.0, 8.0], [10.0, 6.0, 1.0]];
        let b_data = [[2.0, 2.0, 4.0], [2.0, 2.0, 0.5]];
        let mut a = FloatTensor::from_data_shape2(&a_data, &dev);
        let mut b = FloatTensor::from_data_shape2(&b_data, &dev);
        let a = a.div(&mut b);
        approx::assert_approx_eq(
            &a.into_vec(),
            &[1.0, 2.0, 2.0, 5.0, 3.0, 2.0],
            Default::default(),
        );
    }
}
