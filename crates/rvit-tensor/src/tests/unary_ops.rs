#[rvit_testgen::testgen(unary_ops)]
mod tests {
    use super::*;
    use rvit_core::approx::Tolerance;

    #[test]
    fn test_sqr_unary_op() {
        let dev = TestDevice::default();
        let data = [[1.222, 1.021], [0.432, 0.786]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.sqr();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[1.493284, 1.042441, 0.186624, 0.617796],
            Default::default(),
        );
    }

    #[test]
    fn test_sqrt_unary_op() {
        let dev = TestDevice::default();
        let data = [[8.4, 9.6532], [5.443, 9.888]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.sqrt();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[2.8982754, 3.10696, 2.3330238, 3.1445189],
            Default::default(),
        );
    }

    #[test]
    fn test_exp_unary_op() {
        let dev = TestDevice::default();
        let data = [[2.1, 2.0], [3.0, 1.1]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.exp();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[8.166169, 7.390625, 20.078125, 3.0041661],
            Tolerance {
                ..Default::default()
            }
            .set_relative(0.001),
        );
    }

    #[test]
    fn test_tanh_unary_op() {
        let dev = TestDevice::default();
        let data = [[0.654, -0.341], [-0.112, 0.219]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.tanh();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[0.5743566, -0.32836986, -0.11153404, 0.21556474],
            Default::default(),
        );
    }

    #[test]
    fn test_cos_unary_op() {
        let dev = TestDevice::default();
        let data = [[0.5, -0.3], [0.1, -0.7]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.cos();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[0.87758256189, 0.95533648912, 0.99500416527, 0.76484218728],
            Default::default(),
        );
    }

    #[test]
    fn test_sin_unary_op() {
        let dev = TestDevice::default();
        let data = [[0.5, -0.3], [0.1, -0.7]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.sin();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[0.4794255386, -0.29552020666, 0.09983341664, -0.64421768723],
            Default::default(),
        );
    }

    #[test]
    fn test_abs_unary_op() {
        let dev = TestDevice::default();
        let data = [[-0.41, -0.987], [0.132, -1.154]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.abs();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[0.41, 0.987, 0.132, 1.154],
            Default::default(),
        );
    }

    #[test]
    fn test_log_unary_op() {
        let dev = TestDevice::default();
        let data = [[0.1, 0.9], [1.2, 2.44]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.log();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[-1.0, -0.04575749056, 0.07918124604, 0.38738982633],
            Default::default(),
        );
    }

    #[test]
    fn test_floor_unary_op() {
        let dev = TestDevice::default();
        let data = [[0.89632, 1.6523], [1.0931, 3.768]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.floor();
        approx::assert_approx_eq(&a.into_vec(), &[0.0, 1.0, 1.0, 3.0], Default::default());
    }

    #[test]
    fn test_ceil_unary_op() {
        let dev = TestDevice::default();
        let data = [[0.89632, 1.6523], [1.0931, 3.768]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.ceil();
        approx::assert_approx_eq(&a.into_vec(), &[1.0, 2.0, 2.0, 4.0], Default::default());
    }

    #[test]
    fn test_ceil_relu_op() {
        let dev = TestDevice::default();
        let data = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.relu();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[0.0, 0.0, 0.0, 1.0, 2.0],
            Default::default(),
        );
    }

    #[test]
    fn test_ceil_gelu_op() {
        let dev = TestDevice::default();
        let data = [[-2.0, -1.0, 0.0, 1.0, 2.0]];
        let mut x: Tensor<TestType, _> = Tensor::from_data_convert_shape2(&data, &dev);
        let a = x.gelu();
        approx::assert_approx_eq(
            &a.into_vec(),
            &[-0.04540229, -0.158808, 0.0, 0.841192, 1.9545977],
            Default::default(),
        );
    }
}
