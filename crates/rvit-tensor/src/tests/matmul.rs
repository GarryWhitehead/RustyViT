#[rvit_testgen::testgen(matmul)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_f32_dim2() {
        let data_a = [
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ];
        let data_b = [[0.4651, 0.9106], [0.3360, 0.5534], [0.8092, 0.3827]];
        let expected = [
            0.62960154, 0.8554974, 1.4642863, 1.5830379, 1.0090116, 0.82806206, 1.0546886, 1.165766,
        ];

        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let mut a = FloatTensor::from_data_shape2(&data_a, &dev.storage);
        let b = FloatTensor::from_data_shape2(&data_b, &dev.storage);
        let c = a.matmul(&b);
        approx::assert_approx_eq(&c.into_vec().as_slice(), &expected, Default::default());
    }

    #[test]
    fn test_matmul_f32_dim3() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());

        let data_a = [
            0.5086, 0.5234, 0.2684, 0.8075, 0.8437, 0.9951, 0.0774, 0.7539, 0.8894, 0.8119, 0.2693,
            0.7249,
        ];
        let data_b = [0.4651, 0.9106, 0.3360, 0.5534, 0.8092, 0.3827];
        let expected = [
            0.62960154, 0.8554974, 1.4642863, 1.5830379, 1.0090116, 0.82806206, 1.0546886, 1.165766,
        ];

        const CHANNEL_COUNT: usize = 6;
        let data_a_batched: Vec<f32> = data_a
            .into_iter()
            .cycle()
            .take(data_a.len() * CHANNEL_COUNT)
            .collect();
        let data_b_batched: Vec<f32> = data_b
            .into_iter()
            .cycle()
            .take(data_b.len() * CHANNEL_COUNT)
            .collect();

        let mut a =
            FloatTensor::from_array_convert(&[CHANNEL_COUNT, 4, 3], &data_a_batched, &dev.storage);
        let b =
            FloatTensor::from_array_convert(&[CHANNEL_COUNT, 3, 2], &data_b_batched, &dev.storage);
        let c = a.matmul(&b);
        let res_data = c.into_vec();
        for i in 0..CHANNEL_COUNT {
            let idx = i * 4 * 2;
            let channel_slice = &res_data[idx..idx + 4 * 2];
            approx::assert_approx_eq(&channel_slice, &expected, Default::default());
        }
    }

    #[test]
    fn test_matmul_f32_dim4() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());

        let data_a = [
            0.5086, 0.5234, 0.2684, 0.8075, 0.8437, 0.9951, 0.0774, 0.7539, 0.8894, 0.8119, 0.2693,
            0.7249,
        ];
        let data_b = [0.4651, 0.9106, 0.3360, 0.5534, 0.8092, 0.3827];
        let expected = [
            0.62960154, 0.8554974, 1.4642863, 1.5830379, 1.0090116, 0.82806206, 1.0546886, 1.165766,
        ];

        const CHANNEL_COUNT: usize = 3;
        const BATCH_COUNT: usize = 6;
        let data_a_batched: Vec<f32> = data_a
            .into_iter()
            .cycle()
            .take(data_a.len() * CHANNEL_COUNT * BATCH_COUNT)
            .collect();
        let data_b_batched: Vec<f32> = data_b
            .into_iter()
            .cycle()
            .take(data_b.len() * CHANNEL_COUNT * BATCH_COUNT)
            .collect();

        let mut a = FloatTensor::from_array_convert(
            &[BATCH_COUNT, CHANNEL_COUNT, 4, 3],
            &data_a_batched,
            &dev.storage,
        );
        let b = FloatTensor::from_array_convert(
            &[BATCH_COUNT, CHANNEL_COUNT, 3, 2],
            &data_b_batched,
            &dev.storage,
        );
        let c = a.matmul(&b);
        let res_data = c.into_vec();
        for b in 0..BATCH_COUNT {
            let base_idx = b * CHANNEL_COUNT * 4 * 2;
            for c in 0..CHANNEL_COUNT {
                let idx = base_idx + c * 4 * 2;
                let channel_slice = &res_data[idx..idx + 4 * 2];
                approx::assert_approx_eq(&channel_slice, &expected, Default::default());
            }
        }
    }
}
