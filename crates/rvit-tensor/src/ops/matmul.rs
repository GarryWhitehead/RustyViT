use crate::tensor::Tensor;
use rvit_core::storage::DeviceStorage;
use rvit_core::type_traits::FloatType;
use rvit_device::op_traits::MatMulKernel;

impl<T: FloatType, D: MatMulKernel<T>> Tensor<T, D> {
    fn matmul(&mut self, rhs: &Tensor<T, D>) -> Tensor<T, D> {
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            panic!("Tensor must have at least two dimensions (for now).");
        }
        if self.shape.len() != rhs.shape.len() {
            panic!("Tensors must have the same dimensions.");
        }
        if self.shape.len() > 2 && (self.shape[0] != rhs.shape[0]) {
            panic!("Tensors must have the same batch size");
        }
        let mut dev = self.clone();
        let (data, out_shape, out_strides) = D::forward(
            &rhs.device,
            &mut self.data,
            &self.shape,
            &self.strides,
            &rhs.data,
            &rhs.shape,
            &rhs.strides,
        );
        Tensor::from_parts(data, &out_shape, &out_strides, &rhs.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approxim::assert_abs_diff_eq;
    use half::f16;
    use rvit_device::tests::TestDevice;

    const DATA_A: [f32; 12] = [
        0.5086, 0.5234, 0.2684, 0.8075, 0.8437, 0.9951, 0.0774, 0.7539, 0.8894, 0.8119, 0.2693,
        0.7249,
    ];
    const DATA_B: [f32; 6] = [0.4651, 0.9106, 0.3360, 0.5534, 0.8092, 0.3827];
    const EXPECTED: [f32; 8] = [
        0.62960154, 0.8554974, 1.4642863, 1.5830379, 1.0090116, 0.82806206, 1.0546886, 1.165766,
    ];

    #[test]
    fn test_matmul_f32_dim2() {
        let mut dev = TestDevice::default();
        let mut a = Tensor::<f32, _>::try_from_data(&[4, 3], &DATA_A, &dev).unwrap();
        let b = Tensor::<f32, _>::try_from_data(&[3, 2], &DATA_B, &dev).unwrap();
        let c = a.matmul(&b);
        assert_abs_diff_eq!(&c.try_get_data().unwrap().as_slice(), &EXPECTED.as_ref());
    }

    #[test]
    fn test_matmul_f32_dim3() {
        let mut dev = TestDevice::default();
        const CHANNEL_COUNT: usize = 6;
        let data_a_batched: Vec<f32> = DATA_A
            .into_iter()
            .cycle()
            .take(DATA_A.len() * CHANNEL_COUNT)
            .collect();
        let data_b_batched: Vec<f32> = DATA_B
            .into_iter()
            .cycle()
            .take(DATA_B.len() * CHANNEL_COUNT)
            .collect();

        let mut a =
            Tensor::<f32, _>::try_from_data(&[CHANNEL_COUNT, 4, 3], &data_a_batched, &dev).unwrap();
        let b =
            Tensor::<f32, _>::try_from_data(&[CHANNEL_COUNT, 3, 2], &data_b_batched, &dev).unwrap();
        let c = a.matmul(&b);
        let res_data = c.try_get_data().unwrap();
        for i in 0..CHANNEL_COUNT {
            let idx = i * 4 * 2;
            let channel_slice = &res_data[idx..idx + 4 * 2];
            assert_abs_diff_eq!(&channel_slice, &EXPECTED.as_ref(),);
        }
    }

    #[test]
    fn test_matmul_f32_dim4() {
        let mut dev = TestDevice::default();
        const CHANNEL_COUNT: usize = 3;
        const BATCH_COUNT: usize = 6;
        let data_a_batched: Vec<f32> = DATA_A
            .into_iter()
            .cycle()
            .take(DATA_A.len() * CHANNEL_COUNT * BATCH_COUNT)
            .collect();
        let data_b_batched: Vec<f32> = DATA_B
            .into_iter()
            .cycle()
            .take(DATA_B.len() * CHANNEL_COUNT * BATCH_COUNT)
            .collect();

        let mut a = Tensor::<f32, _>::try_from_data(
            &[BATCH_COUNT, CHANNEL_COUNT, 4, 3],
            &data_a_batched,
            &dev,
        )
        .unwrap();
        let b = Tensor::<f32, _>::try_from_data(
            &[BATCH_COUNT, CHANNEL_COUNT, 3, 2],
            &data_b_batched,
            &dev,
        )
        .unwrap();
        let c = a.matmul(&b);
        let res_data = c.try_get_data().unwrap();
        for b in 0..BATCH_COUNT {
            let base_idx = b * CHANNEL_COUNT * 4 * 2;
            for c in 0..CHANNEL_COUNT {
                let idx = base_idx + c * 4 * 2;
                let channel_slice = &res_data[idx..idx + 4 * 2];
                assert_abs_diff_eq!(&channel_slice, &EXPECTED.as_ref());
            }
        }
    }

    #[test]
    fn test_matmul_f16_dim2() {
        let mut dev = TestDevice::default();
        let data_a_f16 = DATA_A.map(|x| f16::from_f32(x));
        let data_b_f16 = DATA_B.map(|x| f16::from_f32(x));
        let mut a = Tensor::<f16, _>::try_from_data(&[4, 3], &data_a_f16, &dev).unwrap();
        let b = Tensor::<f16, _>::try_from_data(&[3, 2], &data_b_f16, &dev).unwrap();
        let c = a.matmul(&b);
        assert_eq!(
            c.try_get_data().unwrap(),
            EXPECTED.map(|x| f16::from_f32(x))
        );
    }

    #[test]
    fn test_matmul_f16_dim3() {
        let mut dev = TestDevice::default();
        const CHANNEL_COUNT: usize = 6;
        let data_a_f16 = DATA_A.map(|x| f16::from_f32(x));
        let data_b_f16 = DATA_B.map(|x| f16::from_f32(x));
        let data_a_batched: Vec<f16> = data_a_f16
            .into_iter()
            .cycle()
            .take(DATA_A.len() * CHANNEL_COUNT)
            .collect();
        let data_b_batched: Vec<f16> = data_b_f16
            .into_iter()
            .cycle()
            .take(DATA_B.len() * CHANNEL_COUNT)
            .collect();

        let mut a =
            Tensor::<f16, _>::try_from_data(&[CHANNEL_COUNT, 4, 3], &data_a_batched, &dev).unwrap();
        let b =
            Tensor::<f16, _>::try_from_data(&[CHANNEL_COUNT, 3, 2], &data_b_batched, &dev).unwrap();
        let c = a.matmul(&b);
        let res_data = c.try_get_data().unwrap();
        for i in 0..CHANNEL_COUNT {
            let idx = i * 4 * 2;
            let channel_slice = &res_data[idx..idx + 4 * 2];
            assert_eq!(channel_slice, EXPECTED.map(|x| f16::from_f32(x)));
        }
    }
}
