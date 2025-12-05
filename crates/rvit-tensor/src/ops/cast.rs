use crate::tensor::Tensor;
use rvit_core::storage::DeviceStorage;
use rvit_core::type_traits::FloatType;
use rvit_device::op_traits::CastKernel;

impl<T: FloatType, D: DeviceStorage<T>> Tensor<T, D> {
    fn cast<O: FloatType>(&self) -> Tensor<O, D>
    where
        D: CastKernel<T, O>,
    {
        let data = D::forward(&self.device, &self.data, &self.shape);
        Tensor::from_parts(data, &self.shape, &self.strides, &self.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use rvit_device::tests::TestDevice;

    #[test]
    fn test_cast_f16_to_f32() {
        let mut dev = TestDevice::default();
        let data_a: Vec<f16> = (0u8..100u8).map(|x| f16::from(x)).collect();
        let a = Tensor::<f16, _>::try_from_data(&[10, 10], &data_a, &dev)
            .unwrap()
            .cast::<f32>();
        assert_eq!(
            a.try_get_data().unwrap(),
            data_a.iter().map(|x| f32::from(*x)).collect::<Vec<_>>(),
        );
    }
}
