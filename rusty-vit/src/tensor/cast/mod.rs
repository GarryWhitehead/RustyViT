use crate::device::DeviceStorage;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

mod cast_cpu;
#[cfg(feature = "cuda")]
mod cast_cu;
#[cfg(feature = "vulkan")]
mod cast_vk;

pub trait CastKernel<T: FloatType, O: FloatType>: DeviceStorage<T> + DeviceStorage<O> {
    fn cast(&mut self, lhs: &Tensor<T, Self>) -> Tensor<O, Self>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::TestDevice;
    use half::f16;

    #[test]
    fn test_cast_f16_to_f32() {
        let mut dev = TestDevice::default();
        let data_a: Vec<f16> = (0u8..100u8).map(|x| f16::from(x)).collect();
        let a = Tensor::<f16, _>::try_from_data(&[10, 10], &data_a, &dev)
            .unwrap()
            .cast::<f32>(&mut dev);
        assert_eq!(
            a.try_get_data().unwrap(),
            data_a.iter().map(|x| f32::from(*x)).collect::<Vec<_>>(),
        );
    }
}
