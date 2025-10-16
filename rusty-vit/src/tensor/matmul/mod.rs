mod matmul_cpu;

use crate::device::DeviceStorage;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

pub trait MatMulKernel<T: FloatType>: DeviceStorage<T> {
    fn matmul(&self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>) -> Tensor<T, Self>;
}

#[cfg(test)]
mod tests {
    use crate::device::cpu::Cpu;
    use crate::tensor::Tensor;

    #[test]
    fn test_matmul() {
        let mut dev = Cpu::default();
        let data_a = [
            0.5086, 0.5234, 0.2684, 0.8075, 0.8437, 0.9951, 0.0774, 0.7539, 0.8894, 0.8119, 0.2693,
            0.7249,
        ];
        let data_b = [0.4651, 0.9106, 0.3360, 0.5534, 0.8092, 0.3827];
        let a = Tensor::<f32, _>::try_from_data(&[4, 3], &data_a, &dev).unwrap();
        let b = Tensor::<f32, _>::try_from_data(&[3, 2], &data_b, &dev).unwrap();
        let c = a.matmul(&b, &mut dev);
        assert_eq!(
            c.try_get_data().unwrap(),
            [
                0.62960154, 0.8554974, 1.4642863, 1.5830379, 1.0090116, 0.82806206, 1.0546886,
                1.165766
            ]
        );
    }
}
