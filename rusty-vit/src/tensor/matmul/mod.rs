mod matmul_cpu;
#[cfg(feature = "cuda")]
mod matmul_cu;
#[cfg(feature = "vulkan")]
mod matmul_vk;

use crate::device::DeviceStorage;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

pub trait MatMulKernel<T: FloatType>: DeviceStorage<T> {
    fn matmul(&mut self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>) -> Tensor<T, Self>;
}

pub fn inner_shape(lhs_shape: &[usize], rhs_shape: &[usize]) -> (usize, usize, usize) {
    let (m, k, n) = match lhs_shape.len() {
        2 => (lhs_shape[0], lhs_shape[1], rhs_shape[1]),
        3 => (lhs_shape[1], lhs_shape[2], rhs_shape[2]),
        4 => (lhs_shape[2], lhs_shape[3], rhs_shape[3]),
        _ => panic!("Unsupported shape dimension"),
    };
    (m, k, n)
}

pub fn compute_shape(lhs_shape: &[usize], m: usize, n: usize) -> Vec<usize> {
    let out_shape: Vec<usize> = match lhs_shape.len() {
        2 => vec![m, n],
        3 => vec![lhs_shape[0], m, n],
        4 => vec![lhs_shape[0], lhs_shape[1], m, n],
        _ => panic!("Unsupported shape dimension"),
    };
    out_shape
}

#[cfg(test)]
mod tests {
    use crate::device::cpu::Cpu;
    //use crate::device::cuda::Cuda;
    //use crate::device::vulkan::Vulkan;
    use crate::tensor::Tensor;
    // use rusty_vk::public_types::DeviceType;

    #[test]
    fn test_matmul() {
        let mut dev = Cpu::default();
        //let mut dev = Cuda::try_new(0).unwrap();
        //let mut dev = Vulkan::new(DeviceType::DiscreteGpu).unwrap();
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
