use crate::Device;
use crate::float_ops::TensorFloatOps;
use crate::tensor_ops::TensorOps;
use rvit_core::element_traits::{FloatElem, IntElem};
use rvit_device::{DAlloc, Device};
use std::fmt::Debug;
use std::marker::PhantomData;

#[derive(Debug, Clone, Copy)]
pub struct Float {}

#[derive(Debug, Clone, Copy)]
pub struct Integer {}

pub trait TensorType: Clone + Copy {}

impl TensorType for Float {}
impl TensorType for Integer {}

#[derive(Clone)]
pub struct Tensor<T: TensorType, D: Device> {
    pub data: DAlloc<D>,
    pub device: D::Storage,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub(crate) phantom_data: PhantomData<T>,
}

impl<T: TensorType, S: Device> Tensor<T, S> {
    pub(crate) fn compute_memory_size(shape: &[usize]) -> usize {
        shape.iter().copied().reduce(|a, b| a * b).unwrap()
    }

    pub(crate) fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let dims = shape.len();
        let mut strides = vec![1; dims];
        for i in (0..(dims - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn from_parts(
        data: DAlloc<S>,
        shape: &[usize],
        strides: &[usize],
        device: &S::Storage,
    ) -> Self {
        if shape.len() != strides.len() {
            panic!("shape and strides must be the same length");
        }
        Self {
            data,
            device: device.clone(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
            phantom_data: PhantomData,
        }
    }

    pub fn total_size(&self) -> usize {
        self.shape.iter().copied().reduce(|a, b| a * b).unwrap()
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for (shape, stride) in self.shape.iter().zip(&self.strides).rev() {
            if *stride != expected_stride {
                return false;
            }
            expected_stride *= shape;
        }
        true
    }

    pub fn permute(&mut self, indices: &[usize]) {
        if indices.len() != self.shape.len() {
            panic!("Indices length doesn't match the tensor dimensions");
        }

        self.shape = indices.iter().map(|i| self.shape[*i]).collect();
        self.strides = indices.iter().map(|i| self.strides[*i]).collect();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FloatTensor;
    use crate::basic_ops::*;
    use half::f16;
    use rvit_device::tests::TestDevice;
    use rvit_device::{DeviceRuntime, Runtime};

    #[test]
    fn test_is_contiguous() {
        let mut dev: DeviceRuntime<_, f32, i32> = Runtime::new(Default::default());
        let mut t = FloatTensor::new(&[1, 2, 3, 3], &mut dev.storage);
        assert!(t.is_contiguous());

        t.permute(&[0, 2, 3, 1]);
        assert!(!t.is_contiguous());
    }

    #[test]
    fn test_permute() {
        let mut dev: DeviceRuntime<_, f32, i32> = Runtime::new(TestDevice::default());
        let mut t = FloatTensor::new(&[1, 2, 3, 3], &mut dev.storage);
        t.permute(&[0, 2, 3, 1]);
        assert_eq!(&t.shape, &[1, 3, 3, 2]);
        assert_eq!(t.strides, vec![18, 3, 1, 9]);
    }

    #[test]
    fn test_from_data_convert_shape1() {
        let d = TestDevice::default();
        let mut dev: DeviceRuntime<_, f32, i32> = Runtime::new(d);
        let t = FloatTensor::from_data_shape1(&[1.0, 2.0, 3.0], &mut dev.storage);
        assert_eq!(t.shape, &[3]);
        assert_eq!(t.total_size(), 3);
    }

    #[test]
    fn test_from_data_convert_shape2() {
        let mut dev: DeviceRuntime<_, f32, i32> = Runtime::new(TestDevice::default());
        let t =
            FloatTensor::from_data_shape2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &mut dev.storage);
        assert_eq!(t.shape, &[2, 3]);
        assert_eq!(t.total_size(), 6);
    }

    #[test]
    fn test_from_data_convert_shape3() {
        let mut dev = DeviceRuntime::<TestDevice, f16, i32>::new(TestDevice::default());
        let t = FloatTensor::from_data_shape3(
            &[
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[4.0, 3.0, 2.0], [9.0, 8.0, 7.0]],
            ],
            &mut dev.storage,
        );
        assert_eq!(t.shape, &[2, 2, 3]);
        assert_eq!(t.total_size(), 12);
    }

    #[test]
    fn test_from_data_convert_shape4() {
        let mut dev = DeviceRuntime::<TestDevice, f16, i32>::new(TestDevice::default());
        let t = FloatTensor::from_data_shape4(
            &[[
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[4.0, 3.0, 2.0], [9.0, 8.0, 7.0]],
            ]],
            &mut dev.storage,
        );
        assert_eq!(t.shape, &[1, 2, 2, 3]);
        assert_eq!(t.total_size(), 12);
    }
}
