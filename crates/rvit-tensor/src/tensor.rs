use rand::distr::Distribution;
use rand::distr::uniform::SampleUniform;
use rvit_core::storage::DeviceStorage;
use rvit_core::type_traits::FloatType;
use std::error::Error;

#[allow(dead_code)]
#[derive(Clone)]
pub struct Tensor<T: FloatType, S: DeviceStorage<T>> {
    pub data: S::Vec,
    pub device: S,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}

impl<T: FloatType, S: DeviceStorage<T>> Tensor<T, S> {
    pub fn try_new(shape: &[usize], dev: &S) -> Result<Self, Box<dyn Error>> {
        if shape.is_empty() {
            return Err("shape cannot be empty".into());
        }

        let sz = shape.iter().copied().reduce(|a, b| a * b).unwrap();
        Ok(Self {
            data: dev.try_alloc(sz)?,
            device: dev.clone(),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        })
    }

    pub fn from_parts(data: S::Vec, shape: &[usize], strides: &[usize], device: &S) -> Self {
        if shape.len() != strides.len() {
            panic!("shape and strides must be the same length");
        }
        Self {
            data,
            device: device.clone(),
            shape: shape.to_vec(),
            strides: strides.to_vec(),
        }
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let dims = shape.len();
        let mut strides = vec![1; dims];
        for i in (0..(dims - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
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

    pub fn try_from_data(shape: &[usize], values: &[T], dev: &S) -> Result<Self, Box<dyn Error>> {
        if shape.is_empty() {
            return Err("shape cannot be empty".into());
        }
        let sz = shape.iter().copied().reduce(|a, b| a * b).unwrap();
        if values.len() != sz {
            return Err("Shape size doesn't match values length".into());
        }

        Ok(Self {
            data: dev.try_alloc_with_slice(values)?,
            device: dev.clone(),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        })
    }

    pub fn total_size(&self) -> usize {
        self.shape.iter().copied().reduce(|a, b| a * b).unwrap()
    }

    pub fn try_get_data(&self) -> Result<Vec<T>, Box<dyn Error>> {
        self.device.try_sync()?;
        self.device.try_from_device_vec(&self.data)
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
    use rvit_device::tests::TestDevice;

    #[test]
    fn test_is_contiguous() {
        let dev = TestDevice::default();
        let mut t: Tensor<f32, _> = Tensor::try_new(&[1, 2, 3, 3], &dev).unwrap();
        assert!(t.is_contiguous());

        t.permute(&[0, 2, 3, 1]);
        assert!(!t.is_contiguous());
    }

    #[test]
    fn test_permute() {
        let dev = TestDevice::default();
        let mut t: Tensor<f32, _> = Tensor::try_new(&[1, 2, 3, 3], &dev).unwrap();
        t.permute(&[0, 2, 3, 1]);
        assert_eq!(&t.shape, &[1, 3, 3, 2]);
        assert_eq!(t.strides, vec![18, 3, 1, 9]);
    }
}
