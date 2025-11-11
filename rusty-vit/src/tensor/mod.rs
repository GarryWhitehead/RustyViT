use crate::device::DeviceStorage;
use crate::tensor::{cast::CastKernel, matmul::MatMulKernel};
use crate::type_traits::FloatType;
#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use std::error::Error;

mod cast;
pub mod convolution;
pub mod layer_norm;
pub mod matmul;
pub mod matrix_funcs;
mod tensor_cpu;

#[cfg(feature = "cuda")]
pub trait SafeZeros: ValidAsZeroBits + DeviceRepr {}
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}

#[allow(dead_code)]
#[derive(Clone)]
pub struct Tensor<T: FloatType, S: DeviceStorage<T>> {
    pub(crate) data: S::Vec,
    pub(crate) device: S,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
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

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let dims = shape.len();
        let mut strides = vec![1; dims];
        for i in (0..(dims - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn total_size(&self) -> usize {
        self.shape.iter().copied().reduce(|a, b| a * b).unwrap()
    }

    pub fn try_get_data(&self) -> Result<Vec<T>, Box<dyn Error>> {
        self.device.try_sync()?;
        self.device.try_from_device_vec(&self.data)
    }
}

impl<T: FloatType, D: MatMulKernel<T>> Tensor<T, D> {
    pub fn matmul(&self, rhs: &Tensor<T, D>, dev: &mut D) -> Tensor<T, D> {
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            panic!("Tensor must have at least two dimensions (for now).");
        }
        if self.shape.len() != rhs.shape.len() {
            panic!("Tensors must have the same dimensions.");
        }
        if self.shape.len() > 2 && (self.shape[0] != rhs.shape[0]) {
            panic!("Tensors must have the same batch size");
        }

        dev.matmul(self, rhs)
    }
}

impl<T: FloatType, D: DeviceStorage<T>> Tensor<T, D> {
    pub fn cast<O: FloatType>(&self, dev: &mut D) -> Tensor<O, D>
    where
        D: CastKernel<T, O>,
    {
        dev.cast(self)
    }
}
