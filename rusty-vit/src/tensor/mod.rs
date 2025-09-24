use crate::device::DeviceStorage;
use crate::type_traits::FloatType;
#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use num::traits::{FromBytes, ToBytes};
use num::{Float, Zero};
use std::error::Error;

mod convolution;
pub mod layer_norm;
pub mod matmul;
pub mod matrix_funcs;
mod tensor_cpu;

#[cfg(feature = "cuda")]
pub trait SafeZeros: ValidAsZeroBits + DeviceRepr {}
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}

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

        let dims = shape.len();
        let mut strides = vec![1; dims];
        for i in (0..(dims - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }

        let size = shape.iter().copied().reduce(|a, b| a * b).unwrap();
        Ok(Self {
            data: dev.try_alloc(size)?,
            device: dev.clone(),
            shape: shape.to_vec(),
            strides,
        })
    }
}
