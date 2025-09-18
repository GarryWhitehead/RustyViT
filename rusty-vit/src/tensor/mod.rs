use std::error::Error;
use num::traits::{FromBytes, ToBytes};
use num::{Zero, Float};
use crate::device::DeviceStorage;
#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

pub mod layer_norm;
pub mod matmul;
pub mod matrix_funcs;
mod convolution;
mod tensor_cpu;

#[cfg(feature = "cuda")]
pub trait SafeZeros: ValidAsZeroBits + DeviceRepr {}
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}

pub trait FloatType:
'static
+ Copy
+ Clone
+ Default
+ std::fmt::Debug
+ PartialEq
+ PartialOrd
+ Send
+ Sync
+ SafeZeros
+ Zero
+ FromBytes
+ ToBytes
+ Float
{
    const ONE: Self;
}

impl SafeZeros for f32 {}
impl SafeZeros for f64 {}
impl FloatType for f32 {
    const ONE: Self = 1.0f32;
}
impl FloatType for f64 {
    const ONE: Self = 1.0f64;
}

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
            strides
        })
    }
}
