pub mod cpu_image;
#[cfg(feature = "cuda")]
pub mod cu_image;

use crate::device::DeviceStorage;
use num::Zero;
use num::traits::{FromBytes, ToBytes};
use std::error::Error;
#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};

#[cfg(feature = "cuda")]
pub trait SafeZeros: ValidAsZeroBits + DeviceRepr {}
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}

pub trait ToFloat: Default + Copy + Clone + 'static {
    fn to_float(self) -> f32;
    fn from_float(f: f32) -> Self;
}

impl ToFloat for u8 {
    fn to_float(self) -> f32 {
        f32::from(self)
    }
    fn from_float(f: f32) -> Self {
        f as u8
    }
}

impl ToFloat for u16 {
    fn to_float(self) -> f32 {
        f32::from(self)
    }
    fn from_float(f: f32) -> Self {
        f as u16
    }
}

impl ToFloat for f32 {
    fn to_float(self) -> f32 {
        self
    }
    fn from_float(f: f32) -> Self {
        f
    }
}

pub trait PixelType:
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
    + num::traits::NumCast
    + num::traits::cast::AsPrimitive<u8>
    + num::traits::cast::AsPrimitive<u16>
    + num::traits::cast::AsPrimitive<f32>
    + num::traits::cast::FromPrimitive
    + ToFloat
{
    const ONE: Self;
}

impl SafeZeros for u8 {}
impl SafeZeros for u16 {}
impl SafeZeros for f32 {}

impl PixelType for u8 {
    const ONE: Self = 1u8;
}
impl PixelType for u16 {
    const ONE: Self = 1u16;
}
impl PixelType for f32 {
    const ONE: Self = 1f32;
}

#[derive(Debug, Clone)]
pub struct Image<T, S: DeviceStorage<T>> {
    pub(crate) batch_size: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
    pub(crate) strides: Vec<usize>,
    pub(crate) data: S::Vec,
    pub(crate) device: S,
}

impl<T: PixelType, S: DeviceStorage<T>> Image<T, S> {
    pub fn try_new(
        batch_size: usize,
        width: usize,
        height: usize,
        channels: usize,
        dev: &S,
    ) -> Result<Self, Box<dyn Error>> {
        let d = dev.try_alloc(batch_size * width * height * channels)?;
        Ok(Self {
            batch_size,
            width,
            height,
            channels,
            strides: Self::compute_strides(batch_size, channels, width, height),
            data: d,
            device: dev.clone(),
        })
    }

    pub fn try_from_slice(
        slice: &[T],
        batch_size: usize,
        width: usize,
        height: usize,
        channels: usize,
        dev: &S,
    ) -> Result<Image<T, S>, Box<dyn Error>> {
        Ok(Image {
            batch_size,
            width,
            height,
            channels,
            strides: Self::compute_strides(batch_size, channels, width, height),
            data: dev.try_alloc_with_slice(slice)?,
            device: dev.clone(),
        })
    }

    pub fn try_get_data(&self) -> Result<Vec<T>, Box<dyn Error>> {
        Ok(self.device.try_from_device_vec(&self.data)?)
    }

    fn compute_strides(batch_size: usize, channels: usize, width: usize, height: usize) -> Vec<usize> {
        let shape = &[batch_size, channels, width, height];
        let mut strides = vec![1; 4];
        for i in (0..(shape.len() - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}




