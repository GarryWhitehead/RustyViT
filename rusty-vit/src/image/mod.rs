pub mod cpu_image;
#[cfg(feature = "cuda")]
pub mod cu_image;

use crate::device::DeviceStorage;
use crate::type_traits::{BType, SafeZeros};
#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use num::Zero;
use num::traits::{FromBytes, ToBytes};
use std::cell::RefCell;
use std::error::Error;
use std::ops::Add;
use std::sync::Arc;

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
    BType
    + Copy
    + Default
    + PartialEq
    + PartialOrd
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
        let buffer = dev.try_alloc_with_slice(slice)?;
        Ok(Image {
            batch_size,
            width,
            height,
            channels,
            strides: Self::compute_strides(batch_size, channels, width, height),
            data: buffer,
            device: dev.clone(),
        })
    }

    pub fn try_get_data(&self) -> Result<Vec<T>, Box<dyn Error>> {
        self.device.try_sync()?;
        Ok(self.device.try_from_device_vec(&self.data)?)
    }

    fn compute_strides(
        batch_size: usize,
        channels: usize,
        width: usize,
        height: usize,
    ) -> Vec<usize> {
        let shape = &[batch_size, channels, width, height];
        let mut strides = vec![1; 4];
        for i in (0..(shape.len() - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }
}
