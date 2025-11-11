#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use half::f16;
use num::traits::{FromBytes, ToBytes};
use num::{Float, Zero};
use std::fmt::Debug;

#[cfg(feature = "cuda")]
pub trait SafeZeros: ValidAsZeroBits + DeviceRepr {}
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}
impl SafeZeros for u8 {}
impl SafeZeros for u16 {}
impl SafeZeros for f32 {}
impl SafeZeros for f16 {}

pub trait BType: Clone + SafeZeros + Zero + Debug + Send + Sync + 'static {}
impl BType for u8 {}
impl BType for u16 {}
impl BType for f32 {}
impl BType for f16 {}

pub trait FloatType:
    BType + Copy + Default + Debug + PartialEq + PartialOrd + Zero + FromBytes + ToBytes + Float
{
    const ONE: Self;
}

impl FloatType for f32 {
    const ONE: Self = 1.0f32;
}
impl FloatType for f16 {
    const ONE: Self = f16::ONE;
}
