#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use half::f16;
use num::traits::{FromBytes, ToBytes};
use num::{Float, Integer, Zero};
use std::fmt::Debug;
use std::iter::Sum;
use std::ops::{Add, Div, DivAssign, Mul, MulAssign, Sub};

#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub enum DataType {
    F16,
    F32,
    U8,
    U16,
    U32,
    I8,
    I16,
    I32,
}

#[cfg(feature = "cuda")]
pub trait SafeZeros: ValidAsZeroBits + DeviceRepr {}
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}
impl SafeZeros for u8 {}
impl SafeZeros for u16 {}
impl SafeZeros for u32 {}
impl SafeZeros for i8 {}
impl SafeZeros for i16 {}
impl SafeZeros for i32 {}

impl SafeZeros for f32 {}
impl SafeZeros for f16 {}

pub trait Elem:
    Clone
    + Copy
    + SafeZeros
    + Zero
    + PartialEq
    + Debug
    + Default
    + Send
    + Sync
    + PartialOrd
    + FromBytes
    + ToBytes
    + num::traits::cast::FromPrimitive
    + num::NumCast
    + 'static
{
}
impl Elem for u8 {}
impl Elem for u16 {}
impl Elem for u32 {}
impl Elem for i8 {}
impl Elem for i16 {}
impl Elem for i32 {}
impl Elem for f32 {}
impl Elem for f16 {}

pub trait DataElem:
    Elem
    + Sum
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Div<Self, Output = Self>
    + Mul<Self, Output = Self>
    + DivAssign
    + MulAssign
{
    const ONE: Self;
    const DTYPE: DataType;
}

impl DataElem for f32 {
    const ONE: Self = 1.0f32;
    const DTYPE: DataType = DataType::F32;
}
impl DataElem for f16 {
    const ONE: Self = f16::ONE;
    const DTYPE: DataType = DataType::F16;
}
impl DataElem for u8 {
    const ONE: Self = 1u8;
    const DTYPE: DataType = DataType::U8;
}
impl DataElem for u16 {
    const ONE: Self = 1u16;
    const DTYPE: DataType = DataType::U16;
}
impl DataElem for u32 {
    const ONE: Self = 1u32;
    const DTYPE: DataType = DataType::U32;
}
impl DataElem for i8 {
    const ONE: Self = 1i8;
    const DTYPE: DataType = DataType::I8;
}
impl DataElem for i16 {
    const ONE: Self = 1i16;
    const DTYPE: DataType = DataType::I16;
}
impl DataElem for i32 {
    const ONE: Self = 1i32;
    const DTYPE: DataType = DataType::I32;
}

pub trait FloatElem: DataElem + Float {}

pub trait IntElem: DataElem + Integer {}

impl IntElem for u8 {}
impl IntElem for u16 {}
impl IntElem for u32 {}
impl IntElem for i8 {}
impl IntElem for i16 {}
impl IntElem for i32 {}

impl FloatElem for f32 {}
impl FloatElem for f16 {}
