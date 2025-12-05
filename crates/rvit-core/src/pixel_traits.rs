use crate::type_traits::BType;
use num::Zero;
use num::traits::{FromBytes, ToBytes};

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

pub trait BorderMode: Clone + Copy {}
#[derive(Debug, Clone, Copy, Default)]
pub struct Constant {}
impl BorderMode for Constant {}

#[derive(Debug, Clone, Copy, Default)]
pub struct ClampToEdge {}
impl BorderMode for ClampToEdge {}

#[derive(Debug, Clone, Copy, Default)]
pub struct Mirror {}
impl BorderMode for Mirror {}

pub trait InterpMode: Clone + Copy {}
#[derive(Default, Copy, Clone)]
pub struct Bilinear {}
impl InterpMode for Bilinear {}
