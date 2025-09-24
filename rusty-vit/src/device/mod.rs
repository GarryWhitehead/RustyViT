use crate::image::{Image, PixelType};
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
#[cfg(feature = "cuda")]
use cudarc::driver::DeviceRepr;
use std::error::Error;
use std::fmt::Debug;

pub mod cpu;
pub mod cu_utils;
#[cfg(feature = "cuda")]
pub mod cuda;

pub trait DeviceStorage<T>: Clone {
    type Vec: 'static + Clone + Send + Sync + Debug;

    fn try_alloc(&self, sz: usize) -> Result<Self::Vec, Box<dyn Error>>;

    fn try_alloc_with_slice(&self, slice: &[T]) -> Result<Self::Vec, Box<dyn Error>>;

    fn try_from_device_vec(&self, src: &Self::Vec) -> Result<Vec<T>, Box<dyn Error>>;

    fn len(vec: &Self::Vec) -> usize;

    fn try_sync_stream0(&self) -> Result<(), Box<dyn Error>>;
}

pub trait ToTensor<P: PixelType, I: DeviceStorage<P>, F: FloatType, D: DeviceStorage<F>> {
    fn to_tensor(
        &mut self,
        image: &Image<P, I>,
        norm: (&[F], &[F]),
    ) -> Result<Tensor<F, D>, Box<dyn Error>>;
}
