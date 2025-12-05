pub use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use std::error::Error;

#[derive(Debug, Clone)]
pub struct Image<T, S: DeviceStorage<T>> {
    pub batch_size: usize,
    pub width: usize,
    pub height: usize,
    pub channels: usize,
    pub strides: Vec<usize>,
    pub data: S::Vec,
    pub device: S,
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

    pub fn from_parts(data: S::Vec, shape: &[usize], strides: &[usize], dev: &S) -> Image<T, S> {
        Self {
            batch_size: shape[0],
            width: shape[2],
            height: shape[3],
            channels: shape[1],
            strides: strides.to_vec(),
            data,
            device: dev.clone(),
        }
    }

    pub fn try_get_data(&self) -> Result<Vec<T>, Box<dyn Error>> {
        self.device.try_sync()?;
        self.device.try_from_device_vec(&self.data)
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
