pub mod convolution;
pub mod crop;
pub mod flip;
pub mod make_border;
pub mod resize;

use crate::device::DeviceStorage;
use crate::image::Image;

pub trait ImageProcessing<T, S: DeviceStorage<T>>: Send + Sync {
    fn process(&self, image: &Image<T, S>) -> Image<T, S>;
}

/*
#[derive(Clone, Debug)]
pub struct Image {
    #[cfg(not(feature = "cuda"))]
    pub data: Vec<u8>,
    pub dim: usize,
    pub channels: usize,
    #[cfg(feature = "cuda")]
    pub dmem: CudaSlice<u8>,
}

impl Image {
    #[cfg(not(feature = "cuda"))]
    pub fn new(dim: usize, channels: usize, _i: CpuInstance) -> Self {
        Self {
            data: vec![0; dim * dim * channels],
            dim,
            channels,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn from(other: &Image) -> Self {
        Self {
            data: other.data.clone(),
            dim: other.dim,
            channels: other.channels,
        }
    }

    #[cfg(not(feature = "cuda"))]
    pub fn from_slice(other_data: &[u8], dim: usize, channels: usize) -> Self {
        Self {
            data: other_data.to_vec(),
            dim,
            channels,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn new(dim: usize, channels: usize, instance: Arc<CuInstance>) -> Self {
        let stream = instance.ctx.default_stream();
        let dmem = unsafe { stream.alloc::<u8>(dim * dim * channels).unwrap() };

        Self {
            dim,
            channels,
            dmem,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn from_slice(
        other_data: &[u8],
        dim: usize,
        channels: usize,
        instance: Arc<CuInstance>,
    ) -> Self {
        let mut image = Self::new(dim, channels, instance.clone());
        let stream = instance.ctx.default_stream();
        stream
            .memcpy_htod(other_data, &mut image.dmem)
            .unwrap();
        Self {
            dim,
            channels,
            dmem: image.dmem,
        }
    }

    #[cfg(feature = "cuda")]
    pub fn download(&self, cui: Arc<CuInstance>) -> Vec<u8> {
        let mut data: Vec<u8> = Vec::new();
        cui.ctx.default_stream().memcpy_dtoh(&self.dmem, &mut data);
        data
    }

    pub fn total_size(&self) -> usize {
        self.channel_size() * self.channels
    }

    pub fn channel_size(&self) -> usize {
        self.dim * self.dim
    }

    pub fn channel_slice(&self, channel: usize) -> &[u8] {
        let channel_size = self.dim * self.dim;
        &self.data[channel_size * channel..channel_size * channel + channel_size]
    }

    pub fn channel_slice_mut(&mut self, channel: usize) -> &mut [u8] {
        let channel_size = self.dim * self.dim;
        &mut self.data[channel_size * channel..channel_size * channel + channel_size]
    }
}

#[derive(Clone, Debug)]
pub struct ImageArray {
    pub data: Vec<u8>,
    pub dim: usize,
    pub channels: usize,
    pub image_count: usize,
}

impl ImageArray {
    pub fn new(dim: usize, channels: usize, image_count: usize) -> Self {
        Self {
            data: vec![0; dim * dim * channels * image_count],
            dim,
            channels,
            image_count,
        }
    }

    pub fn from(other: &ImageArray) -> Self {
        Self {
            data: other.data.clone(),
            dim: other.dim,
            channels: other.channels,
            image_count: other.image_count,
        }
    }

    pub fn from_slice(slice: &[u8], dim: usize, channels: usize, image_count: usize) -> Self {
        Self {
            data: slice.to_vec(),
            dim,
            channels,
            image_count,
        }
    }

    pub fn total_size(&self) -> usize {
        self.dim * self.dim * self.channels * self.image_count
    }

    pub fn size_per_image(&self) -> usize {
        self.dim * self.dim * self.channels
    }

    pub fn to_slice_u8(&self, idx: usize, count: usize) -> &[u8] {
        self.data[idx..idx + count].as_ref()
    }

    pub fn to_image_slice(&self, idx: usize) -> Image {
        let count = self.dim * self.dim * self.channels;
        Image::from_slice(&self.data[idx..idx + count], self.dim, self.channels)
    }
}*/
