use num::Zero;
use rand::prelude::SliceRandom;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefIterator;
use rayon::iter::ParallelIterator;
use rayon::prelude::ParallelSliceMut;
use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use rvit_image::image::Image;
use std::error::Error;
use std::marker::PhantomData;

pub mod cifar;

#[derive(Debug, Clone, Copy)]
pub enum DataType {
    TEST,
    TRAINING,
}

pub struct VitData<P: PixelType, D: DeviceStorage<P>> {
    pub images: Image<P, D>,
    pub labels: D::Vec,
}

impl<P: PixelType, D: DeviceStorage<P>> VitData<P, D> {
    pub fn try_new(
        width: usize,
        height: usize,
        channels: usize,
        count: usize,
        dev: &D,
    ) -> Result<VitData<P, D>, Box<dyn Error>> {
        Ok(VitData {
            images: Image::try_new(width, height, channels, count, dev)?,
            labels: dev.try_alloc(count)?,
        })
    }
}

pub trait DataSetReader {
    type Type: PixelType + cifar::DataSetBytes;

    fn read_bytes_from_buffer(file: &str, out: &[Self::Type]) -> Result<(), Box<dyn Error>>;
}

pub trait DataSetFormat {
    type Type: PixelType + Sized;
    const IMAGE_FORMAT_TOTAL_IMAGE_SIZE: usize;
    const IMAGE_FORMAT_IMAGES_PER_FILE: usize;
    const IMAGE_FORMAT_DIM: usize;
    const IMAGE_FORMAT_CHANNELS: usize;
    const IMAGE_FORMAT_LABEL_SIZE: usize;

    fn get_training_file(idx: usize) -> &'static str;
    fn get_test_file(idx: usize) -> &'static str;

    fn read_bytes_from_buffer(
        file: &str,
        image_idx: usize,
        out: &mut [Self::Type],
    ) -> Result<u8, Box<dyn Error>>;
}

#[derive(Clone, Debug)]
pub struct DataLoader<F: DataSetFormat> {
    path: String,
    indices: Vec<usize>,
    pub(crate) batch_size: usize,
    pub(crate) width: usize,
    pub(crate) height: usize,
    pub(crate) channels: usize,
    current: usize,
    data_type: DataType,
    // Temporary buffer storage for CUDA device - images
    // are first copied to the temporary workspace, then
    // uploaded to the device with a single memcpy.
    workspace: Vec<F::Type>,
    phantom_data: PhantomData<F>,
}

#[allow(clippy::too_many_arguments)]
impl<F: DataSetFormat<Type = u8>> DataLoader<F> {
    pub fn try_new<D: DeviceStorage<F::Type>>(
        path: &str,
        batch_size: usize,
        width: usize,
        height: usize,
        channels: usize,
        shuffle_order: bool,
        drop: bool,
        data_type: DataType,
    ) -> Result<DataLoader<F>, Box<dyn Error>> {
        let mut indices: Vec<usize> = (0..F::IMAGE_FORMAT_TOTAL_IMAGE_SIZE).collect();
        if drop {
            let remainder = F::IMAGE_FORMAT_TOTAL_IMAGE_SIZE % batch_size;
            if remainder > 0 {
                indices.truncate(remainder);
            }
        }
        if shuffle_order {
            let mut rng = rand::rng();
            indices.shuffle(&mut rng);
        }

        Ok(DataLoader {
            path: path.to_string(),
            indices,
            batch_size,
            width,
            height,
            channels,
            current: 0,
            data_type,
            workspace: vec![F::Type::zero(); batch_size * channels * width * height],
            phantom_data: PhantomData,
        })
    }

    #[allow(clippy::type_complexity)]
    pub fn next_batch<D: DeviceStorage<F::Type>>(
        &mut self,
        dev: &D,
    ) -> Option<(D::Vec, Image<F::Type, D>)> {
        if self.current >= self.indices.len() {
            return None;
        }
        let (labels, image) = self.load_batch(dev).unwrap();
        // Update the current index ready for the "next" call.
        self.current += self.batch_size;

        Some((labels, image))
    }

    #[allow(clippy::type_complexity)]
    fn load_batch<D: DeviceStorage<F::Type>>(
        &mut self,
        dev: &D,
    ) -> Result<(D::Vec, Image<F::Type, D>), Box<dyn Error>> {
        let stride = F::IMAGE_FORMAT_DIM * F::IMAGE_FORMAT_DIM * F::IMAGE_FORMAT_CHANNELS;
        let chunk_size = F::IMAGE_FORMAT_LABEL_SIZE + stride;

        let mut labels = vec![0u8; self.batch_size];
        let batch_i = &self.indices[self.current..self.current + self.batch_size];

        let image: Image<F::Type, D> =
            Image::try_new(self.batch_size, self.width, self.height, self.channels, dev)?;

        // Load the images into the temp workspace to begin with; this is to reduce the
        // number of uploads to the device in the case of CUDA/Vulkan.
        batch_i
            .par_iter()
            .zip(self.workspace.par_chunks_mut(chunk_size))
            .zip(labels.par_chunks_mut(1))
            .for_each(|((i, image), l)| {
                let file_idx = i / F::IMAGE_FORMAT_IMAGES_PER_FILE;
                let image_idx = i % F::IMAGE_FORMAT_IMAGES_PER_FILE;
                let file = match self.data_type {
                    DataType::TRAINING => F::get_training_file(file_idx),
                    DataType::TEST => F::get_test_file(file_idx),
                };

                let file_path = format!("{}/{}", self.path, file);
                l[0] = F::read_bytes_from_buffer(file_path.as_str(), image_idx, image).unwrap();
            });

        let dev_labels = dev.try_alloc_with_slice(labels.as_slice())?;
        Ok((dev_labels, image))
    }
}
