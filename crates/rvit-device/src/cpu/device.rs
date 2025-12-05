use rvit_core::storage::DeviceStorage;
use std::fmt::Debug;
use std::ops::Range;
use std::{
    error::Error,
    ops::{Deref, DerefMut},
};

#[derive(Clone, Default)]
pub struct Cpu {}

#[derive(Clone, Debug, Default)]
pub struct VecPool<T> {
    pub data: Vec<T>,
}

impl<T: num::Zero + Clone> VecPool<T> {
    pub fn new(sz: usize) -> Self {
        Self {
            data: vec![T::zero(); sz],
        }
    }
}

impl<T> Deref for VecPool<T> {
    type Target = Vec<T>;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T> DerefMut for VecPool<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T> std::ops::Index<usize> for VecPool<T> {
    type Output = T;
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<T> std::ops::Index<Range<usize>> for VecPool<T> {
    type Output = [T];
    fn index(&self, range: Range<usize>) -> &Self::Output {
        &self.data[range]
    }
}

impl<T> std::ops::IndexMut<usize> for VecPool<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T> std::ops::IndexMut<Range<usize>> for VecPool<T> {
    fn index_mut(&mut self, index: Range<usize>) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<T: Clone + Copy + num::Zero + Sync + Send + Debug + 'static> DeviceStorage<T> for Cpu {
    type Vec = VecPool<T>;

    fn try_alloc(&self, sz: usize) -> Result<Self::Vec, Box<dyn Error>> {
        // Just using a simple vector for now. Will be updated to a memory pool/arena.
        Ok(VecPool::new(sz))
    }

    fn try_alloc_with_slice(&self, slice: &[T]) -> Result<Self::Vec, Box<dyn Error>> {
        let mut v = VecPool::new(slice.len());
        v.data.copy_from_slice(slice);
        Ok(v)
    }

    fn try_from_device_vec(&self, src: &Self::Vec) -> Result<Vec<T>, Box<dyn Error>> {
        Ok(src.data.clone())
    }

    fn len(vec: &Self::Vec) -> usize {
        vec.len()
    }

    fn slice(v: &Self::Vec) -> &[T] {
        v.as_slice()
    }

    fn slice_mut(v: &mut Self::Vec) -> &mut [T] {
        v.as_mut_slice()
    }

    fn try_sync(&self) -> Result<(), Box<dyn Error>> {
        Ok(())
    }
}

/*impl<P: PixelType, F: FloatType> ToTensor<P, Self, F, Self> for Cpu {
    fn to_tensor(
        &mut self,
        image: &Image<P, Self>,
        norm: (&[F], &[F]),
    ) -> Result<Tensor<F, Self>, Box<dyn Error>> {
        if norm.0.len() != norm.1.len() {
            panic!("Mean and std arrays are different lengths");
        }
        let mut tensor: Tensor<F, Self> = Tensor::try_new(
            &[image.batch_size, image.channels, image.width, image.height],
            self,
        )?;

        // Slice for [W, H, C]
        for b in 0..image.batch_size {
            for c in 0..image.channels {
                for y in 0..image.height {
                    for x in 0..image.width {
                        let pixel: P = image[[b, c, x, y]];
                        // f = (p[channel] - mean[channel]) / std[channel]
                        tensor[[b, c, y, x].to_vec()] =
                            (F::from(pixel).unwrap() - norm.0[c]) / norm.1[c];
                    }
                }
            }
        }
        Ok(tensor)
    }
}*/
