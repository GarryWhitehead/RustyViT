use std::error::Error;
use std::fmt::Debug;
use std::ops::Range;

pub trait DeviceStorage<T>: Clone {
    type Vec: 'static
        + Clone
        + Send
        + Sync
        + Debug
        + std::ops::Index<usize, Output = T>
        + std::ops::Index<Range<usize>, Output = [T]>
        + std::ops::IndexMut<usize, Output = T>
        + std::ops::IndexMut<Range<usize>, Output = [T]>;

    fn try_alloc(&self, sz: usize) -> Result<Self::Vec, Box<dyn Error>>;

    fn try_alloc_with_slice(&self, slice: &[T]) -> Result<Self::Vec, Box<dyn Error>>;

    fn try_from_device_vec(&self, src: &Self::Vec) -> Result<Vec<T>, Box<dyn Error>>;

    fn len(vec: &Self::Vec) -> usize;

    fn slice(v: &Self::Vec) -> &[T];

    fn slice_mut(v: &mut Self::Vec) -> &mut [T];

    fn try_sync(&self) -> Result<(), Box<dyn Error>>;
}

/*pub trait ToTensor<P: PixelType, I: DeviceStorage<P>, F: FloatType, D: DeviceStorage<F>> {
    fn to_tensor(
        &mut self,
        image: &Image<P, I>,
        norm: (&[F], &[F]),
    ) -> Result<Tensor<F, D>, Box<dyn Error>>;
}*/
