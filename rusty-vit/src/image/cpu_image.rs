use crate::device::DeviceStorage;
use crate::device::cpu::Cpu;
use crate::image::{Image, PixelType};


type ImageIndex = [usize; 4];

fn to_image_idx(index: ImageIndex, width: usize, height: usize, channels: usize) -> usize {
    let (B, C, X, Y) = (index[0], index[1], index[2], index[3]);
    let channel_size = width * height;
    let image_size = channels * channel_size;
    B * image_size + C * image_size + Y * width + X
}

impl<P: PixelType> std::ops::Index<ImageIndex> for Image<P, Cpu> {
    type Output = P;
    #[inline(always)]
    fn index(&self, index: ImageIndex) -> &Self::Output {
        let i = to_image_idx(index, self.width, self.height, self.channels);
        &self.data[i]
    }
}

impl<P: PixelType> std::ops::IndexMut<ImageIndex> for Image<P, Cpu> {
    #[inline(always)]
    fn index_mut(&mut self, index: ImageIndex) -> &mut Self::Output {
        let i = to_image_idx(index, self.width, self.height, self.channels);
        &mut self.data[i]
    }
}