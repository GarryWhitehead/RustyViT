use crate::image::Image;
use rvit_core::pixel_traits::PixelType;
use rvit_device::cpu::device::Cpu;

type ImageIndex = [usize; 4];

fn to_image_idx(index: ImageIndex, width: usize, height: usize, channels: usize) -> usize {
    let (b, c, x, y) = (index[0], index[1], index[2], index[3]);
    let channel_size = width * height;
    let image_size = channels * channel_size;
    b * image_size + c * image_size + y * width + x
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
