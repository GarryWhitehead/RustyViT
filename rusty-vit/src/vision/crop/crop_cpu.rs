use crate::device::cpu::Cpu;
use crate::image::{Image, PixelType};
use rayon::prelude::*;

impl<T: PixelType> super::CropKernel<T> for Cpu {
    fn crop(
        &mut self,
        src: &mut Image<T, Self>,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> Image<T, Self>
    where
        Self: Sized,
    {
        let mut crop_img =
            Image::try_new(src.batch_size, crop_width, crop_height, src.channels, self).unwrap();
        src.data
            .par_chunks(src.strides[0])
            .zip(crop_img.data.par_chunks_mut(crop_img.strides[0]))
            .for_each(|(in_slice, out_slice)| {
                Self::crop_kernel(
                    in_slice,
                    src.width,
                    src.height,
                    src.channels,
                    crop_width,
                    crop_height,
                    x,
                    y,
                    out_slice,
                );
            });
        crop_img
    }
}

#[allow(clippy::too_many_arguments)]
impl Cpu {
    pub fn crop_kernel<T: PixelType>(
        src: &[T],
        width: usize,
        height: usize,
        channels: usize,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
        out: &mut [T],
    ) {
        assert!(crop_width <= width);
        assert!(crop_height <= height);
        assert!(x + crop_width <= width);
        assert!(y + crop_height <= height);

        // Nothing to be done if the cropped sizes match the original.
        if crop_width == width && crop_height == height {
            out.copy_from_slice(src);
            return;
        }

        for c in 0..channels {
            let src_offset = c * width * height;
            let crop_offset = c * crop_width * crop_height;
            let src_slice = &src[src_offset..src_offset + width * height];
            let crop_slice = &mut out[crop_offset..crop_offset + crop_width * crop_height];
            for row in 0..crop_height {
                let row_idx = y + row * width + x;
                crop_slice[row * crop_width..row * crop_width + crop_width]
                    .copy_from_slice(&src_slice[row_idx..row_idx + crop_width])
            }
        }
    }
}
