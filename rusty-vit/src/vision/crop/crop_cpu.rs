use crate::device::DeviceStorage;
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
    ) -> Self::Vec
    where
        Self: Sized,
    {
        let image_size = src.width * src.height;
        let chunk_size = src.channels * image_size;

        let crop_image_size = crop_width * crop_height;
        let crop_chunk_size = src.channels * crop_image_size;
        let mut crop_data = src
            .device
            .try_alloc(src.batch_size * crop_chunk_size)
            .unwrap();
        src.data
            .par_chunks(chunk_size)
            .zip(crop_data.data.par_chunks_mut(crop_chunk_size))
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
        crop_data
    }
}

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
        }
        
        for c in 0..channels {
            let src_channel_offset = c * width * height;
            let crop_channel_offset = c * crop_width * crop_height;
            let src_channel_slice = &src[src_channel_offset..src_channel_offset + width * height];
            let crop_channel_slice = &mut out[crop_channel_offset..crop_channel_offset + crop_width * crop_height];
            for row in y..crop_height + y {
                let row_idx = row * width + x;
                crop_channel_slice[row - y * crop_width..row - y * crop_width + crop_width]
                    .copy_from_slice(&src_channel_slice[row_idx..row_idx + crop_width])
            }
        }
    }
}
