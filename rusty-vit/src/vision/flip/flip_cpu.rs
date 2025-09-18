use crate::device::cpu::Cpu;
use crate::image::{Image, PixelType};
use rand::Rng;
use rayon::prelude::*;

impl<T: PixelType> super::HorizFlipKernel<T> for Cpu {
    fn flip_horizontal(&mut self, src: &mut Image<T, Self>, prob: f32) {
        let total_image_size = src.width * src.height;
        let chunk_size = total_image_size * src.channels;

        src.data.par_chunks_mut(chunk_size).for_each(|in_image| {
            let mut rng = rand::rng();
            // Only flip the image if the generated random distribution is less than the
            // user-specified probability threshold.
            if rng.random_range(0.0..1.0) < prob {
                Self::flip_kernel(in_image, src.width, src.height, src.channels);
            }
        });
    }
}

impl Cpu {
    fn flip_kernel<TYPE: PixelType>(
        src: &mut [TYPE],
        width: usize,
        height: usize,
        channels: usize,
    ) {
        let half_size = height >> 1;
        for channels in 0..channels {
            for row in 0..half_size {
                for col in 0..width {
                    let base_offset = channels * col * width;
                    let bottom_offset = base_offset + (height - row - 1) * width;
                    let tmp = src[bottom_offset + col];
                    src[bottom_offset + col] = src[base_offset + row * width + col];
                    src[base_offset + row * width + col] = tmp;
                }
            }
        }
    }
}
