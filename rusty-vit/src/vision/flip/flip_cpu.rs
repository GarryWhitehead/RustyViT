use crate::device::cpu::Cpu;
use crate::image::{Image, PixelType};
use rand::Rng;
use rayon::prelude::*;

impl<T: PixelType> super::HorizFlipKernel<T> for Cpu {
    fn flip_horizontal(&mut self, src: &mut Image<T, Self>, prob: f32) {
        let chunk_size = src.strides[0];
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
        for c in 0..channels {
            for row in 0..half_size {
                for col in 0..width {
                    let base_offset = c * width * height;
                    let bottom_offset = base_offset + (height - row - 1) * width;
                    src.swap(bottom_offset + col, base_offset + row * width + col);
                }
            }
        }
    }
}
