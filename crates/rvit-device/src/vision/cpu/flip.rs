use crate::cpu::device::Cpu;
use crate::vision::op_traits::HorizontalFlipKernel;
use crate::{DAlloc, Runtime};
use rand::Rng;
use rayon::prelude::*;
use rvit_core::element_traits::DataElem;

impl<T: DataElem> HorizontalFlipKernel<T> for Runtime {
    fn flip_horizontal(
        &mut self,
        src: &mut DAlloc<Self>,
        src_shape: &[usize],
        src_strides: &[usize],
        prob: f32,
    ) {
        let chunk_size = src_strides[0];
        let (channels, width, height) = (src_shape[1], src_shape[2], src_shape[3]);
        src.as_mut_slice()
            .unwrap()
            .par_chunks_mut(chunk_size)
            .for_each(|in_image: &mut [T]| {
                let mut rng = rand::rng();
                // Only flip the image if the generated random distribution is less than the
                // user-specified probability threshold.
                if rng.random_range(0.0..1.0) < prob {
                    Self::flip_kernel(in_image, width, height, channels);
                }
            });
    }
}

impl Runtime {
    fn flip_kernel<TYPE: DataElem>(src: &mut [TYPE], width: usize, height: usize, channels: usize) {
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
