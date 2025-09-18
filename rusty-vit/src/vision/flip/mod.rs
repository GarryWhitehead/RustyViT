mod flip_cpu;
#[cfg(feature = "cuda")]
mod flip_cu;

use crate::device::DeviceStorage;
use crate::image::Image;

pub trait HorizFlipKernel<T>: DeviceStorage<T> {
    fn flip_horizontal(&mut self, src: &mut Image<T, Self>, prob: f32)
    where
        Self: Sized;
}

pub struct RandomFlipHorizontal {
    probability: f32,
}

impl RandomFlipHorizontal {
    pub fn new(probability: f32) -> Self {
        Self { probability }
    }
}

impl RandomFlipHorizontal {
    fn flip<T, S: HorizFlipKernel<T>>(&self, image: &mut Image<T, S>) {
        if image.height & (image.height - 1) != 0 {
            panic!(
                "Image dimensions must be a power of two. Image dims: {}",
                image.height
            );
        }
        let dev = &mut image.device.clone();
        dev.flip_horizontal(image, self.probability);
    }
}

#[cfg(not(feature = "cuda"))]
pub fn flip_vertical(image: &[u8], image_size: usize, stride: usize, out: &mut [u8]) {
    assert!(image_size & (image_size - 1) == 0);

    let half_size = image_size / 2;

    for row in 0..image_size {
        for col in 0..half_size {
            let left_idx = row * stride + col;
            let right_idx = (image_size - col - 1) + row * stride;
            let left_pixel = image[left_idx];
            out[left_idx] = image[right_idx];
            out[right_idx] = left_pixel;
        }
    }
}

mod tests {
    use super::*;
    use crate::device::cpu::Cpu;
    #[cfg(feature = "cuda")]
    use crate::device::cuda::Cuda;

    #[test]
    fn test_flip_horizontal() {
        let src: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
        let dev = Cuda::try_new(0).unwrap();
        //let dev = Cpu::default();
        let flipper = RandomFlipHorizontal::new(2.0);
        let mut img = Image::try_from_slice(&src, 1, 4, 4, 1, &dev).unwrap();
        flipper.flip(&mut img);
        assert_eq!(
            img.try_get_data().unwrap(),
            &[4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
        );
    }

    #[test]
    #[cfg(not(feature = "cuda"))]
    fn test_flip_vertical() {
        let src: Vec<u8> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let mut dst = vec![0u8; src.len()];
        flip_vertical(&src, 4, 4, &mut dst);
        assert_eq!(&dst, &[4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1]);
    }
}
