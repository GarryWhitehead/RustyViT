mod crop_cpu;
#[cfg(feature = "cuda")]
mod crop_cu;

use crate::vision::{Image};
use rand::Rng;
use rand::distr::Uniform;
use crate::device::DeviceStorage;

pub trait CropKernel<T>: DeviceStorage<T> {
    fn crop(&mut self, src: &mut Image<T, Self>, crop_width: usize, crop_height: usize, x: usize, y: usize)  -> Self::Vec
    where
        Self: Sized;
}

struct RandomCrop {
    crop_width: usize,
    crop_height: usize,
    x: usize,
    y: usize,
}

impl RandomCrop {
    pub fn new(dim: usize, crop_width: usize, crop_height: usize) -> Self {
        if crop_width > dim {
            panic!(
                "Crop width must be less than the src dimensions; crop_width: {crop_width} vs \
                image dim: {dim}"
            );
        }
        if crop_height > dim {
            panic!(
                "Crop height must be less than the src dimensions; crop_height: {crop_height} \
            vs image dim: {dim}"
            );
        };

        let uniform_x = Uniform::try_from(0..dim - crop_width + 1).unwrap();
        let uniform_y = Uniform::try_from(0..dim - crop_height + 1).unwrap();
        let mut rng = rand::rng();
        Self {
            crop_width,
            crop_height,
            x: rng.sample(uniform_x),
            y: rng.sample(uniform_y),
        }
    }
}

/*mod tests {
    use super::*;

    #[test]
    fn test_crop_matching_dims() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut dst = vec![0u8; 3 * 3];
        crop(src, 3, 3, 3, 0, 0, &mut dst);
        assert_eq!(dst, src);
    }

    #[test]
    fn test_crop_1() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0u8; 3 * 3];
        crop(src, 4, 3, 3, 0, 0, &mut dst);
        assert_eq!(dst, &[1, 2, 3, 5, 6, 7, 9, 10, 11]);
    }

    #[test]
    fn test_crop_3() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0u8; 3 * 3];
        crop(src, 4, 3, 3, 1, 1, &mut dst);
        assert_eq!(dst, &[6, 7, 8, 10, 11, 12, 14, 15, 16]);
    }

    #[test]
    fn test_crop_2() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0u8; 2 * 2];
        crop(src, 4, 2, 2, 2, 2, &mut dst);
        assert_eq!(dst, &[11, 12, 15, 16]);
    }

    #[test]
    fn test_crop_process() {
        let src_image = Image::from_slice(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            4,
            1,
        );
        let cropper = RandomCrop::new(4, 3, 3);
        let out_image = cropper.process(&src_image);
        assert_eq!(out_image.dim, 3);
        assert_eq!(out_image.channels, 1);
    }

    #[test]
    fn test_crop_batched() {
        let src_image = ImageArray::from_slice(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            2,
            1,
            4,
        );
        let mut cropper = RandomCrop::new(2, 2, 2);
        let out_image = cropper.process_batched(&src_image);
        assert_eq!(out_image.dim, 2);
        assert_eq!(out_image.channels, 1);
    }
}*/
