use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use rvit_device::vision_traits::HorizontalFlipKernel;
use rvit_image::image::Image;

#[derive(Debug, Clone)]
pub struct RandomFlipHorizontal {
    probability: f32,
}

impl RandomFlipHorizontal {
    pub fn new(probability: f32) -> Self {
        Self { probability }
    }
}

impl RandomFlipHorizontal {
    pub fn process<T: PixelType, D: HorizontalFlipKernel<T>>(&mut self, src: &mut Image<T, D>) {
        let shape = [src.batch_size, src.channels, src.width, src.height];
        D::flip_horizontal(
            &mut src.device,
            &mut src.data,
            &shape,
            &src.strides,
            self.probability,
        );
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

#[cfg(test)]
mod tests {
    use super::*;
    use rvit_device::tests::TestDevice;

    #[test]
    fn test_flip_horizontal() {
        let src: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
        let mut dev = TestDevice::default();
        let mut flipper = RandomFlipHorizontal::new(2.0);
        let mut img = Image::try_from_slice(&src, 1, 4, 4, 1, &dev).unwrap();
        flipper.process(&mut img);
        assert_eq!(
            img.try_get_data().unwrap(),
            &[4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]
        );
    }

    #[test]
    fn test_flip_horizontal_batch() {
        let (b, c, w, h) = (10usize, 3, 4, 4);
        let template: Vec<u8> = vec![1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
        let mut src: Vec<u8> = vec![0u8; b * c * w * h];
        src.chunks_mut(w * h)
            .for_each(|chunk| chunk.copy_from_slice(&template));

        let mut dev = TestDevice::default();
        let mut flipper = RandomFlipHorizontal::new(2.0);
        let mut img: Image<u8, _> = Image::try_from_slice(&src, b, w, h, c, &dev).unwrap();
        flipper.process(&mut img);
        let dst = img.try_get_data().unwrap();
        dst.chunks(w * h).for_each(|chunk| {
            assert_eq!(&[4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1], &chunk);
        })
    }

    /*#[test]
    #[cfg(not(feature = "cuda"))]
    fn test_flip_vertical() {
        let src: Vec<u8> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let mut dst = vec![0u8; src.len()];
        flip_vertical(&src, 4, 4, &mut dst);
        assert_eq!(&dst, &[4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1]);
    }*/
}
