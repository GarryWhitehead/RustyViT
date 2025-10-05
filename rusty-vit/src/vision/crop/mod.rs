mod crop_cpu;
#[cfg(feature = "cuda")]
mod crop_cu;
#[cfg(feature = "vulkan")]
mod crop_vk;

use crate::device::DeviceStorage;
use crate::image::{Image, PixelType};
use rand::Rng;
use rand::distr::Uniform;

pub trait CropKernel<T>: DeviceStorage<T> {
    fn crop(
        &mut self,
        src: &mut Image<T, Self>,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> Image<T, Self>
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
    pub fn new(width: usize, height: usize, crop_width: usize, crop_height: usize) -> Self {
        if crop_width > width {
            panic!(
                "Crop width must be less than the src dimensions; crop_width: {crop_width} vs \
                image dim: {width}"
            );
        }
        if crop_height > height {
            panic!(
                "Crop height must be less than the src dimensions; crop_height: {crop_height} \
            vs image dim: {height}"
            );
        };

        let uniform_x = Uniform::try_from(0..width - crop_width + 1).unwrap();
        let uniform_y = Uniform::try_from(0..height - crop_height + 1).unwrap();
        let mut rng = rand::rng();
        Self {
            crop_width,
            crop_height,
            x: rng.sample(uniform_x),
            y: rng.sample(uniform_y),
        }
    }

    pub fn crop<P: PixelType, D: CropKernel<P>>(
        &self,
        src: &mut Image<P, D>,
        dev: &mut D,
    ) -> Image<P, D> {
        dev.crop(src, self.crop_width, self.crop_height, self.x, self.y)
    }
}

mod tests {
    use super::*;
    use crate::device::cpu::Cpu;
    #[cfg(feature = "cuda")]
    use crate::device::cuda::Cuda;
    #[cfg(feature = "vulkan")]
    use crate::device::vulkan::Vulkan;
    //use rusty_vk::public_types::DeviceType;

    #[test]
    fn test_crop() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
        //let dev = Cuda::try_new(0).unwrap();
        let mut dev = Cpu::default();
        //let mut dev = Vulkan::new(DeviceType::DiscreteGpu).unwrap();
        let cropper = RandomCrop::new(3, 3, 3, 3);
        let mut img: Image<u8, _> = Image::try_from_slice(src, 1, 3, 3, 1, &mut dev).unwrap();
        let dst = cropper.crop(&mut img, &mut dev);
        assert_eq!(dst.try_get_data().unwrap(), src);
    }

    #[test]
    fn test_crop_batched() {
        //let dev = Cuda::try_new(0).unwrap();
        let mut dev = Cpu::default();
        //let mut dev = Vulkan::new(DeviceType::DiscreteGpu).unwrap();
        let (b, c, w, h) = (20, 3, 3, 3);
        let template = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut src = vec![0u8; b * c * w * h];
        src.chunks_mut(w * h).for_each(|slice| {
            slice.copy_from_slice(template);
        });
        let cropper = RandomCrop::new(3, 3, 3, 3);
        let mut img: Image<u8, _> = Image::try_from_slice(&src, b, w, h, c, &mut dev).unwrap();
        let crop_img = cropper.crop(&mut img, &mut dev);
        let crop_img = crop_img.try_get_data().unwrap();
        crop_img.chunks(w * h).for_each(|slice| {
            assert_eq!(slice, template);
        })
    }
}
/*#[test]
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
