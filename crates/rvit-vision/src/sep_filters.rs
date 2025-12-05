use crate::convolution::Convolution;
use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use rvit_device::vision_traits::ConvKernel;
use rvit_image::image::Image;
use std::error::Error;
use std::iter::Sum;
use std::ops::DivAssign;

#[derive(Debug, Clone)]
pub struct GaussianBlur<T: PixelType, D: DeviceStorage<T> + ConvKernel<T>> {
    conv_kernel: Convolution<T, D>,
}

impl<T: PixelType + Sum + DivAssign, D: DeviceStorage<T> + ConvKernel<T>> GaussianBlur<T, D> {
    pub fn try_new(sigma: f32, kernel_dim: usize, dev: &D) -> Result<Self, Box<dyn Error>> {
        if sigma <= 0.0 {
            panic!("Sigma value must be non-zero.");
        }
        if kernel_dim == 0 {
            panic!("Kernel dimensions must be non-zero.");
        }
        if (kernel_dim & 1) == 0 {
            panic!("Kernel dimensions must be even.");
        }

        let mut kernel = vec![T::zero(); kernel_dim];
        let s2 = -0.5 / (sigma * sigma);
        let sum: T = (0..kernel_dim)
            .map(|idx| {
                let x = idx as f32 - (kernel_dim - 1) as f32 / 2.0;
                let v = (s2 * x * x).exp();
                kernel[idx] = T::from(v).unwrap();
                kernel[idx]
            })
            .sum();
        (0..kernel_dim).for_each(|i| kernel[i] /= sum);

        let conv = Convolution::try_new(&kernel, &kernel, dev)?;
        Ok(Self { conv_kernel: conv })
    }

    pub fn process(&mut self, src: &mut Image<T, D>) {
        self.conv_kernel.process(src);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvit_device::tests::TestDevice;

    #[test]
    fn test_blur() {
        let mut dev = TestDevice::default();
        let data: Vec<u8> = vec![3u8; 32 * 32];
        let mut blur = GaussianBlur::<u8, _>::try_new(2.0, 3, &mut dev).unwrap();
        let mut img = Image::try_from_slice(&data, 1, 32, 32, 1, &mut dev).unwrap();
        blur.process(&mut img);
        assert_eq!(
            img.try_get_data().unwrap(),
            &[2, 2, 2, 1, 2, 3, 2, 1, 3, 4, 3, 1, 4, 4, 2, 1]
        );
    }
}
