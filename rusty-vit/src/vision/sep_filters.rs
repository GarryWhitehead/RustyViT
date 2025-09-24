use crate::device::DeviceStorage;
use crate::image::{Image, PixelType};
use crate::type_traits::FloatType;
use crate::vision::convolution::{Conv, Convolution};
use std::error::Error;
use std::iter::Sum;
use std::ops::{BitAnd, DivAssign};

#[derive(Debug, Clone)]
pub struct GaussianBlur<F: FloatType, T: PixelType, D: DeviceStorage<F> + Conv<T, F>> {
    conv_kernel: Convolution<F, T, D>,
}

impl<F: FloatType + Sum + DivAssign, T: PixelType, D: DeviceStorage<F> + Conv<T, F>>
    GaussianBlur<F, T, D>
{
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

        let mut kernel = vec![F::zero(); kernel_dim];
        let s2 = -0.5 / (sigma * sigma);
        let sum: F = (0..kernel_dim)
            .map(|idx| {
                let x = F::from(idx as f32 - (kernel_dim - 1) as f32 / 2.0).unwrap();
                let v = (F::from(s2).unwrap() * x * x).exp();
                kernel[idx] = v;
                v
            })
            .sum();
        (0..kernel_dim).for_each(|i| kernel[i] /= sum);
        println!("KERNEL: {:?}", kernel);
        let conv = Convolution::try_new(&kernel, &kernel, dev)?;
        Ok(Self { conv_kernel: conv })
    }

    pub fn process(&self, src: &mut Image<T, D>) {
        self.conv_kernel.process(src);
    }
}

mod tests {
    use super::*;
    use crate::device::cpu::Cpu;
    #[cfg(feature = "cuda")]
    use crate::device::cuda::Cuda;

    #[test]
    fn test_blur() {
        let dev = Cpu::default();
        //let dev = Cuda::try_new(0).unwrap();
        let data = &[3, 5, 2, 1, 0, 0, 3, 1, 6, 9, 10, 0, 5, 3, 2, 1];
        let blur = GaussianBlur::<f32, u8, _>::try_new(2.0, 3, &dev).unwrap();
        let mut img: Image<u8, _> = Image::try_from_slice(&data, 1, 16, 16, 1, &dev).unwrap();
        blur.process(&mut img);
        assert_eq!(
            img.try_get_data().unwrap(),
            &[2, 2, 2, 1, 2, 3, 2, 1, 3, 4, 3, 1, 4, 4, 2, 1]
        );
    }
}
