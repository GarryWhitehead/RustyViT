use crate::device::DeviceStorage;
use crate::image::{Image, PixelType};

#[cfg(feature = "cuda")]
mod conv_cu;
mod conv_cpu;

pub trait ConvKernel<T>: DeviceStorage<T> {
    fn convolution(&self, src: &mut Image<T, Self>, kernel: &Kernel)
    where
        Self: Sized;
}

#[derive(Debug, Clone)]
pub struct Kernel {
    width: usize,
    height: usize,
    data: Vec<f32>,
}

impl Kernel {
    pub fn new(width: usize, height: usize, data: &[f32]) -> Self {
        if width == 0 || height == 0 {
            panic!("Kernel width and height must be non-zero");
        }
        if data.len() != width * height {
            panic!("Kernel data size does not equal the kernel dimensions");
        }

        Self {
            width,
            height,
            data: data.to_vec(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Convolution {
    kernel: Kernel
}

impl Convolution {
    pub fn new(
        kernel: &Kernel,
    ) -> Self {
        Self {
            kernel: kernel.clone(),
        }
    }
}

impl Convolution {
    pub fn convolution<T: PixelType, S: ConvKernel<T>>(self, src: &mut Image<T, S>) {
        if self.kernel.width >= src.width {
            panic!("kernel width cannot be greater than the kernel width");
        }
        if self.kernel.height >= src.height {
            panic!("kernel height cannot be greater than the kernel height");
        }
        let dev = &src.device.clone();
        dev.convolution(src, &self.kernel);
    }
}

/*mod tests {
    use super::*;
    #[test]
    fn test_convolution_no_pad() {
        let src = &[
            3.0, 3.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0, 3.0, 1.0, 3.0, 1.0, 2.0, 2.0, 3.0, 2.0, 0.0,
            0.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0,
        ];
        let mut dst = vec![0.0f32; 3 * 3];
        let kernel = &[0.0, 1.0, 2.0, 2.0, 2.0, 0.0, 0.0, 1.0, 2.0];
        let conv = Convolution::new(1, 1, 1, 1, 0, 3, 5, 0.0);
        conv.convolution(src, 5, kernel, &mut dst);
        assert_eq!(dst, [12.0, 12.0, 17.0, 10.0, 17.0, 19.0, 9.0, 6.0, 14.0]);
    }

    #[test]
    fn test_convolution_one_pad() {
        let src = &[
            2.0, 2.0, 3.0, 3.0, 3.0, 0.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 1.0, 3.0, 3.0, 3.0,
            2.0, 1.0, 2.0, 3.0, 3.0, 0.0, 2.0, 3.0,
        ];
        let mut dst = vec![0.0f32; 5 * 5];
        let kernel = &[2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let conv = Convolution::new(1, 1, 1, 1, 1, 3, 5, 0.0);
        conv.convolution(src, 5, kernel, &mut dst);
        assert_eq!(
            dst,
            [
                1.0, 6.0, 5.0, 6.0, 6.0, 7.0, 10.0, 9.0, 16.0, 9.0, 7.0, 10.0, 8.0, 12.0, 3.0, 9.0,
                10.0, 12.0, 10.0, 6.0, 3.0, 11.0, 10.0, 6.0, 4.0
            ]
        );
    }

    #[test]
    fn test_convolution_stride_two() {
        let src = &[
            2.0, 2.0, 3.0, 3.0, 3.0, 0.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 1.0, 3.0, 3.0, 3.0,
            2.0, 1.0, 2.0, 3.0, 3.0, 0.0, 2.0, 3.0,
        ];
        let mut dst = vec![0.0f32; 3 * 3];
        let kernel = &[2.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let conv = Convolution::new(1, 1, 1, 2, 1, 3, 5, 0.0);
        conv.convolution(src, 5, kernel, &mut dst);
        assert_eq!(dst, [1.0, 5.0, 6.0, 7.0, 8.0, 3.0, 3.0, 10.0, 4.0]);
    }

    #[test]
    fn test_convolution_process() {
        let src = &[
            2.0, 2.0, 3.0, 3.0, 3.0, 0.0, 1.0, 3.0, 0.0, 3.0, 2.0, 3.0, 0.0, 1.0, 3.0, 3.0, 3.0,
            2.0, 1.0, 2.0, 3.0, 3.0, 0.0, 2.0, 3.0,
        ];
        let matrix = Matrix::from_slice_copy(src, &[1, 1, 5, 5]);

        let conv = Convolution::new(1, 1, 1, 1, 0, 3, 5, 0.0);
        let conv_mat = conv.process(&matrix);
        assert_eq!(conv_mat.shape, &[1, 1, 3, 3]);
    }
}*/