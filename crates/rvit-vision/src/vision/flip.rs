use rvit_device::Device;
use rvit_device::vision::op_traits::HorizontalFlipKernel;
use rvit_tensor::tensor::{Float, Integer, Tensor, TensorType};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct RandomFlipHorizontal<T: TensorType, D: Device> {
    probability: f32,
    _type: PhantomData<T>,
    _device: PhantomData<D>,
}

impl<T: TensorType, D: Device> RandomFlipHorizontal<T, D> {
    pub fn new(probability: f32) -> Self {
        Self {
            probability,
            _type: PhantomData,
            _device: PhantomData,
        }
    }
}

trait FloatFlipHorizontal<D: Device>: HorizontalFlipKernel<D::FloatElem> {
    fn float_flip(&mut self, src: &mut Tensor<Float, Self>, probability: f32) {
        self.flip_horizontal(&mut src.data, &src.shape, &src.strides, probability);
    }
}

trait IntFlipHorizontal<D: Device>: HorizontalFlipKernel<D::IntElem> {
    fn int_flip(&mut self, src: &mut Tensor<Integer, Self>, probability: f32) {
        self.flip_horizontal(&mut src.data, &src.shape, &src.strides, probability);
    }
}

impl<D: Device> RandomFlipHorizontal<Float, D>
where
    Self: FloatFlipHorizontal<D>,
{
    pub fn process(&mut self, src: &mut Tensor<Float, Self>) {
        self.flip_horizontal(&mut src.data, &src.shape, &src.strides, self.probability);
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
