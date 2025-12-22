use rand::Rng;
use rand::distr::Uniform;
use rvit_device::vision::op_traits::CropKernel;
use rvit_device::{DAlloc, Device};
use rvit_tensor::tensor::{Float, Integer, Tensor, TensorType};
use std::marker::PhantomData;

pub struct RandomCrop<T: TensorType, D: Device> {
    crop_width: usize,
    crop_height: usize,
    x: usize,
    y: usize,
    _type: PhantomData<T>,
    _device: PhantomData<D>,
}

impl<T: TensorType, D: Device> RandomCrop<T, D> {
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
            _type: PhantomData::default(),
            _device: PhantomData::default(),
        }
    }
}

trait FloatCrop<D: Device>: CropKernel<D::FloatElem> {
    fn float_crop(
        &mut self,
        src: &mut Tensor<Float, Self>,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>) {
        self.crop(
            &mut src.data,
            &src.shape,
            &src.strides,
            crop_width,
            crop_height,
            x,
            y,
        )
    }
}

trait IntCrop<D: Device>: CropKernel<D::IntElem> {
    fn int_crop(
        &mut self,
        src: &mut Tensor<Integer, Self>,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>) {
        self.crop(
            &mut src.data,
            &src.shape,
            &src.strides,
            crop_width,
            crop_height,
            x,
            y,
        )
    }
}

impl<D: Device> RandomCrop<Float, D>
where
    Self: FloatCrop<D>,
{
    pub fn process(&mut self, src: &mut Tensor<Float, Self>) -> Tensor<Float, Self> {
        let (data, out_shape, out_strides) =
            self.float_crop(src, self.crop_width, self.crop_height, self.x, self.y);
        Tensor::from_parts(data, &out_shape, &out_strides, &src.device)
    }
}
