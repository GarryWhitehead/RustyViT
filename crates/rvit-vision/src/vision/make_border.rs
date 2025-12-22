use rvit_core::element_traits::Elem;
use rvit_core::vision::border::*;
use rvit_device::vision::op_traits::MakeBorderKernel;
use rvit_device::{DAlloc, Device};
use rvit_tensor::tensor::{Float, Integer, Tensor, TensorType};
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct MakeBorder<T: TensorType, D: Device> {
    padding: usize,
    _type: PhantomData<T>,
    _device: PhantomData<D>,
}

impl<T: TensorType, D: Device> MakeBorder<T, D> {
    pub fn new(padding: usize) -> Self {
        Self {
            padding,
            _type: PhantomData,
            _device: PhantomData,
        }
    }
}

trait FloatMakeBorder<D: Device, B: BorderMode>: MakeBorderKernel<D::FloatElem, B> {
    fn float_make_border(
        &mut self,
        src: Tensor<Float, Self>,
        padding: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>) {
        self.make_border(&src.data, &src.shape, &src.strides, padding)
    }
}

trait IntMakeBorder<D: Device, B: BorderMode>: MakeBorderKernel<D::IntElem, B> {
    fn int_make_border(
        &mut self,
        src: Tensor<Integer, Self>,
        padding: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>) {
        self.make_border(&src.data, &src.shape, &src.strides, padding)
    }
}

impl<D: Device> MakeBorder<Float, D> {
    pub fn process<B: BorderMode>(&mut self, src: &mut Tensor<Float, Self>) -> Tensor<Float, Self>
    where
        Self: FloatMakeBorder<D, B>,
    {
        validate(&src.shape, self.padding);
        let (data, out_shape, out_strides) =
            self.make_border(&src.data, &src.shape, &src.strides, self.padding);
        Tensor::from_parts(data, &out_shape, &out_strides, &src.device)
    }
}

impl<D: Device> MakeBorder<Integer, D> {
    pub fn process<B: BorderMode>(
        &mut self,
        src: &mut Tensor<Integer, Self>,
    ) -> Tensor<Integer, Self>
    where
        Self: IntMakeBorder<D, B>,
    {
        validate(&src.shape, self.padding);
        let (data, out_shape, out_strides) =
            self.make_border(&src.data, &src.shape, &src.strides, self.padding);
        Tensor::from_parts(data, &out_shape, &out_strides, &src.device)
    }
}

fn validate(shape: &[usize], padding: usize) {
    let (height, width) = (shape[2], shape[3]);
    if 2 * padding >= width || 2 * padding >= height {
        panic!("Padding size must be less than the image dimensions");
    }
}
