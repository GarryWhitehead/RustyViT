use rvit_core::element_traits::Elem;
use rvit_core::vision::interpolation::InterpMode;
use rvit_device::vision::op_traits::ResizeKernel;
use rvit_device::{DAlloc, Device};
use rvit_tensor::tensor::{Float, Integer, Tensor, TensorType};
use std::marker::PhantomData;

pub struct Resize<T: TensorType, D: Device> {
    dst_width: usize,
    dst_height: usize,
    _type: PhantomData<T>,
    _device: PhantomData<D>,
}

impl<T: TensorType, D: Device> Resize<T, D> {
    pub fn new(dst_width: usize, dst_height: usize) -> Self {
        if dst_width == 0 || dst_height == 0 {
            panic!("Resized image dimensions must be non-zero");
        }
        Self {
            dst_width,
            dst_height,
            _type: PhantomData::default(),
            _device: PhantomData::default(),
        }
    }
}

trait FloatResize<D: Device, I: InterpMode>: ResizeKernel<D::FloatElem, I> {
    fn float_resize(
        &mut self,
        src: &Tensor<Float, Self>,
        dst_width: usize,
        dst_height: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>) {
        self.resize(&src.data, &src.shape, &src.strides, dst_width, dst_height)
    }
}

trait IntResize<D: Device, I: InterpMode>: ResizeKernel<D::IntElem, I> {
    fn int_resize(
        &mut self,
        src: &Tensor<Integer, Self>,
        dst_width: usize,
        dst_height: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>) {
        self.resize(&src.data, &src.shape, &src.strides, dst_width, dst_height)
    }
}

impl<D: Device> Resize<Float, D> {
    pub fn process<I: InterpMode>(&mut self, src: &mut Tensor<Float, Self>) -> Tensor<Float, Self>
    where
        Self: FloatResize<D, I>,
    {
        let (data, out_shape, out_strides) =
            self.float_resize(src, self.dst_width, self.dst_height);
        Tensor::from_parts(data, &out_shape, &out_strides, &src.device)
    }
}

impl<D: Device> Resize<Integer, D> {
    pub fn process<I: InterpMode>(
        &mut self,
        src: &mut Tensor<Integer, Self>,
    ) -> Tensor<Integer, Self>
    where
        Self: IntResize<D, I>,
    {
        let (data, out_shape, out_strides) = self.int_resize(src, self.dst_width, self.dst_height);
        Tensor::from_parts(data, &out_shape, &out_strides, &src.device)
    }
}
