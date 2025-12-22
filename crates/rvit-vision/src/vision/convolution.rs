use rvit_core::memory::storage::DeviceStorage;
use rvit_device::vision::op_traits::ConvKernel;
use rvit_device::{DAlloc, Device};
use rvit_tensor::tensor::{Float, Integer, Tensor, TensorType};

#[derive(Clone)]
pub struct Convolution<T: TensorType, D: Device> {
    pub x_kernel: Tensor<T, D>,
    pub y_kernel: Tensor<T, D>,
}

impl<T: TensorType, D: Device> Convolution<T, D> {
    pub fn new(x_kernel: Tensor<T, D>, y_kernel: Tensor<T, D>) -> Self {
        Self { x_kernel, y_kernel }
    }
}

pub trait FloatConv<D: Device>: ConvKernel<D::FloatElem> {
    fn float_conv(
        &mut self,
        src: &mut Tensor<Float, Self>,
        x_kernel: &Tensor<Float, Self>,
        y_kernel: &Tensor<Float, Self>,
    ) {
        self.convolution(
            &mut src.data,
            &src.shape,
            &src.strides,
            &x_kernel.data,
            &y_kernel.data,
        );
    }
}

pub trait IntConv<D: Device>: ConvKernel<D::IntElem> {
    fn int_conv(
        &mut self,
        src: &mut Tensor<Integer, Self>,
        x_kernel: &Tensor<Integer, Self>,
        y_kernel: &Tensor<Integer, Self>,
    ) {
        self.convolution(
            &mut src.data,
            &src.shape,
            &src.strides,
            &x_kernel.data,
            &y_kernel.data,
        );
    }
}

impl<D: Device> FloatConv<D> for Convolution<Float, D> where Self: ConvKernel<D::FloatElem> {}

impl<D: Device> Convolution<Float, D>
where
    Self: FloatConv<D>,
    Self: ConvKernel<D::FloatElem>,
{
    pub fn process(
        &mut self,
        src: &mut Tensor<Float, Self>,
        x_k: &Tensor<Float, Self>,
        y_k: &Tensor<Float, Self>,
    ) {
        let (height, width) = (src.shape[2], src.shape[3]);
        validate(width, height, &self.x_kernel, &self.y_kernel);
        self.float_conv(src, x_k, y_k);
    }
}

impl<D: Device> Convolution<Integer, D>
where
    Self: IntConv<D>,
{
    pub fn process(
        &mut self,
        src: &mut Tensor<Integer, Self>,
        x_k: &Tensor<Integer, Self>,
        y_k: &Tensor<Integer, Self>,
    ) {
        let (height, width) = (src.shape[2], src.shape[3]);
        validate(width, height, &self.x_kernel, &self.y_kernel);
        self.int_conv(src, x_k, y_k);
    }
}

fn validate<T: TensorType, D: Device>(
    width: usize,
    height: usize,
    x_kernel: &Tensor<T, D>,
    y_kernel: &Tensor<T, D>,
) {
    if D::Storage::len(&x_kernel.data) >= width {
        panic!("kernel width cannot be greater than the image width");
    }
    if D::Storage::len(&y_kernel.data) >= height {
        panic!("kernel height cannot be greater than the image height");
    }
}
