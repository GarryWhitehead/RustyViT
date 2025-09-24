use crate::device::DeviceStorage;
use crate::image::{Image, PixelType};
use crate::type_traits::FloatType;
use std::error::Error;
use std::marker::PhantomData;

mod conv_cpu;
#[cfg(feature = "cuda")]
mod conv_cu;

pub trait Conv<T: PixelType, F: FloatType>: DeviceStorage<T> + DeviceStorage<F> {
    fn convolution(
        &mut self,
        src: &mut Image<T, Self>,
        x_kernel: &Kernel<F, Self>,
        y_kernel: &Kernel<F, Self>,
    ) where
        Self: Sized;
}

#[derive(Debug, Clone)]
struct Kernel<F: FloatType, D: DeviceStorage<F>> {
    data: D::Vec,
    device: D,
}

impl<F: FloatType, D: DeviceStorage<F>> Kernel<F, D> {
    fn try_new(data: &[F], dev: &D) -> Result<Self, Box<dyn Error>> {
        let k = dev.try_alloc_with_slice(data)?;
        Ok(Self {
            data: k,
            device: dev.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Convolution<F: FloatType, T: PixelType, D: DeviceStorage<F> + Conv<T, F>> {
    x_kernel: Kernel<F, D>,
    y_kernel: Kernel<F, D>,
    phantom_data: PhantomData<T>,
}

impl<F: FloatType, T: PixelType, D: DeviceStorage<F> + Conv<T, F>> Convolution<F, T, D> {
    pub fn try_new(x_kernel: &[F], y_kernel: &[F], dev: &D) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            x_kernel: Kernel::try_new(x_kernel, dev)?,
            y_kernel: Kernel::try_new(y_kernel, dev)?,
            phantom_data: Default::default(),
        })
    }
}

impl<F: FloatType, T: PixelType, D: Conv<T, F>> Convolution<F, T, D> {
    pub fn process(&self, src: &mut Image<T, D>) {
        if <D as DeviceStorage<F>>::len(&self.x_kernel.data) >= src.width {
            panic!("kernel width cannot be greater than the kernel width");
        }
        if <D as DeviceStorage<F>>::len(&self.x_kernel.data) >= src.height {
            panic!("kernel height cannot be greater than the kernel height");
        }
        let dev = &mut src.device.clone();
        dev.convolution(src, &self.x_kernel, &self.y_kernel);
    }
}
