use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use rvit_device::vision_traits::ConvKernel;
use rvit_image::image::Image;
use std::error::Error;
use std::marker::PhantomData;

#[derive(Debug, Clone)]
pub struct Kernel<T: PixelType, D: DeviceStorage<T>> {
    pub data: D::Vec,
    pub device: D,
}

impl<T: PixelType, D: DeviceStorage<T>> Kernel<T, D> {
    fn try_new(data: &[T], dev: &D) -> Result<Self, Box<dyn Error>> {
        let k = dev.try_alloc_with_slice(data)?;
        Ok(Self {
            data: k,
            device: dev.clone(),
        })
    }
}

#[derive(Debug, Clone)]
pub struct Convolution<T: PixelType, D: DeviceStorage<T> + ConvKernel<T>> {
    x_kernel: Kernel<T, D>,
    y_kernel: Kernel<T, D>,
    phantom_data: PhantomData<T>,
}

impl<T: PixelType, D: DeviceStorage<T> + ConvKernel<T>> Convolution<T, D> {
    pub fn try_new(x_kernel: &[T], y_kernel: &[T], dev: &D) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            x_kernel: Kernel::try_new(x_kernel, dev)?,
            y_kernel: Kernel::try_new(y_kernel, dev)?,
            phantom_data: Default::default(),
        })
    }
}

impl<T: PixelType, D: ConvKernel<T>> Convolution<T, D> {
    pub fn process(&mut self, src: &mut Image<T, D>) {
        if <D as DeviceStorage<T>>::len(&self.x_kernel.data) >= src.width {
            panic!("kernel width cannot be greater than the kernel width");
        }
        if <D as DeviceStorage<T>>::len(&self.x_kernel.data) >= src.height {
            panic!("kernel height cannot be greater than the kernel height");
        }

        let shape = [src.batch_size, src.channels, src.width, src.height];
        D::convolution(
            &mut src.device,
            &mut src.data,
            &shape,
            &src.strides,
            &self.x_kernel.data,
            &self.y_kernel.data,
        );
    }
}
