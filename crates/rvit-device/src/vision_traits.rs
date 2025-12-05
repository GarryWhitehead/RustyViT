use rvit_core::pixel_traits::*;
use rvit_core::storage::DeviceStorage;

pub trait MakeBorderKernel<T: PixelType, I: BorderMode>: DeviceStorage<T> {
    fn make_border(
        &mut self,
        src: &Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        padding: usize,
    ) -> (Self::Vec, Vec<usize>, Vec<usize>);
}

pub trait ResizeKernel<T: PixelType, I: InterpMode>: DeviceStorage<T> {
    fn resize(
        &mut self,
        src: &Self::Vec,
        in_shape: &[usize],
        in_strides: &[usize],
        dst_width: usize,
        dst_height: usize,
    ) -> (Self::Vec, Vec<usize>, Vec<usize>);
}

pub trait HorizontalFlipKernel<T: PixelType>: DeviceStorage<T> {
    fn flip_horizontal(
        &mut self,
        src: &mut Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        prob: f32,
    );
}

pub trait CropKernel<T: PixelType>: DeviceStorage<T> {
    fn crop(
        &mut self,
        src: &mut Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> (Self::Vec, Vec<usize>, Vec<usize>);
}

pub trait ConvKernel<T: PixelType>: DeviceStorage<T> {
    fn convolution(
        &mut self,
        src: &mut Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        x_kernel: &Self::Vec,
        y_kernel: &Self::Vec,
    );
}
