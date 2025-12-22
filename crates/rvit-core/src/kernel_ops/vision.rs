use crate::device::{DAlloc, Device};
use crate::element_traits::DataElem;
use crate::vision::border::BorderMode;
use crate::vision::interpolation::InterpMode;

pub trait MakeBorderKernel<T: DataElem, I: BorderMode>: Device {
    fn make_border(
        &mut self,
        src: &DAlloc<Self>,
        src_shape: &[usize],
        src_strides: &[usize],
        padding: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>);
}

pub trait ResizeKernel<T: DataElem, I: InterpMode>: Device {
    fn resize(
        &mut self,
        src: &DAlloc<Self>,
        in_shape: &[usize],
        in_strides: &[usize],
        dst_width: usize,
        dst_height: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>);
}

pub trait HorizontalFlipKernel<T: DataElem>: Device {
    fn flip_horizontal(
        &mut self,
        src: &mut DAlloc<Self>,
        src_shape: &[usize],
        src_strides: &[usize],
        prob: f32,
    );
}

pub trait CropKernel<T: DataElem>: Device {
    fn crop(
        &mut self,
        src: &mut DAlloc<Self>,
        src_shape: &[usize],
        src_strides: &[usize],
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> (DAlloc<Self>, Vec<usize>, Vec<usize>);
}

pub trait ConvKernel<T: DataElem>: Device {
    fn convolution(
        &mut self,
        src: &mut DAlloc<Self>,
        src_shape: &[usize],
        src_strides: &[usize],
        x_kernel: &DAlloc<Self>,
        y_kernel: &DAlloc<Self>,
    );
}
