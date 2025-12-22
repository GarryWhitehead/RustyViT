use crate::convolution::{Convolution, FloatConv};
use rvit_core::element_traits::DataElem;
use rvit_device::Device;
use rvit_device::vision::op_traits::ConvKernel;
use rvit_tensor::basic_ops::*;
use rvit_tensor::tensor::{Float, Integer, Tensor, TensorType};
use std::error::Error;
use std::marker::PhantomData;
/*
#[derive(Clone)]
pub struct GaussianBlur<T: TensorType, D: Device> {
    pub conv_kernel: Convolution<T, D>,
}

impl<D: Device> GaussianBlur<Float, D> {
    pub fn try_new(
        sigma: f32,
        kernel_dim: usize,
        dev: &mut D::Storage,
    ) -> Result<Self, Box<dyn Error>> {
        validate(sigma, kernel_dim);
        let x_kernel = compute_gaussian_kernel::<D::FloatElem, Float, _>(sigma, kernel_dim, dev);
        let y_kernel = compute_gaussian_kernel::<D::FloatElem, Float, _>(sigma, kernel_dim, dev);
        let conv = Convolution::new(x_kernel, y_kernel);
        Ok(Self { conv_kernel: conv })
    }

    pub fn process(&mut self, src: &mut Tensor<Float, D>) {
        self.conv_kernel
            .process(src, &self.conv_kernel.x_kernel, &self.conv_kernel.y_kernel);
    }
}

impl<D: Device> GaussianBlur<Integer, D> {
    pub fn try_new(
        sigma: f32,
        kernel_dim: usize,
        dev: &mut D::Storage,
    ) -> Result<Self, Box<dyn Error>> {
        validate(sigma, kernel_dim);
        let x_kernel = compute_gaussian_kernel::<D::IntElem, Integer, _>(sigma, kernel_dim, dev);
        let y_kernel = compute_gaussian_kernel::<D::IntElem, Integer, _>(sigma, kernel_dim, dev);
        let conv = Convolution::new(x_kernel, y_kernel);
        Ok(Self { conv_kernel: conv })
    }

    pub fn process(&mut self, src: &mut Tensor<Integer, D>) {
        self.conv_kernel.process(src);
    }
}

fn compute_gaussian_kernel<E: DataElem, T: TensorType, D: Device>(
    sigma: f32,
    kernel_dim: usize,
    dev: &mut D::Storage,
) -> Tensor<T, D> {
    let mut kernel = vec![E::zero(); kernel_dim];
    let s2 = -0.5 / (sigma * sigma);
    let sum: E = (0..kernel_dim)
        .map(|idx| {
            let x = idx as f32 - (kernel_dim - 1) as f32 / 2.0;
            let v = (s2 * x * x).exp();
            kernel[idx] = E::from(v).unwrap();
            kernel[idx].clone()
        })
        .sum();
    (0..kernel_dim).for_each(|i| kernel[i] /= sum);
    Tensor::from_array(&[kernel_dim], &kernel, dev)
}

fn validate(sigma: f32, kernel_dim: usize) {
    if sigma <= 0.0 {
        panic!("Sigma value must be non-zero.");
    }
    if kernel_dim == 0 {
        panic!("Kernel dimensions must be non-zero.");
    }
    if (kernel_dim & 1) == 0 {
        panic!("Kernel dimensions must be even.");
    }
}*/
