use crate::basic_ops::BasicOps;
use crate::distribution::DistributionMethod;
use crate::tensor::{Float, Integer, Tensor, TensorType};
use crate::{Device, distribution};
use rand::distr::Distribution;
use rand::distr::uniform::SampleUniform;
use rand_distr::StandardNormal;
use rvit_core::element_traits::DataElem;
use rvit_core::memory::storage::DeviceStorage;
use rvit_core::tensor::compute_strides;
use rvit_device::Device;
use rvit_device::tensor::op_traits::{Conv2dKernel, ConvConvertKernel};
use std::error::Error;
use std::marker::PhantomData;

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum ConvShape {
    Nchw,
    Nhwc,
}

#[derive(Copy, Clone, Debug)]
pub struct Conv2d<T: TensorType, D: Device> {
    pub out_channels: usize,
    pub groups: usize,
    pub stride: usize,
    pub padding: usize,
    pub kernel_size: usize,
    pub conv_shape: ConvShape,
    pub dist_method: Option<DistributionMethod>,
    pub _ty: PhantomData<T>,
    pub _device: PhantomData<D>,
}

pub struct ConvInput<T: TensorType, D: Device> {
    pub out_width: usize,
    pub out_height: usize,
    pub groups: usize,
    pub stride: usize,
    pub padding: usize,
    pub filters: Tensor<T, D>,
    pub batch_size: usize,
    pub in_channels: usize,
    pub in_width: usize,
    pub in_height: usize,
    pub conv_shape: ConvShape,
}

impl<T: TensorType, D: Device> Default for Conv2d<T, D> {
    fn default() -> Self {
        Self {
            out_channels: 1,
            groups: 1,
            stride: 1,
            padding: 0,
            kernel_size: 2,
            conv_shape: ConvShape::Nchw,
            dist_method: None,
            _ty: PhantomData,
            _device: PhantomData,
        }
    }
}

impl<D: Device> Conv2d<Float, D>
where
    StandardNormal: Distribution<D::FloatElem>,
    D::FloatElem: SampleUniform,
{
    pub fn init(
        &mut self,
        x: &Tensor<Float, D>,
        dev: &mut D::Storage,
    ) -> Result<ConvInput<Float, D>, Box<dyn Error>> {
        let (batch_size, in_channels, in_height, in_width) =
            (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        Self::validate(in_channels, self.out_channels, self.stride, self.groups);

        let out_height =
            Self::compute_out_size(in_height, self.padding, self.kernel_size, self.stride);
        let out_width =
            Self::compute_out_size(in_width, self.padding, self.kernel_size, self.stride);

        let f_shape = [
            self.out_channels,
            in_channels / self.groups,
            self.kernel_size,
            self.kernel_size,
        ];
        let f_buffer = Self::compute_weights::<D::FloatElem>(self, in_channels);
        let filters = Tensor::<Float, _>::from_array(&f_shape, &f_buffer, dev);

        Ok(ConvInput {
            out_width,
            out_height,
            groups: self.groups,
            stride: self.stride,
            padding: self.padding,
            filters,
            batch_size,
            in_channels,
            in_width,
            in_height,
            conv_shape: self.conv_shape,
        })
    }
}

impl<D: Device> Conv2d<Integer, D>
where
    StandardNormal: Distribution<D::IntElem>,
    D::IntElem: SampleUniform,
{
    pub fn init(
        &mut self,
        x: &Tensor<Integer, D>,
        dev: &mut D::Storage,
    ) -> Result<ConvInput<Integer, D>, Box<dyn Error>> {
        let (batch_size, in_channels, in_height, in_width) =
            (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
        Self::validate(in_channels, self.out_channels, self.stride, self.groups);

        let out_height =
            Self::compute_out_size(in_height, self.padding, self.kernel_size, self.stride);
        let out_width =
            Self::compute_out_size(in_width, self.padding, self.kernel_size, self.stride);

        let f_shape = [
            self.out_channels,
            in_channels / self.groups,
            self.kernel_size,
            self.kernel_size,
        ];
        let f_buffer = Self::compute_weights::<D::IntElem>(self, in_channels);
        let filters = Tensor::<Integer, _>::from_array(&f_shape, &f_buffer, dev);

        Ok(ConvInput {
            out_width,
            out_height,
            groups: self.groups,
            stride: self.stride,
            padding: self.padding,
            filters,
            batch_size,
            in_channels,
            in_width,
            in_height,
            conv_shape: self.conv_shape,
        })
    }
}

impl<T: TensorType, D: Device> Conv2d<T, D> {
    fn compute_out_size(dim: usize, padding: usize, kernel_size: usize, stride: usize) -> usize {
        (dim + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    }

    fn validate(in_channels: usize, out_channels: usize, stride: usize, groups: usize) {
        if stride < 1 {
            panic!("The stride must be greater than zero");
        }
        if !in_channels.is_multiple_of(groups) || !out_channels.is_multiple_of(groups) {
            panic!("The group size must be a multiple of the in/out channel size");
        }
    }

    fn compute_weights<E: DataElem + SampleUniform>(
        conv: &Conv2d<T, D>,
        in_channels: usize,
    ) -> Vec<E> {
        let fan_in = conv.kernel_size * conv.kernel_size * in_channels;
        let fan_out = conv.kernel_size * conv.kernel_size * conv.out_channels;
        let f_size =
            conv.out_channels * (in_channels / conv.groups) * conv.kernel_size * conv.kernel_size;
        Self::distribute_weights::<E>(f_size, fan_in, fan_out, conv.dist_method)
    }

    fn distribute_weights<E: DataElem + SampleUniform>(
        count: usize,
        fan_in: usize,
        fan_out: usize,
        method: Option<DistributionMethod>,
    ) -> Vec<E> {
        let mut out = vec![E::ONE; count];
        match method {
            Some(m) => {
                distribution::sample(fan_in, fan_out, &mut out, m);
                out
            }
            None => out,
        }
    }
}

impl<S: Device> ConvInput<Float, S> {
    pub fn update_weights(mut self, weights: &[S::FloatElem]) -> ConvInput<Float, S> {
        self.filters =
            Tensor::<Float, _>::from_array(&self.filters.shape, weights, &mut self.filters.device);
        self
    }
}

impl<S: Device> ConvInput<Integer, S> {
    pub fn update_weights(mut self, weights: &[S::IntElem]) -> ConvInput<Integer, S> {
        self.filters = Tensor::<Integer, _>::from_array(
            &self.filters.shape,
            weights,
            &mut self.filters.device,
        );
        self
    }
}

impl<T: TensorType, D: Device<Storage = D>> Tensor<T, D>
where
    D: Conv2dKernel<T>,
    D: DeviceStorage,
{
    pub fn conv2d(&mut self, p: &ConvInput<T, D>) -> Tensor<T, D> {
        let out_shape = [p.batch_size, p.filters.shape[0], p.out_height, p.out_width];
        let out_strides = compute_strides(&out_shape);
        let to_nhwc = match p.conv_shape {
            ConvShape::Nchw => false,
            ConvShape::Nhwc => true,
        };
        let data = D::conv2d_fwd(
            &mut self.device,
            &self.data,
            &self.shape,
            &self.strides,
            &p.filters.data,
            &p.filters.shape,
            &out_shape,
            p.stride,
            p.padding,
            p.groups,
            to_nhwc,
        );
        Tensor::from_parts(data, &out_shape, &out_strides, &self.device)
    }
}

impl<T: TensorType, D: Device<Storage = D>> Tensor<T, D>
where
    D: ConvConvertKernel<T>,
    D: DeviceStorage,
{
    pub fn im2col(&mut self, p: &ConvInput<T, D>) -> Tensor<T, D> {
        let to_nhwc = match p.conv_shape {
            ConvShape::Nchw => false,
            ConvShape::Nhwc => true,
        };
        let (data, out_shape, out_strides) = D::im2col(
            &mut self.device,
            &self.data,
            &self.shape,
            &p.filters.shape,
            p.batch_size,
            p.out_width,
            p.out_height,
            p.stride,
            p.padding,
            to_nhwc,
        );
        Tensor::from_parts(data, &out_shape, &out_strides, &self.device)
    }

    pub fn nchw_to_nhwc(&mut self) -> Tensor<T, D> {
        let data = D::nchw_to_nhwc(&mut self.device, &self.data, &self.shape);
        let new_shape = [self.shape[0], self.shape[2], self.shape[3], self.shape[1]];
        let new_strides = compute_strides(&new_shape);
        Tensor::from_parts(data, &new_shape, &new_strides, &self.device)
    }

    pub fn nhwc_to_nchw(&mut self) -> Tensor<T, D> {
        let data = D::nhwc_to_nchw(&mut self.device, &self.data, &self.shape);
        let new_shape = [self.shape[0], self.shape[2], self.shape[3], self.shape[1]];
        let new_strides = compute_strides(&new_shape);
        Tensor::from_parts(data, &new_shape, &new_strides, &self.device)
    }
}

pub fn permute_nchw_to_nhwc<T: TensorType, D: Device>(
    mut tensor: Tensor<T, D>,
) -> Tensor<T, D>
where
    T: DataElem,
{
    let indices = [0, 2, 3, 1];
    tensor.permute(&indices);
    tensor
}

pub fn permute_nhwc_to_nchw<T: TensorType, D: Device>(
    mut tensor: Tensor<T, D>,
) -> Tensor<T, D>
where
    T: DataElem,
{
    let indices = [0, 3, 1, 2];
    tensor.permute(&indices);
    tensor
}
