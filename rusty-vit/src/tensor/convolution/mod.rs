use crate::device::DeviceStorage;
use crate::tensor::Tensor;
use crate::tensor::distribution;
use crate::tensor::distribution::DistributionMethod;
use crate::type_traits::FloatType;
use rand::distr::Distribution;
use rand::distr::uniform::SampleUniform;
use rand_distr::StandardNormal;
use std::error::Error;
use std::marker::PhantomData;

mod conv_cpu;
#[cfg(feature = "cuda")]
mod conv_cu;

pub trait ConvKernel<T: FloatType>: DeviceStorage<T> {
    fn conv2d_fwd(
        &mut self,
        params: &ConvInput<T, Self>,
        tensor: Tensor<T, Self>,
    ) -> Tensor<T, Self>;
    fn im2col(&mut self, p: &ConvInput<T, Self>, tensor: &Tensor<T, Self>) -> Tensor<T, Self>;
    fn nchw_to_nhwc(&mut self, tensor: &Tensor<T, Self>) -> Tensor<T, Self>;
    fn nhwc_to_nchw(&mut self, tensor: &Tensor<T, Self>) -> Tensor<T, Self>;
}

#[derive(PartialEq, Clone, Copy, Debug)]
pub enum ConvShape {
    Nchw,
    Nhwc,
}

#[derive(Copy, Clone, Debug)]
pub struct Conv2d<T: FloatType> {
    pub out_channels: usize,
    pub groups: usize,
    pub stride: usize,
    pub padding: usize,
    pub kernel_size: usize,
    pub conv_shape: ConvShape,
    pub dist_method: Option<DistributionMethod>,
    phantom_data: PhantomData<T>,
}

pub struct ConvInput<T: FloatType, S: DeviceStorage<T>> {
    out_width: usize,
    out_height: usize,
    groups: usize,
    stride: usize,
    padding: usize,
    filters: Tensor<T, S>,
    batch_size: usize,
    in_channels: usize,
    in_width: usize,
    in_height: usize,
    conv_shape: ConvShape,
}

impl<T: FloatType> Default for Conv2d<T> {
    fn default() -> Self {
        Self {
            out_channels: 1,
            groups: 1,
            stride: 1,
            padding: 0,
            kernel_size: 2,
            conv_shape: ConvShape::Nchw,
            dist_method: None,
            phantom_data: PhantomData,
        }
    }
}

impl<T: FloatType> Conv2d<T>
where
    StandardNormal: Distribution<T>,
    T: SampleUniform,
{
    pub fn init<S: DeviceStorage<T>>(
        &self,
        input: &Tensor<T, S>,
        dev: &S,
    ) -> Result<ConvInput<T, S>, Box<dyn Error>> {
        let (batch_size, in_channels, in_height, in_width) = (
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        );

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

        let fan_in = self.kernel_size * self.kernel_size * in_channels;
        let fan_out = self.kernel_size * self.kernel_size * self.out_channels;
        let f_size =
            self.out_channels * (in_channels / self.groups) * self.kernel_size * self.kernel_size;
        let f_buffer = Self::init_weights(f_size, fan_in, fan_out, self.dist_method);

        let filters: Tensor<T, S> = Tensor::try_from_data(&f_shape, &f_buffer, dev).unwrap();

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

    fn compute_out_size(dim: usize, padding: usize, kernel_size: usize, stride: usize) -> usize {
        (dim + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    }

    fn init_weights(
        count: usize,
        fan_in: usize,
        fan_out: usize,
        method: Option<DistributionMethod>,
    ) -> Vec<T> {
        let mut out = vec![T::one(); count];
        match method {
            Some(m) => {
                distribution::sample(fan_in, fan_out, &mut out, m);
                out
            }
            None => out,
        }
    }
}

impl<T: FloatType, S: DeviceStorage<T>> ConvInput<T, S> {
    pub fn update_weights(mut self, weights: &[T], dev: &S) -> ConvInput<T, S> {
        self.filters = Tensor::try_from_data(&self.filters.shape, weights, dev).unwrap();
        self
    }
}

impl<T: FloatType, S: ConvKernel<T>> ConvInput<T, S> {
    pub fn forward(&self, input: &Tensor<T, S>, dev: &mut S) -> Tensor<T, S> {
        dev.conv2d_fwd(self, input.clone())
    }
}

pub(crate) fn permute_nchw_to_nhwc<T: FloatType, D: DeviceStorage<T>>(
    mut tensor: Tensor<T, D>,
) -> Tensor<T, D> {
    let indices = [0, 2, 3, 1];
    tensor.permute(&indices);
    tensor
}

pub(crate) fn permute_nhwc_to_nchw<T: FloatType, D: DeviceStorage<T>>(
    mut tensor: Tensor<T, D>,
) -> Tensor<T, D> {
    let indices = [0, 3, 1, 2];
    tensor.permute(&indices);
    tensor
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::cpu::Cpu;
    use crate::tests::TestDevice;
    use approxim::assert_abs_diff_eq;

    #[test]
    fn test_permute_nchw_to_nhwc() {
        let dev = Cpu::default();
        let t: Tensor<f32, _> = Tensor::try_new(&[3, 5, 4, 4], &dev).unwrap();
        let pt = permute_nchw_to_nhwc(t);
        assert_eq!(pt.shape, [3, 4, 4, 5]);
        assert_eq!(pt.strides, [80, 4, 1, 16]);

        let pt = permute_nhwc_to_nchw(pt);
        assert_eq!(pt.shape, [3, 5, 4, 4]);
        assert_eq!(pt.strides, [80, 16, 4, 1]);
    }

    #[test]
    fn test_to_nhwc() {
        let mut dev = TestDevice::default();
        let data: [f32; 18] = [
            2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            7.0,
        ];
        let t = Tensor::try_from_data(&[1, 3, 2, 3], &data, &dev).unwrap();
        let p = Conv2d::default();
        let _conv = t.init_conv2d(&p, &mut dev).unwrap();
        let t = t.to_nhwc(&mut dev);
        assert_eq!(
            t.try_get_data().unwrap(),
            &[
                2.0, 4.0, 2.0, 3.0, 5.0, 3.0, 4.0, 6.0, 4.0, 1.0, 7.0, 5.0, 2.0, 8.0, 6.0, 3.0,
                9.0, 7.0
            ]
        );
    }

    #[test]
    fn test_to_nchw() {
        let mut dev = TestDevice::default();
        let data: [f32; 18] = [
            2.0, 4.0, 2.0, 3.0, 5.0, 3.0, 4.0, 6.0, 4.0, 1.0, 7.0, 5.0, 2.0, 8.0, 6.0, 3.0, 9.0,
            7.0,
        ];
        let t = Tensor::try_from_data(&[1, 2, 3, 3], &data, &dev).unwrap();
        let p = Conv2d::default();
        let _conv = t.init_conv2d(&p, &mut dev).unwrap();
        let t = t.to_nchw(&mut dev);
        assert_eq!(
            t.try_get_data().unwrap(),
            &[
                2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0,
            ]
        );
    }

    #[test]
    fn test_im2col_nchw() {
        let mut dev = TestDevice::default();
        #[rustfmt::skip]
        let data: [f32; 18] = [
            2.0, 3.0, 4.0, 
            1.0, 2.0, 3.0, 
        
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0, 
        
            2.0, 3.0, 4.0, 
            5.0, 6.0, 7.0,
        ];
        let t = Tensor::try_from_data(&[1, 3, 2, 3], &data, &dev).unwrap();
        let p = Conv2d {
            out_channels: 3,
            ..Default::default()
        };
        let conv = t.init_conv2d(&p, &mut dev).unwrap();
        let v = t.im2col(&conv, &mut dev);
        let res = v.try_get_data().unwrap();
        assert_eq!(res.len(), 24); // B * OC * KH * KW * OH * OW;
        assert_eq!(
            &res,
            &[
                2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0,
                2.0, 3.0, 5.0, 6.0, 3.0, 4.0, 6.0, 7.0
            ]
        );
    }

    #[test]
    fn test_im2col_nchw_padding() {
        let mut dev = TestDevice::default();
        #[rustfmt::skip]
        let data: [f32; 18] = [
            2.0, 3.0, 4.0, 
            1.0, 2.0, 3.0,
        
            4.0, 5.0, 6.0, 
            7.0, 8.0, 9.0,
        
            2.0, 3.0, 4.0, 
            5.0, 6.0, 7.0,
        ];
        let t = Tensor::try_from_data(&[1, 3, 2, 3], &data, &dev).unwrap();
        let p = Conv2d {
            out_channels: 3,
            padding: 1,
            ..Default::default()
        };
        let conv = t.init_conv2d(&p, &mut dev).unwrap();
        let v = t.im2col(&conv, &mut dev);
        let res = v.try_get_data().unwrap();
        assert_eq!(res.len(), 144); // B * OC * KH * KW * OH * OW;
        assert_eq!(
            &res,
            &[
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0,
                0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 0.0, 3.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 6.0, 0.0,
                0.0, 4.0, 0.0, 7.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0, 6.0, 0.0, 9.0, 0.0,
                0.0, 7.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 8.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0,
                0.0, 2.0, 0.0, 5.0, 2.0, 3.0, 5.0, 6.0, 3.0, 4.0, 6.0, 7.0, 4.0, 0.0, 7.0, 0.0,
                0.0, 5.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0
            ]
        );
    }

    #[test]
    fn test_im2col_nchw_stride() {
        let mut dev = TestDevice::default();
        #[rustfmt::skip]
        let data: [f32; 24] = [
            3.0, 5.0, 1.0, 2.0, 
            4.0, 2.0, 1.0, 9.0, 
            
            3.0, 3.0, 8.0, 3.0, 
            4.0, 2.0, 1.0, 3.0, 
            
            6.0, 5.0, 7.0, 3.0, 
            2.0, 2.0, 1.0, 0.0,
        ];
        let t = Tensor::try_from_data(&[1, 3, 2, 4], &data, &dev).unwrap();
        let p = Conv2d {
            out_channels: 3,
            stride: 2,
            ..Default::default()
        };
        let conv = t.init_conv2d(&p, &mut dev).unwrap();
        let v = t.im2col(&conv, &mut dev);
        let res = v.try_get_data().unwrap();
        assert_eq!(res.len(), 24); // B * OC * KH * KW * OH * OW;
        assert_eq!(
            &res,
            &[
                3.0, 5.0, 4.0, 2.0, 1.0, 2.0, 1.0, 9.0, 3.0, 3.0, 4.0, 2.0, 8.0, 3.0, 1.0, 3.0,
                6.0, 5.0, 2.0, 2.0, 7.0, 3.0, 1.0, 0.0
            ]
        );
    }

    #[test]
    fn test_conv2d() {
        let mut dev = TestDevice::default();
        let weight = [-0.049, -0.43, 0.019, 0.097];
        let data = [-0.867, 0.527, -0.952, -0.645, 0.778, -0.490];
        let t = Tensor::try_from_data(&[1, 1, 2, 3], &data, &dev).unwrap();
        let p = Conv2d::default();
        let conv = t
            .init_conv2d(&p, &mut dev)
            .unwrap()
            .update_weights(&weight, &mut dev);
        let t = conv.forward(&t, &mut dev);
        assert_abs_diff_eq!(
            &t.try_get_data().unwrap().as_ref(),
            &[-0.120916, 0.350789].as_ref()
        );
    }

    #[test]
    fn test_conv2d_with_padding() {
        let mut dev = TestDevice::default();
        let weight = [-0.049, -0.43, 0.019, 0.097];
        let data = [-0.867, 0.527, -0.952, -0.645, 0.778, -0.490];
        let t = Tensor::try_from_data(&[1, 1, 2, 3], &data, &dev).unwrap();
        let p = Conv2d {
            padding: 1,
            ..Default::default()
        };
        let conv = t
            .init_conv2d(&p, &mut dev)
            .unwrap()
            .update_weights(&weight, &mut dev);
        let t = conv.forward(&t, &mut dev);
        #[rustfmt::skip]
        assert_abs_diff_eq!(
            &t.try_get_data().unwrap().as_ref(),
            &[
                -0.084099, 0.034646004, -0.082331, -0.018088,
                0.310245, -0.120916, 0.35078904, 0.037338,
                0.27735, -0.302935, 0.172578, 0.02401
            ].as_ref()
        );
    }

    #[test]
    fn test_conv2d_with_stride() {
        let mut dev = TestDevice::default();
        let weight = [0.67, 0.13, -0.002, 0.001];
        let data = [
            0.563, 0.112, 0.812, -0.445, -0.999, -0.02, 0.147, 0.671, -0.310, -0.022, 0.136,
            -0.111, 0.527, 0.007, 0.317, -0.094,
        ];
        let t = Tensor::try_from_data(&[2, 1, 2, 4], &data, &dev).unwrap();
        let p = Conv2d {
            stride: 2,
            ..Default::default()
        };
        let conv = t
            .init_conv2d(&p, &mut dev)
            .unwrap()
            .update_weights(&weight, &mut dev);
        let t = conv.forward(&t, &mut dev);
        #[rustfmt::skip]
        assert_abs_diff_eq!(
            &t.try_get_data().unwrap().as_ref(),
            &[
                0.39374804, 0.48656702, -0.21160701, 0.07596201
            ].as_ref()
        );
    }
}
