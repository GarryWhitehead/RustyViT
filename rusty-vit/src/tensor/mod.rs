use crate::device::DeviceStorage;
use crate::tensor::convolution::{Conv2d, ConvInput, ConvKernel};
use crate::tensor::{cast::CastKernel, matmul::MatMulKernel};
use crate::type_traits::FloatType;
#[cfg(feature = "cuda")]
use cudarc::driver::{DeviceRepr, ValidAsZeroBits};
use rand::distr::Distribution;
use rand::distr::uniform::SampleUniform;
use rand_distr::StandardNormal;
use std::error::Error;

pub mod binary_op;
mod cast;
pub mod convolution;
pub mod distribution;
pub mod layer_norm;
pub mod matmul;
pub mod matrix_funcs;
mod tensor_cpu;
pub mod unary_op;

#[cfg(feature = "cuda")]
pub trait SafeZeros: ValidAsZeroBits + DeviceRepr {}
#[cfg(not(feature = "cuda"))]
pub trait SafeZeros {}

#[allow(dead_code)]
#[derive(Clone)]
pub struct Tensor<T: FloatType, S: DeviceStorage<T>> {
    pub(crate) data: S::Vec,
    pub(crate) device: S,
    pub(crate) shape: Vec<usize>,
    pub(crate) strides: Vec<usize>,
}

impl<T: FloatType, S: DeviceStorage<T>> Tensor<T, S> {
    pub fn try_new(shape: &[usize], dev: &S) -> Result<Self, Box<dyn Error>> {
        if shape.is_empty() {
            return Err("shape cannot be empty".into());
        }

        let sz = shape.iter().copied().reduce(|a, b| a * b).unwrap();
        Ok(Self {
            data: dev.try_alloc(sz)?,
            device: dev.clone(),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        })
    }

    fn compute_strides(shape: &[usize]) -> Vec<usize> {
        let dims = shape.len();
        let mut strides = vec![1; dims];
        for i in (0..(dims - 1)).rev() {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        strides
    }

    pub fn is_contiguous(&self) -> bool {
        let mut expected_stride = 1;
        for (shape, stride) in self.shape.iter().zip(&self.strides).rev() {
            if *stride != expected_stride {
                return false;
            }
            expected_stride *= shape;
        }
        true
    }

    pub fn try_from_data(shape: &[usize], values: &[T], dev: &S) -> Result<Self, Box<dyn Error>> {
        if shape.is_empty() {
            return Err("shape cannot be empty".into());
        }
        let sz = shape.iter().copied().reduce(|a, b| a * b).unwrap();
        if values.len() != sz {
            return Err("Shape size doesn't match values length".into());
        }

        Ok(Self {
            data: dev.try_alloc_with_slice(values)?,
            device: dev.clone(),
            shape: shape.to_vec(),
            strides: Self::compute_strides(shape),
        })
    }

    pub fn total_size(&self) -> usize {
        self.shape.iter().copied().reduce(|a, b| a * b).unwrap()
    }

    pub fn try_get_data(&self) -> Result<Vec<T>, Box<dyn Error>> {
        self.device.try_sync()?;
        self.device.try_from_device_vec(&self.data)
    }

    pub fn permute(&mut self, indices: &[usize]) {
        if indices.len() != self.shape.len() {
            panic!("Indices length doesn't match the tensor dimensions");
        }

        self.shape = indices.iter().map(|i| self.shape[*i]).collect();
        self.strides = indices.iter().map(|i| self.strides[*i]).collect();
    }
}

impl<T: FloatType, D: MatMulKernel<T>> Tensor<T, D> {
    pub fn matmul(&self, rhs: &Tensor<T, D>, dev: &mut D) -> Tensor<T, D> {
        if self.shape.len() < 2 || rhs.shape.len() < 2 {
            panic!("Tensor must have at least two dimensions (for now).");
        }
        if self.shape.len() != rhs.shape.len() {
            panic!("Tensors must have the same dimensions.");
        }
        if self.shape.len() > 2 && (self.shape[0] != rhs.shape[0]) {
            panic!("Tensors must have the same batch size");
        }

        dev.matmul(self, rhs)
    }
}

impl<T: FloatType, D: DeviceStorage<T>> Tensor<T, D> {
    pub fn cast<O: FloatType>(&self, dev: &mut D) -> Tensor<O, D>
    where
        D: CastKernel<T, O>,
    {
        dev.cast(self)
    }
}

impl<T: FloatType, D: ConvKernel<T>> Tensor<T, D>
where
    StandardNormal: Distribution<T>,
    T: SampleUniform,
{
    pub fn init_conv2d(
        &self,
        p: &Conv2d<T>,
        dev: &mut D,
    ) -> Result<ConvInput<T, D>, Box<dyn Error>> {
        if p.stride < 1 {
            panic!("The stride must be greater than zero");
        }
        let in_channels = self.shape[1];
        if !in_channels.is_multiple_of(p.groups) || !p.out_channels.is_multiple_of(p.groups) {
            panic!("The group size must be a multiple of the in/out channel size");
        }
        p.init(self, dev)
    }

    pub fn to_nhwc(&self, dev: &mut D) -> Tensor<T, D> {
        dev.nchw_to_nhwc(self)
    }

    pub fn to_nchw(&self, dev: &mut D) -> Tensor<T, D> {
        dev.nhwc_to_nchw(self)
    }

    pub fn im2col(&self, p: &ConvInput<T, D>, dev: &mut D) -> Tensor<T, D> {
        dev.im2col(p, self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::device::cpu::Cpu;

    #[test]
    fn test_is_contiguous() {
        let dev = Cpu::default();
        let mut t: Tensor<f32, _> = Tensor::try_new(&[1, 2, 3, 3], &dev).unwrap();
        assert!(t.is_contiguous());

        t.permute(&[0, 2, 3, 1]);
        assert!(!t.is_contiguous());
    }

    #[test]
    fn test_permute() {
        let dev = Cpu::default();
        let mut t: Tensor<f32, _> = Tensor::try_new(&[1, 2, 3, 3], &dev).unwrap();
        t.permute(&[0, 2, 3, 1]);
        assert_eq!(&t.shape, &[1, 3, 3, 2]);
        assert_eq!(t.strides, vec![18, 3, 1, 9]);
    }
}
