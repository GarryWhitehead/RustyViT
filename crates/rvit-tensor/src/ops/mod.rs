use crate::tensor::TensorType;
use num_traits::Float;
use rvit_core::element_traits::{DataElem, FloatElem};
use rvit_device::tensor::op_traits::{BinaryOpKernel, UnaryOpKernel};
use rvit_device::{DAlloc, Device};

pub mod convolution;
pub mod float_ops;
pub mod integer_ops;

struct TensorPrimitive<D: Device> {
    pub data: DAlloc<D>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
}
