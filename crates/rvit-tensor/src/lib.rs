pub mod basic_ops;
pub mod distribution;
pub mod index;
pub mod ops;
pub mod tensor;
pub mod tensor_ops;
pub(crate) mod tests;

use crate::basic_ops::BasicOps;
use crate::float_ops::TensorFloatOps;
use crate::tensor::{Float, Integer, Tensor};
pub use ops::*;
use rvit_core::element_traits::{FloatElem, IntElem};
use rvit_core::memory::storage::DeviceStorage;
use rvit_device::tensor::op_traits::*;
use rvit_device::{Device, DeviceRuntime, Runtime};

pub type FloatTensor = Tensor<Float, Runtime>;
pub type IntTensor = Tensor<Integer, Runtime>;

trait VisionOps<D: Device>: BinaryOpKernel<D::FloatElem, BinaryAddOp> {}
trait Device: Device + TensorFloatOps<Self> + VisionOps<Self> + BasicOps<Self> {}
