use crate::cpu::device::Cpu;
use rvit_core::element_traits::{FloatElem, IntElem};
use rvit_core::memory::storage::DeviceStorage;
use std::marker::PhantomData;
use std::sync::Arc;

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "vulkan")]
pub mod vulkan;

pub mod tensor;
pub mod vision;

#[derive(Debug, Clone)]
pub struct DeviceRuntime<D: DeviceStorage, F: FloatElem, I: IntElem> {
    _float_elem: PhantomData<F>,
    _int_elem: PhantomData<I>,
    pub storage: D,
}

impl<D: DeviceStorage, F: FloatElem, I: IntElem> DeviceRuntime<D, F, I> {
    pub fn new(device: D) -> Self {
        Self {
            _float_elem: PhantomData,
            _int_elem: PhantomData,
            storage: device,
        }
    }
}

pub type Runtime<F = f32, I = u32> = DeviceRuntime<Cpu, F, I>;
#[cfg(feature = "cuda")]
pub type CudaRuntime<F = f32, I = u32> = DeviceRuntime<Cuda, F, I>;

#[cfg(feature = "vulkan")]
pub type VulkanRuntime<F = f32, I = u32> = DeviceRuntime<Vulkan, F, I>;

pub mod tests {

    #[cfg(feature = "cpu")]
    pub type TestDevice = crate::cpu::device::Cpu;

    #[cfg(feature = "cuda")]
    pub type TestDevice = crate::cuda::device::Cuda;

    #[cfg(feature = "vulkan")]
    pub type TestDevice = crate::vulkan::device::Vulkan;
}
