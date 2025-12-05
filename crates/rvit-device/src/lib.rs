pub mod cpu;
pub mod op_traits;

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "vulkan")]
pub mod vulkan;

pub(crate) mod tensor;
pub mod vision_traits;

//#[cfg(test)]
pub mod tests {

    #[cfg(feature = "cpu")]
    pub type TestDevice = crate::cpu::device::Cpu;

    #[cfg(feature = "cuda")]
    pub type TestDevice = crate::cuda::device::Cuda;

    #[cfg(feature = "vulkan")]
    pub type TestDevice = crate::vulkan::device::Vulkan;
}
