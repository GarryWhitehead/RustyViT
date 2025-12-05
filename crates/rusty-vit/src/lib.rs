#[cfg(feature = "cpu")]
pub mod cpu {
    pub use rvit_device::cpu::device::Cpu;
}

#[cfg(feature = "cuda")]
pub mod cpu {
    pub use rvit_cuda::cuda::Cuda;
}

#[cfg(feature = "vulkan")]
pub mod cpu {
    pub use rvit_cuda::vk::Vulkan;
}
