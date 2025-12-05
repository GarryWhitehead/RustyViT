pub mod convolution;
pub mod crop;
pub mod flip;
pub mod make_border;
pub mod resize;
pub mod sep_filters;

#[cfg(test)]
pub mod tests {

    #[cfg(feature = "test-cpu")]
    pub type TestDevice = rvit_cpu::device::Cpu;

    #[cfg(feature = "test-cuda")]
    pub type TestDevice = rvit_cuda::device::Cuda;

    #[cfg(feature = "test-vulkan")]
    pub type TestDevice = rvit_vulkan::device::Vulkan;
}
