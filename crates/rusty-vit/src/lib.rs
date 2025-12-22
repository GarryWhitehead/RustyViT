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

#[cfg(test)]
pub mod tests {
    //use rvit_vision::testgen_vision;
    //testgen_vision!([f32], [u8, u16]);

    use half::f16;
    use rvit_tensor::testgen_tensor;
    testgen_tensor!([f32, f16]);
}
