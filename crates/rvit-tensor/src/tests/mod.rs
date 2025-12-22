mod binary_ops;
mod cast;
mod convolution;
mod matmul;
mod unary_ops;

#[macro_export]
macro_rules! testgen_tensor {
    ([$($float:ident),*]) => {
        pub mod tensor {
            use super::*;
            use rvit_tensor::tensor::Tensor;
            use rvit_core::element_traits::DataElem;
            use rvit_device::{Runtime, DeviceRuntime};
            use rvit_tensor::FloatTensor;
            use rvit_tensor::basic_ops::BasicOps;
            use rvit_tensor::tensor_ops::TensorOps;
            use num_traits::Zero;
            use rvit_core::approx;
            use paste::paste;

            #[cfg(feature = "cpu")]
            type TestDevice = rvit_device::cpu::device::Cpu;

            #[cfg(feature = "cuda")]
            type TestDevice = rvit_device::cuda::device::Cuda;

            #[cfg(feature = "vulkan")]
            type TestDevice = rvit_device::vulkan::device::Vulkan;

            paste! {
                $(mod [<$float _tests>] {
                    use super::*;

                    pub type TestType = $float;
                    $crate::testgen_tensor_tests!();
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_tensor_tests {
    () => {
        rvit_tensor::testgen_binary_ops!();
        //rvit_tensor::testgen_convolution2d!();
        rvit_tensor::testgen_matmul!();
        //rvit_tensor::testgen_unary_ops!();
        rvit_tensor::testgen_cast!();
    };
}
