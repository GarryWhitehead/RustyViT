mod convolution;
mod crop;
mod flip;
mod make_border;
mod resize;

#[macro_export]
macro_rules! testgen_vision {
    ([$($float:ident),*], [$($int:ident),*]) => {
        pub mod vision {
            use super::*;
            use rvit_image::image::Image;
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
                    $crate::testgen_vision_tests!();
                })*
                $(mod [<$int _tests>] {
                    use super::*;

                    pub type TestType = $int;
                    $crate::testgen_vision_tests!();
                })*
            }
        }
    };
}

#[macro_export]
macro_rules! testgen_vision_tests {
    () => {
        rvit_vision::testgen_make_border!();
        rvit_vision::testgen_crop!();
        rvit_vision::testgen_flip!();
        rvit_vision::testgen_resize!();
        rvit_vision::testgen_sep_filter_convolution!();
    };
}
