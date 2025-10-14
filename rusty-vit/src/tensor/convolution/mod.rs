mod conv_cpu;
#[cfg(feature = "cuda")]
mod conv_cu;

#[allow(dead_code)]
pub struct Convolution {
    in_channels: usize,
    out_channels: usize,
    group: usize,
    stride: usize,
    padding: usize,
    pad_value: f32,
    kernel_size: usize,
}

impl Convolution {
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        group: usize,
        stride: usize,
        padding: usize,
        kernel_size: usize,
        pad_value: f32,
    ) -> Self {
        if stride < 1 {
            panic!("The stride must be greater than zero");
        }
        if !in_channels.is_multiple_of(group) || !out_channels.is_multiple_of(group) {
            panic!("The group size must be a multiple of the in/out channel size");
        }

        Self {
            in_channels,
            out_channels,
            group,
            stride,
            padding,
            kernel_size,
            pad_value,
        }
    }
}
