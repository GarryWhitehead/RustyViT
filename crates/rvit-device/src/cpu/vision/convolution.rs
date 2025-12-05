use crate::cpu::device::Cpu;
use crate::vision_traits::ConvKernel;
use rayon::prelude::*;
use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;

impl<T: PixelType> ConvKernel<T> for Cpu {
    fn convolution(
        &mut self,
        src: &mut Self::Vec,
        src_shape: &[usize],
        _src_strides: &[usize],
        x_kernel: &Self::Vec,
        y_kernel: &Self::Vec,
    ) {
        let (channels, width, height) = (src_shape[1], src_shape[2], src_shape[3]);
        let image_size = width * height;
        let chunk_size = channels * image_size;

        src.par_chunks_mut(chunk_size).for_each(|in_slice| {
            let mut dev = &mut self.clone();
            dev.convolution_kernel::<T>(
                in_slice,
                width,
                height,
                channels,
                x_kernel.as_slice(),
                y_kernel.as_slice(),
            );
        });
    }
}

impl Cpu {
    fn convolution_kernel<T: PixelType>(
        &mut self,
        src: &mut [T],
        width: usize,
        height: usize,
        channels: usize,
        x_kernel: &[T],
        y_kernel: &[T],
    ) {
        let mut horiz_lookup = vec![T::zero(); width + 2 * (x_kernel.len() >> 1)];
        let vert_lookup = self.compute_border_table(height, y_kernel.len());

        let mut temp = self.try_alloc(channels * width * height).unwrap();

        for c in 0..channels {
            let base = c * width * height;
            let t_slice = &mut temp[base..base + width * height];
            // Vertical convolution pass.
            for row in 0..height {
                let row_slice = &mut t_slice[row * width..row * width + width];
                self.sep_vertical_pass(src, width, row, &vert_lookup, y_kernel, row_slice);
            }

            // Horizontal convolution pass.
            for row in 0..height {
                let input_slice = &t_slice[row * width..row * width + width];
                let out_slice = &mut src[row * width..row * width + width];
                self.sep_horizontal_pass(
                    input_slice,
                    width,
                    &mut horiz_lookup,
                    x_kernel,
                    out_slice,
                );
            }
        }
    }
}

impl Cpu {
    fn sep_vertical_pass<T: PixelType>(
        &self,
        input: &[T],
        width: usize,
        row_idx: usize,
        vert_lookup: &[usize],
        kernel: &[T],
        out: &mut [T],
    ) {
        for (col, o) in out.iter_mut().enumerate().take(width) {
            let mut accum = 0.0;
            for k in 0..kernel.len() {
                let idx = col + (width * vert_lookup[row_idx + k]);
                let val: f32 = input[idx].to_float();
                accum += val * kernel[k].to_f32().unwrap();
            }
            *o = T::from_float(accum);
        }
    }

    fn sep_horizontal_pass<T: PixelType>(
        &self,
        input: &[T],
        width: usize,
        lookup: &mut [T],
        kernel: &[T],
        out: &mut [T],
    ) {
        self.fill_border_table(input, width, kernel.len(), lookup);
        for col in 0..width {
            let mut accum = 0.0;
            for k in 0..kernel.len() {
                let val: f32 = lookup[col + k].to_float();
                accum += val * kernel[k].to_f32().unwrap();
            }
            out[col] = T::from_float(accum);
        }
    }

    fn fill_border_table<T: PixelType>(
        &self,
        input: &[T],
        dim: usize,
        kernel_size: usize,
        out: &mut [T],
    ) {
        let k_half_size = kernel_size >> 1;
        out[k_half_size..k_half_size + dim].copy_from_slice(input);
        for i in 0..k_half_size {
            out[i] = input[0];
            out[i + dim + k_half_size] = input[dim - 1];
        }
    }

    fn compute_border_table(&self, dim: usize, kernel_size: usize) -> Vec<usize> {
        let k_half_size = kernel_size >> 1;
        let mut out = vec![0usize; dim + kernel_size];
        // Indices for the top border.
        for o in out.iter_mut().take(k_half_size) {
            *o = 0;
        }
        // Input indices.
        for i in 0..dim {
            out[i + k_half_size] = i;
        }
        // Indices for the bottom border.
        for i in 0..k_half_size {
            out[i + k_half_size + dim] = dim - 1;
        }
        out
    }
}
