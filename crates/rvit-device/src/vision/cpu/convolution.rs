use crate::cpu::device::Cpu;
use crate::vision::op_traits::ConvKernel;
use crate::{DAlloc, Runtime};
use rayon::prelude::*;
use rvit_core::element_traits::{DataElem, Elem, FloatElem, IntElem};
use rvit_core::memory::storage::DeviceStorage;

impl<E: DataElem> ConvKernel<E> for Runtime {
    fn convolution(
        &mut self,
        src: &mut DAlloc<Self>,
        src_shape: &[usize],
        _src_strides: &[usize],
        x_kernel: &DAlloc<Self>,
        y_kernel: &DAlloc<Self>,
    ) {
        let (channels, width, height) = (src_shape[1], src_shape[2], src_shape[3]);
        let image_size = width * height;
        let chunk_size = channels * image_size;

        src.as_vec()
            .unwrap()
            .chunks_mut(chunk_size)
            .for_each(|in_slice: &mut [E]| {
                let dev = &mut self.clone();
                dev.convolution_kernel(
                    in_slice,
                    width,
                    height,
                    channels,
                    x_kernel.as_slice().unwrap(),
                    y_kernel.as_slice().unwrap(),
                );
            });
    }
}

impl Runtime {
    fn convolution_kernel<E: DataElem>(
        &mut self,
        src: &mut [E],
        width: usize,
        height: usize,
        channels: usize,
        x_kernel: &[E],
        y_kernel: &[E],
    ) {
        let mut horiz_lookup = vec![E::zero(); width + 2 * (x_kernel.len() >> 1)];
        let vert_lookup = self.compute_border_table(height, y_kernel.len());

        let mut temp = self
            .storage
            .try_alloc(channels * width * height, E::DTYPE)
            .unwrap();
        let temp_slice = temp.as_mut_slice().unwrap();

        for c in 0..channels {
            let base = c * width * height;
            let t_slice = &mut temp_slice[base..base + width * height];
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

impl Runtime {
    fn sep_vertical_pass<E: DataElem>(
        &self,
        input: &[E],
        width: usize,
        row_idx: usize,
        vert_lookup: &[usize],
        kernel: &[E],
        out: &mut [E],
    ) {
        for (col, o) in out.iter_mut().enumerate().take(width) {
            let mut accum = 0.0;
            for k in 0..kernel.len() {
                let idx = col + (width * vert_lookup[row_idx + k]);
                let val: f32 = input[idx].to_f32().unwrap();
                accum += val * kernel[k].to_f32().unwrap();
            }
            *o = E::from(accum).unwrap();
        }
    }

    fn sep_horizontal_pass<T: DataElem>(
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
                let val: f32 = lookup[col + k].to_f32().unwrap();
                accum += val * kernel[k].to_f32().unwrap();
            }
            out[col] = T::from(accum).unwrap();
        }
    }

    fn fill_border_table<T: DataElem>(
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
