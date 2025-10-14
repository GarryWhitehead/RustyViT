use super::*;
use crate::device::cpu::Cpu;
use crate::image::PixelType;
use rayon::prelude::*;

impl<T: PixelType, F: FloatType> Conv<T, F> for Cpu {
    fn convolution(
        &mut self,
        src: &mut Image<T, Self>,
        x_kernel: &Kernel<F, Self>,
        y_kernel: &Kernel<F, Self>,
    ) {
        let image_size = src.width * src.height;
        let chunk_size = src.channels * image_size;

        src.data.par_chunks_mut(chunk_size).for_each(|in_slice| {
            x_kernel.clone().device.convolution_kernel::<T, F>(
                in_slice,
                src.width,
                src.height,
                src.channels,
                x_kernel.data.as_slice(),
                y_kernel.data.as_slice(),
            );
        });
    }
}

impl Cpu {
    fn convolution_kernel<T: PixelType, F: FloatType>(
        &mut self,
        src: &mut [T],
        width: usize,
        height: usize,
        channels: usize,
        x_kernel: &[F],
        y_kernel: &[F],
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
    fn sep_vertical_pass<T: PixelType, F: FloatType>(
        &self,
        input: &[T],
        width: usize,
        row_idx: usize,
        vert_lookup: &[usize],
        kernel: &[F],
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

    fn sep_horizontal_pass<T: PixelType, F: FloatType>(
        &self,
        input: &[T],
        width: usize,
        lookup: &mut [T],
        kernel: &[F],
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

/*pub fn convolution(&self, input: &[f32], dim: usize, kernel: &[f32], out: &mut [f32]) {
    let mut output_idx = 0;
    let i32_padding = self.padding as i32;
    let conv_size = (dim - self.kernel_size + self.padding + 1) as i32;
    for y in (-i32_padding..conv_size).step_by(self.stride) {
        for x in (-i32_padding..conv_size).step_by(self.stride) {
            let mut accum = 0.0;
            // Adjust the kernel size and start index if we are at the image borders to
            // take into account padding.
            let (kx_size, kx_start) = self.compute_padded_size(x, dim);
            let (ky_size, ky_start) = self.compute_padded_size(y, dim);
            for ky in ky_start..ky_size {
                for kx in kx_start..kx_size {
                    let ky_y = (ky as i32 + y) as usize;
                    let kx_x = (kx as i32 + x) as usize;
                    accum += input[ky_y * dim + kx_x] * kernel[ky * self.kernel_size + kx];
                }
            }
            out[output_idx] = accum;
            output_idx += 1;
        }
    }
}*/
