use super::*;
use crate::device::cpu::Cpu;
use crate::image::PixelType;
use rayon::prelude::*;

impl<T: PixelType> ConvKernel<T> for Cpu {
    fn convolution(&self, src: &mut Image<T, Self>, kernel: &Kernel) {
        let image_size = src.width * src.height;
        let chunk_size = src.channels * image_size;

        src.data.par_chunks_mut(chunk_size).for_each(|in_slice| {
            self.convolution_kernel(in_slice, src.width, src.height, src.channels, &kernel.data, kernel.width, kernel.height);
        });
    }
}

impl Cpu {
    pub fn convolution_kernel<T: PixelType>(&self, src: &mut [T], width: usize, height: usize, channels: usize, kernel: &[f32], k_width: usize, k_height: usize) {
        let mut horiz_lookup = vec![T::zero(); width];
        let vert_lookup = self.compute_border_table(height, k_height);

        let mut temp = self.try_alloc(channels * width * height).unwrap();

        for c in 0..channels {
            let base = c * width * height;
            let t_slice = &mut temp[base..base + width * height];
            // Vertical convolution pass.
            for row in 0..height {
                let row_slice =
                    &mut t_slice[row * width..row * width + width];
                self.sep_vertical_pass(
                    src,
                    width,
                    row,
                    &vert_lookup,
                    &kernel,
                    k_height,
                    row_slice,
                );
            }

            // Horizontal convolution pass.
            for row in 0..height {
                let input_slice = &t_slice[row * width..row * width + width];
                let out_slice =
                    &mut src[row * width..row * width + width];
                self.sep_horizontal_pass(
                    input_slice,
                    width,
                    &mut horiz_lookup,
                    &kernel,
                    k_width,
                    out_slice,
                );
            }
        }
    }

    fn sep_vertical_pass<T: PixelType>(
        &self,
        input: &[T],
        width: usize,
        row_idx: usize,
        vert_lookup: &[usize],
        kernel: &[f32],
        kernel_size: usize,
        out: &mut [T],
    ) {
        for col in 0..width {
            let mut accum = 0.0;
            for k in 0..kernel_size {
                let idx = col + (width * vert_lookup[row_idx + k]);
                let val: f32 = input[idx].to_float();
                accum += val * kernel[k];
            }
            out[col] = T::from_float(accum);
        }
    }

    fn sep_horizontal_pass<T: PixelType>(
        &self,
        input: &[T],
        width: usize,
        lookup: &mut [T],
        kernel: &[f32],
        kernel_size: usize,
        out: &mut [T],
    ) {
        self.fill_border_table(input, width, kernel_size, lookup);
        for col in 0..width {
            let mut accum = 0.0;
            for k in 0..kernel_size {
                let val: f32 = lookup[col + k].to_float();
                accum += val * kernel[k];
            }
            out[col] = T::from_float(accum);
        }
    }

    fn fill_border_table<T: PixelType>(&self, input: &[T], dim: usize, kernel_size: usize, out: &mut [T]) {
        let k_half_size = kernel_size >> 1;
        out.fill(T::zero());
        out[k_half_size..dim].copy_from_slice(input);
    }

    fn compute_border_table(&self, dim: usize, kernel_size: usize) -> Vec<usize> {
        let k_half_size = kernel_size >> 1;
        let mut out = vec![0usize, dim + kernel_size];
        // Indices for the top border.
        for i in 0..k_half_size {
            out[i] = k_half_size - i;
        }
        // Input indices.
        for i in 0..dim {
            out[i + k_half_size] = i;
        }
        // Indices for the bottom border.
        for i in 0..k_half_size {
            out[i + k_half_size + dim] = dim - i;
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
