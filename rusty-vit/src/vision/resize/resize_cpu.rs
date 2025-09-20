use crate::device::DeviceStorage;
use crate::device::cpu::Cpu;
use crate::image::{Image, PixelType};
use crate::vision::resize::{Bilinear, InterpMode};
use rayon::iter::IndexedParallelIterator;
use rayon::prelude::*;

trait InterpOp<I: InterpMode> {
    fn support() -> f32;
    fn filter(x: f32) -> f32;
}

impl InterpOp<Bilinear> for Cpu {
    fn support() -> f32 {
        1.0
    }
    fn filter(x: f32) -> f32 {
        f32::max(1.0 - x.abs(), 0.0)
    }
}

impl<T: PixelType, I: InterpMode> super::ResizeKernel<T, I> for Cpu
where
    Self: InterpOp<I>,
{
    fn resize(
        &mut self,
        src: &mut Image<T, Self>,
        dst_width: usize,
        dst_height: usize,
    ) -> Image<T, Self> {
        let mut rz_img =
            Image::try_new(src.batch_size, dst_width, dst_height, src.channels, self).unwrap();
        src.data
            .par_chunks(src.strides[0])
            .zip(rz_img.data.par_chunks_mut(rz_img.strides[0]))
            .for_each(|(in_slice, out_slice)| {
                let tmp = self.resize_kernel(
                    in_slice,
                    src.channels,
                    src.width,
                    src.height,
                    dst_width,
                    dst_height,
                );
                out_slice.copy_from_slice(tmp.as_slice());
            });
        rz_img
    }
}

impl Cpu {
    fn compute_coeffs(src_size: i32, dst_size: i32) -> (i32, Vec<i32>, Vec<f32>) {
        let ratio = src_size as f32 / dst_size as f32;
        let filter_scale = ratio.max(1.0);
        let filter_radius: f32 = Self::support() * filter_scale;
        let filter_size = filter_radius.ceil() as i32 * 2 + 1;

        let mut bounds_buffer = vec![0i32; (2 * dst_size) as usize];
        let mut coeff_buffer = vec![0.0f32; (filter_size * dst_size) as usize];

        for xx in 0..dst_size {
            let center = (xx as f32 + 0.5) * ratio;
            let xmin = (center - filter_radius + 0.5).max(0.0) as i32;
            let xmax = (center + filter_radius + 0.5).min(src_size as f32) as i32 - xmin;

            let filter_idx = (xx * filter_size) as usize;
            let filter_slice = &mut coeff_buffer[filter_idx..filter_idx + filter_size as usize];
            let accum: f32 = (0..xmax)
                .map(|x| {
                    let val = Self::filter((xmin as f32 + x as f32 - center + 0.5) / filter_scale);
                    filter_slice[x as usize] = val;
                    val
                })
                .sum();
            if accum > 0.0 {
                // Normalise the co-efficients.
                (0..xmax).for_each(|i| filter_slice[i as usize] /= accum);
            }
            bounds_buffer[xx as usize * 2 + 0] = xmin;
            bounds_buffer[xx as usize * 2 + 1] = xmax;
        }
        (filter_size, bounds_buffer.to_vec(), coeff_buffer.to_vec())
    }

    pub fn resize_kernel<T: PixelType>(
        &self,
        src: &[T],
        channels: usize,
        src_width: usize,
        src_height: usize,
        dst_width: usize,
        dst_height: usize,
    ) -> <Cpu as DeviceStorage<T>>::Vec {
        let (vert_k_size, mut vert_bounds, vert_coeffs) =
            Self::compute_coeffs(src_height as i32, dst_height as i32);
        let (horiz_k_size, horiz_bounds, horiz_coeffs) =
            Self::compute_coeffs(src_width as i32, dst_width as i32);

        // Adjust all xmax values to start from the first valid row.
        (0..dst_height).for_each(|i| vert_bounds[i * 2] -= vert_bounds[0]);

        let ybox_first = vert_bounds[0];
        let ybox_last = vert_bounds[dst_height * 2 - 2] + vert_bounds[dst_height * 2 - 1];
        let temp_height = ybox_last - ybox_first;

        let mut temp_image = <Cpu as DeviceStorage<T>>::Vec::default();
        let mut out = self.try_alloc(dst_width * dst_height).unwrap();

        // Horizontal pass.
        if src_width != dst_width {
            temp_image = self.try_alloc(temp_height as usize * dst_width).unwrap();
            self.horizontal_pass(
                src,
                src_width,
                ybox_first as usize,
                dst_width,
                temp_height as usize,
                horiz_k_size as usize,
                &horiz_bounds,
                &horiz_coeffs,
                &mut temp_image,
            );
        }

        // Vertical pass.
        if src_height != dst_height {
            if temp_image.is_empty() {
                let t_src = self.transpose(src, src_width, src_height);
                self.vertical_pass(
                    &t_src,
                    src_height,
                    dst_height,
                    dst_width,
                    vert_k_size as usize,
                    &vert_bounds,
                    &vert_coeffs,
                    out.as_mut_slice(),
                );
            } else {
                // Transpose so can do another horizontal pass to reduce cache misses - W * H -> H * W
                let t_temp =
                    self.transpose(temp_image.as_mut_slice(), dst_width, temp_height as usize);
                self.vertical_pass(
                    t_temp.as_slice(),
                    temp_height as usize,
                    dst_height,
                    dst_width,
                    vert_k_size as usize,
                    &vert_bounds,
                    &vert_coeffs,
                    out.as_mut_slice(),
                );
            }
            return self.transpose(out.as_slice(), dst_height, dst_width);
        }
        temp_image
    }

    fn transpose<T: PixelType>(
        &self,
        input: &[T],
        width: usize,
        height: usize,
    ) -> <Cpu as DeviceStorage<T>>::Vec {
        assert_eq!(input.len(), width * height);
        let mut dst = self.try_alloc(width * height).unwrap();
        for n in 0..width * height {
            let i = n / height;
            let j = n % height;
            dst[n] = input[width * j + i];
        }
        dst
    }

    fn transpose_block<const BLOCK_SIZE: usize>(
        input: &[u8],
        width: usize,
        height: usize,
    ) -> Vec<u8> {
        assert_eq!(input.len(), width * height);
        let mut dst = vec![0u8; width * height];
        for y in (0..height).step_by(BLOCK_SIZE) {
            for x in (0..width).step_by(BLOCK_SIZE) {
                for yy in y..y + BLOCK_SIZE {
                    for xx in x..x + BLOCK_SIZE {
                        dst[xx + yy * width] = input[yy + xx * width];
                    }
                }
            }
        }
        dst
    }

    fn resample<T: PixelType>(
        src: &[T],
        src_stride: usize,
        dst_width: usize,
        dst_height: usize,
        offset: usize,
        k_size: usize,
        bounds: &[i32],
        coeffs: &[f32],
        out: &mut [T],
    ) {
        for yy in 0..dst_height {
            for xx in 0..dst_width {
                let xmin = bounds[2 * xx];
                let xmax = bounds[2 * xx + 1];
                let coeff_slice = &coeffs[xx * k_size..xx * k_size + xmax as usize];

                let mut accum: f32 = 0.0;
                for x in 0..xmax {
                    accum += src[(yy + offset) * src_stride + (x + xmin) as usize].to_float()
                        * coeff_slice[x as usize];
                }
                out[yy * dst_width + xx] = T::from_float(accum.round());
            }
        }
    }

    fn vertical_pass<T: PixelType>(
        &self,
        src: &[T],
        src_stride: usize,
        dst_width: usize,
        dst_height: usize,
        k_size: usize,
        bounds: &[i32],
        coeffs: &[f32],
        out: &mut [T],
    ) {
        Self::resample(
            src, src_stride, dst_width, dst_height, 0, k_size, bounds, coeffs, out,
        );
    }

    fn horizontal_pass<T: PixelType>(
        &self,
        src: &[T],
        src_stride: usize,
        offset: usize,
        dst_width: usize,
        dst_height: usize,
        k_size: usize,
        bounds: &[i32],
        coeffs: &[f32],
        out: &mut [T],
    ) {
        Self::resample(
            src, src_stride, dst_width, dst_height, offset, k_size, bounds, coeffs, out,
        );
    }
}
