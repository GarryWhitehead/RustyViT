use crate::device::DeviceStorage;
use crate::device::cpu::Cpu;
use crate::image::{Image, PixelType};
use crate::vision::make_border::{BorderMode, ClampToEdge, Constant, Mirror};
use rayon::prelude::*;

pub(crate) trait InterpOp<I: BorderMode> {
    fn compute_idx(pos: usize, pad_size: usize, image_size: usize) -> usize;
}

impl InterpOp<Constant> for Cpu {
    fn compute_idx(_pos: usize, _pad_size: usize, _image_size: usize) -> usize {
        usize::MAX
    }
}
impl InterpOp<ClampToEdge> for Cpu {
    fn compute_idx(pos: usize, _pad_size: usize, image_size: usize) -> usize {
        if pos < 0 { 0 } else { image_size - 1 }
    }
}
impl InterpOp<Mirror> for Cpu {
    fn compute_idx(pos: usize, pad_size: usize, image_size: usize) -> usize {
        if pos < pad_size {
            pad_size - pos
        } else {
            image_size - (pos - image_size) - 2
        }
    }
}

impl<T: PixelType, I: BorderMode> super::MakeBorderKernel<T, I> for Cpu
where
    Self: InterpOp<I>,
{
    fn make_border(
        &mut self,
        src: &Image<T, Self>,
        padding: usize,
        fill_value: T,
    ) -> Image<T, Self> {
        // New batched image size with padding added.
        let new_image_width = src.width + 2 * padding;
        let new_image_height = src.height + 2 * padding;
        let mut mb_image = Image::try_new(
            src.batch_size,
            new_image_width,
            new_image_height,
            src.channels,
            self,
        )
        .unwrap();

        src.data
            .par_chunks(src.strides[0])
            .zip(mb_image.data.par_chunks_mut(mb_image.strides[0]))
            .for_each(|(in_slice, out_slice)| {
                self.make_border_kernel(
                    in_slice,
                    src.width,
                    src.height,
                    src.channels,
                    padding,
                    Self::compute_idx,
                    out_slice,
                );
            });
        mb_image
    }
}

impl Cpu {
    fn make_border_kernel<T: PixelType, F>(
        &self,
        src: &[T],
        width: usize,
        height: usize,
        channels: usize,
        padding: usize,
        op: F,
        out: &mut [T],
    ) where
        F: Fn(usize, usize, usize) -> usize,
    {
        let new_width = width + padding * 2;
        let new_height = height + padding * 2;

        for c in 0..channels {
            let base_offset = c * width * height;
            let new_base_offset = c * new_width * new_height;
            let out_channel_slice =
                &mut out[new_base_offset..new_base_offset + new_width * new_height];
            let src_channel_slice = &src[base_offset..base_offset + width * height];

            // Copy the image into the new buffer taking note of the padding
            // requirements.
            let start_offset = padding * new_width + padding;
            for row in 0..height {
                let row_offset = start_offset + row * new_width;
                out_channel_slice[row_offset..row_offset + width]
                    .copy_from_slice(&src_channel_slice[row * width..row * width + width])
            }

            // If the border mode is constant, we are done here.
            if op(0, 0, 0) == usize::MAX {
                continue;
            }

            // Apply the border to the left and right edges.
            for row in 0..height {
                // Left edge.
                for i in 0..padding {
                    let interp_idx = op(i, padding, width);
                    out_channel_slice[i + new_width * padding + row] =
                        src_channel_slice[interp_idx + width * row]
                }
                // Right edge.
                for i in 0..padding {
                    let interp_idx = op(i + width, padding, width);
                    out_channel_slice[i + width * new_width * padding + row] =
                        src_channel_slice[interp_idx + width * row]
                }
            }

            // Apply the border to the top and bottom regions.
            // Top edge.
            {
                let (out, input) = out_channel_slice.split_at_mut(new_width * padding);
                for row in 0..padding {
                    let idx = op(row, padding, height);
                    out[row * new_width..row * new_width + new_width]
                        .copy_from_slice(&input[idx * new_width..idx * new_width + new_width]);
                }
            }
            {
                // Bottom edge.
                let (input, out) = out_channel_slice.split_at_mut(padding + height * new_width);
                for row in 0..padding {
                    let idx = op(row + height, padding, height);
                    out[row * new_width..row * new_width + new_width].copy_from_slice(
                        &input[idx + padding * new_width..idx + padding * new_width + new_width],
                    );
                }
            }
        }
    }
}
