use crate::cpu::device::Cpu;
use crate::vision_traits::CropKernel;
use rayon::prelude::*;
use rvit_core::pixel_traits::PixelType;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::*;

impl<T: PixelType> CropKernel<T> for Cpu {
    fn crop(
        &mut self,
        src: &mut Self::Vec,
        src_shape: &[usize],
        src_strides: &[usize],
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
    ) -> (Self::Vec, Vec<usize>, Vec<usize>)
    where
        Self: Sized,
    {
        let (batch_size, channels, width, height) =
            (src_shape[0], src_shape[1], src_shape[2], src_shape[3]);
        let crop_shape = [batch_size, channels, crop_width, crop_height];
        let crop_sz = tensor_size(&crop_shape);
        let crop_strides = compute_strides(&crop_shape);

        let mut crop_img = self.try_alloc(crop_sz).unwrap();
        src.data
            .par_chunks(src_strides[0])
            .zip(crop_img.data.par_chunks_mut(crop_strides[0]))
            .for_each(|(in_slice, out_slice)| {
                Self::crop_kernel(
                    in_slice,
                    width,
                    height,
                    channels,
                    crop_width,
                    crop_height,
                    x,
                    y,
                    out_slice,
                );
            });

        (crop_img, crop_shape.to_vec(), crop_strides)
    }
}

#[allow(clippy::too_many_arguments)]
impl Cpu {
    pub fn crop_kernel<T: PixelType>(
        src: &[T],
        width: usize,
        height: usize,
        channels: usize,
        crop_width: usize,
        crop_height: usize,
        x: usize,
        y: usize,
        out: &mut [T],
    ) {
        assert!(crop_width <= width);
        assert!(crop_height <= height);
        assert!(x + crop_width <= width);
        assert!(y + crop_height <= height);

        // Nothing to be done if the cropped sizes match the original.
        if crop_width == width && crop_height == height {
            out.copy_from_slice(src);
            return;
        }

        for c in 0..channels {
            let src_offset = c * width * height;
            let crop_offset = c * crop_width * crop_height;
            let src_slice = &src[src_offset..src_offset + width * height];
            let crop_slice = &mut out[crop_offset..crop_offset + crop_width * crop_height];
            for row in 0..crop_height {
                let row_idx = y + row * width + x;
                crop_slice[row * crop_width..row * crop_width + crop_width]
                    .copy_from_slice(&src_slice[row_idx..row_idx + crop_width])
            }
        }
    }
}
