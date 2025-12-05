use rvit_core::pixel_traits::{BorderMode, Constant};
use rvit_core::storage::DeviceStorage;
use rvit_device::vision_traits::MakeBorderKernel;
use rvit_image::image::{Image, PixelType};

#[derive(Debug, Clone)]
pub struct MakeBorder {
    padding: usize,
}

impl MakeBorder {
    pub fn new(padding: usize) -> Self {
        Self { padding }
    }
}

impl MakeBorder {
    pub fn process<B: BorderMode, T: PixelType, D: MakeBorderKernel<T, B>>(
        &mut self,
        src: &mut Image<T, D>,
    ) -> Image<T, D> {
        if 2 * self.padding >= src.width || 2 * self.padding >= src.height {
            panic!("Padding size must be less than the image dimensions");
        }
        let shape = [src.batch_size, src.channels, src.width, src.height];
        let (data, out_shape, out_strides) = D::make_border(
            &mut src.device,
            &src.data,
            &shape,
            &src.strides,
            self.padding,
        );
        Image::from_parts(data, &out_shape, &out_strides, &src.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvit_device::tests::TestDevice;

    #[test]
    fn test_make_border_one_pad() {
        let dev = TestDevice::default();
        let mut mb = MakeBorder::new(1);
        let mut src: Image<u8, _> = Image::try_from_slice(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            1,
            4,
            4,
            1,
            &dev,
        )
        .unwrap();
        let dst = mb.process::<Constant, _, _>(&mut src);
        assert_eq!(
            &dst.try_get_data().unwrap(),
            &[
                0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 13,
                14, 15, 16, 0, 0, 0, 0, 0, 0, 0,
            ]
        );
    }

    #[test]
    fn test_make_border_none_power_two_dims() {
        let mut dev = TestDevice::default();
        let mut mb = MakeBorder::new(1);
        let mut src: Image<u8, _> =
            Image::try_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 1, 3, 3, 1, &mut dev).unwrap();
        let dst = mb.process::<Constant, _, _>(&mut src);
        assert_eq!(
            &dst.try_get_data().unwrap(),
            &[
                0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn test_make_border_batched() {
        let mut dev = TestDevice::default();
        let (b, c, w, h) = (20, 3, 3, 3);
        let template = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut src = vec![0u8; b * c * w * h];
        src.chunks_mut(w * h).for_each(|p| {
            p.copy_from_slice(template);
        });
        let mut img: Image<u8, _> = Image::try_from_slice(&src, 20, 3, 3, 3, &mut dev).unwrap();
        let mut mb = MakeBorder::new(1);
        let mb_img = mb.process::<Constant, _, _>(&mut img);
        let dst = mb_img.try_get_data().unwrap();
        dst.chunks(mb_img.width * mb_img.height).for_each(|p| {
            assert_eq!(
                p,
                &[
                    0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0
                ]
            );
        })
    }
}
