use rvit_core::pixel_traits::{Bilinear, InterpMode, PixelType};
use rvit_core::storage::DeviceStorage;
use rvit_core::type_traits::FloatType;
use rvit_device::vision_traits::ResizeKernel;
use rvit_image::image::Image;

pub struct Resize {
    dst_width: usize,
    dst_height: usize,
}

impl Resize {
    pub fn new(dst_width: usize, dst_height: usize) -> Self {
        if dst_width == 0 || dst_height == 0 {
            panic!("Resized image dimensions must be non-zero");
        }
        Self {
            dst_width,
            dst_height,
        }
    }
}

impl Resize {
    pub fn process<T: PixelType, I: InterpMode, D: ResizeKernel<T, I>>(
        &mut self,
        src: &mut Image<T, D>,
    ) -> Image<T, D> {
        let shape = [
            src.batch_size,
            src.channels,
            self.dst_width,
            self.dst_height,
        ];
        let (data, out_shape, out_strides) = D::resize(
            &mut src.device,
            &src.data,
            &shape,
            &src.strides,
            self.dst_width,
            self.dst_height,
        );
        Image::from_parts(data, &out_shape, &out_strides, &src.device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvit_device::tests::TestDevice;

    #[test]
    fn test_resize_upsample() {
        let mut dev = TestDevice::default();
        let mut resizer = Resize::new(3, 4);
        let src = &[255, 0, 255, 255];
        let mut img = Image::try_from_slice(src, 1, 2, 2, 1, &mut dev).unwrap();
        let rz_img = resizer.process::<u8, Bilinear, _>(&mut img);
        assert_eq!(
            &rz_img.try_get_data().unwrap(),
            &[255, 128, 0, 255, 160, 64, 255, 223, 191, 255, 255, 255]
        );
    }

    #[test]
    fn test_resize_downsample() {
        let mut dev = TestDevice::default();
        let mut resizer = Resize::new(2, 2);
        let src = &[255, 255, 0, 255, 255, 0, 255, 255, 0, 0, 0, 0];
        let mut img = Image::try_from_slice(src, 1, 3, 4, 1, &mut dev).unwrap();
        let rz_img = resizer.process::<u8, Bilinear, _>(&mut img);
        assert_eq!(&rz_img.try_get_data().unwrap(), &[255, 96, 146, 55]);
    }

    #[test]
    fn test_resize_y_axis_only() {
        let mut dev = TestDevice::default();
        let mut resizer = Resize::new(3, 4);
        let src = &[255, 255, 0, 255, 255, 0, 0, 0, 0];
        let mut img = Image::try_from_slice(src, 1, 3, 3, 1, &mut dev).unwrap();
        let rz_img = resizer.process::<u8, Bilinear, _>(&mut img);
        assert_eq!(
            &rz_img.try_get_data().unwrap(),
            &[255, 255, 0, 255, 255, 0, 159, 159, 0, 0, 0, 0]
        );
    }
}
