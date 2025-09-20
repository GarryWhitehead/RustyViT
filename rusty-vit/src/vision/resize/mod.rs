mod resize_cpu;
#[cfg(feature = "cuda")]
mod resize_cu;

use crate::vision::{Image};
use crate::device::DeviceStorage;
use crate::image::PixelType;

pub(crate) trait InterpMode: Clone + Copy {}
#[derive(Default, Copy, Clone)]
struct Bilinear {}
 impl InterpMode for Bilinear {}

pub(crate) trait ResizeKernel<T, I: InterpMode>: DeviceStorage<T> {
    fn resize(&mut self, src: &mut Image<T, Self>, dst_width: usize, dst_height: usize) -> Image<T, Self>
    where
        Self: Sized;
}

struct Resize {
    dst_width: usize,
    dst_height: usize,
}

impl Resize {
    fn new(
        dst_width: usize,
        dst_height: usize,
    ) -> Self
    {
        if dst_width == 0 || dst_height == 0 {
            panic!("Resized image dimensions must be non-zero");
        }
        Self {
            dst_width,
            dst_height
        }
    }
}

impl Resize {
    fn resize<T: PixelType, I: InterpMode, S: ResizeKernel<T, I>>(&self, image: &mut Image<T, S>) -> Image<T, S> {
        let dev = &mut image.device.clone();
        dev.resize(image, self.dst_width, self.dst_height)
    }
}

mod tests {
    use crate::device::cpu::Cpu;
    use crate::device::cuda::Cuda;
    use super::*;

    #[test]
    fn test_resize_upsample() {
        //let dev = Cpu::default();
        let dev = Cuda::try_new(0).unwrap();
        let resizer = Resize::new(3, 4);
        let src = &[255, 0, 255, 255];
        let mut img = Image::try_from_slice(src, 1, 2, 2, 1, &dev).unwrap();
        let rz_img = resizer.resize::<u8, Bilinear, _>(&mut img);
        assert_eq!(
            &rz_img.try_get_data().unwrap(),
            &[255, 128, 0, 255, 160, 64, 255, 223, 191, 255, 255, 255]
        );
    }

    #[test]
    fn test_resize_downsample() {
        //let dev = Cpu::default();
        let dev = Cuda::try_new(0).unwrap();
        let resizer = Resize::new(2, 2);
        let src = &[255, 255, 0, 255, 255, 0, 255, 255, 0, 0, 0, 0];
        let mut img = Image::try_from_slice(src, 1, 3, 4, 1, &dev).unwrap();
        let rz_img = resizer.resize::<u8, Bilinear, _>(&mut img);
        assert_eq!(
            &rz_img.try_get_data().unwrap(),
            &[255, 96, 146, 55]
        );
    }

    #[test]
    fn test_resize_y_axis_only() {
        let dev = Cpu::default();
        let resizer = Resize::new(3, 4);
        let src = &[255, 255, 0, 255, 255, 0, 0, 0, 0];
        let mut img = Image::try_from_slice(src, 1, 3, 3, 1, &dev).unwrap();
        let rz_img = resizer.resize::<u8, Bilinear, _>(&mut img);
        assert_eq!(
            &rz_img.try_get_data().unwrap(),
            &[255, 255, 0, 255, 255, 0, 159, 159, 0, 0, 0, 0]
        );
    }
}

/*#[test]
   fn test_transpose() {
       let input = &mut [255, 127, 68, 34, 228, 189, 45, 6, 0, 4, 21, 90];
       let dst = Resize::transpose(input, 4, 3);
       assert_eq!(dst, &[255, 228, 0, 127, 189, 4, 68, 45, 21, 34, 6, 90]);
   }*/


