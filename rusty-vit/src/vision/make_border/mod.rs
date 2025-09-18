mod make_border_cpu;
#[cfg(feature = "cuda")]
mod make_border_cu;

use crate::device::DeviceStorage;
use crate::device::cpu::Cpu;
use crate::image::PixelType;
use crate::vision::Image;

pub trait MakeBorderKernel<T, I>: DeviceStorage<T> {
    fn make_border(&mut self, src: &Image<T, Self>, padding: usize, fill_value: T) -> Self::Vec
    where
        Self: Sized;
}

pub(crate) trait BorderMode: Clone + Copy {}
#[derive(Debug, Clone, Copy, Default)]
struct Constant {}
impl BorderMode for Constant {}

#[derive(Debug, Clone, Copy, Default)]
struct ClampToEdge {}
impl BorderMode for ClampToEdge {}

#[derive(Debug, Clone, Copy, Default)]
struct Mirror {}
impl BorderMode for Mirror {}

struct MakeBorder<T: PixelType> {
    padding: usize,
    fill_value: T,
}

impl<T: PixelType> MakeBorder<T> {
    fn new(padding: usize, fill_value: T) -> Self {
        Self {
            padding,
            fill_value,
        }
    }
}

impl<T: PixelType> MakeBorder<T> {
    pub fn process<B: BorderMode, S: MakeBorderKernel<T, B>>(
        &self,
        src: &mut Image<T, S>,
    ) -> S::Vec {
        if 2 * self.padding >= src.width || 2 * self.padding >= src.height {
            panic!("Padding size must be less than the image dimensions");
        }
        let dev = &mut src.device.clone();
        dev.make_border(src, self.padding, self.fill_value)
    }
}

mod tests {
    use super::*;

    #[test]
    fn test_make_border_one_pad() {
        let dev = Cpu::default();
        let mb = MakeBorder::new(1, 0);
        let mut src: Image<u8, Cpu> = Image::try_from_slice(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            1,
            4,
            4,
            1,
            &dev,
        )
        .unwrap();
        let dst = mb.process::<Constant, _>(&mut src);
        assert_eq!(
            &dst.data,
            &[0, 0, 0, 0, 0, 6, 7, 0, 0, 10, 11, 0, 0, 0, 0, 0]
        );
    }

    #[test]
    fn test_make_border_none_power_two_dims() {
        let dev = Cpu::default();
        let mb = MakeBorder::new(1, 0);
        let mut src: Image<u8, Cpu> = Image::try_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 1, 3, 3, 1, &dev).unwrap();
        let dst = mb.process::<Constant, _>(&mut src);
        assert_eq!(&dst.data, &[0, 0, 0, 0, 5, 0, 0, 0, 0]);
    }
}
