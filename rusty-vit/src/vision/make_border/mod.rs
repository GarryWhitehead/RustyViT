mod make_border_cpu;
#[cfg(feature = "cuda")]
mod make_border_cu;
#[cfg(feature = "vulkan")]
mod make_border_vk;

use crate::device::DeviceStorage;
use crate::image::{Image, PixelType};

pub trait MakeBorderKernel<T, I>: DeviceStorage<T> {
    fn make_border(&mut self, src: &Image<T, Self>, padding: usize) -> Image<T, Self>
    where
        Self: Sized;
}

pub trait BorderMode: Clone + Copy {}
#[derive(Debug, Clone, Copy, Default)]
struct Constant {}
impl BorderMode for Constant {}

#[derive(Debug, Clone, Copy, Default)]
struct ClampToEdge {}
impl BorderMode for ClampToEdge {}

#[derive(Debug, Clone, Copy, Default)]
struct Mirror {}
impl BorderMode for Mirror {}

pub struct MakeBorder {
    padding: usize,
}

impl MakeBorder {
    pub fn new(padding: usize) -> Self {
        Self { padding }
    }
}

impl MakeBorder {
    pub fn process<T: PixelType, B: BorderMode, S: MakeBorderKernel<T, B>>(
        &self,
        src: &mut Image<T, S>,
        dev: &mut S,
    ) -> Image<T, S> {
        if 2 * self.padding >= src.width || 2 * self.padding >= src.height {
            panic!("Padding size must be less than the image dimensions");
        }
        dev.make_border(src, self.padding)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_make_border_one_pad() {
        let mut dev = crate::device::cpu::Cpu::default();
        //let dev = Cuda::try_new(0).unwrap();
        //let mut dev = Vulkan::new(DeviceType::DiscreteGpu).unwrap();
        let mb = crate::vision::make_border::MakeBorder::new(1);
        let mut src: crate::image::Image<u8, _> = crate::image::Image::try_from_slice(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            1,
            4,
            4,
            1,
            &dev,
        )
        .unwrap();
        let dst = mb.process::<_, crate::vision::make_border::Constant, _>(&mut src, &mut dev);
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
        let mut dev = crate::device::cpu::Cpu::default();
        //let mut dev = Vulkan::new(DeviceType::DiscreteGpu).unwrap();
        let mb = crate::vision::make_border::MakeBorder::new(1);
        let mut src: crate::image::Image<u8, _> =
            crate::image::Image::try_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 9], 1, 3, 3, 1, &mut dev)
                .unwrap();
        let dst = mb.process::<_, crate::vision::make_border::Constant, _>(&mut src, &mut dev);
        assert_eq!(
            &dst.try_get_data().unwrap(),
            &[
                0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0
            ]
        );
    }

    #[test]
    fn test_make_border_batched() {
        //let dev = Cuda::try_new(0).unwrap();
        let mut dev = crate::device::cpu::Cpu::default();
        //let mut dev = Vulkan::new(DeviceType::DiscreteGpu).unwrap();
        let (b, c, w, h) = (20, 3, 3, 3);
        let template = &[1, 2, 3, 4, 5, 6, 7, 8, 9];
        let mut src = vec![0u8; b * c * w * h];
        src.chunks_mut(w * h).for_each(|p| {
            p.copy_from_slice(template);
        });
        let mut img: crate::image::Image<u8, _> =
            crate::image::Image::try_from_slice(&src, 20, 3, 3, 3, &mut dev).unwrap();
        let mb = crate::vision::make_border::MakeBorder::new(1);
        let mb_img = mb.process::<_, crate::vision::make_border::Constant, _>(&mut img, &mut dev);
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
