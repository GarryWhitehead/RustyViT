#[rvit_testgen::testgen(make_border)]
mod tests {
    use super::*;
    use rvit_core::pixel_traits::Constant;
    use rvit_vision::vision::make_border::MakeBorder;

    #[test]
    fn test_make_border_one_pad() {
        let dev = TestDevice::default();
        let src = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
            .map(|x| num_traits::cast(x).unwrap());
        let mut mb = MakeBorder::new(1);
        let mut src: Image<TestType, _> = Image::try_from_slice(&src, 1, 4, 4, 1, &dev).unwrap();
        let dst = mb.process::<Constant, _, _>(&mut src);
        approx::assert_eq(
            &dst.try_get_data().unwrap(),
            &[
                0, 0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 0, 0, 5, 6, 7, 8, 0, 0, 9, 10, 11, 12, 0, 0, 13,
                14, 15, 16, 0, 0, 0, 0, 0, 0, 0,
            ],
        );
    }

    #[test]
    fn test_make_border_none_power_two_dims() {
        let mut dev = TestDevice::default();
        let src = [1, 2, 3, 4, 5, 6, 7, 8, 9].map(|x| num_traits::cast(x).unwrap());
        let mut mb = MakeBorder::new(1);
        let mut src: Image<TestType, _> =
            Image::try_from_slice(&src, 1, 3, 3, 1, &mut dev).unwrap();
        let dst = mb.process::<Constant, _, _>(&mut src);
        approx::assert_eq(
            &dst.try_get_data().unwrap(),
            &[
                0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0,
            ],
        );
    }

    #[test]
    fn test_make_border_batched() {
        let mut dev = TestDevice::default();
        let (b, c, w, h) = (20, 3, 3, 3);
        let template = &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|x| num_traits::cast(x).unwrap());
        let mut src = vec![TestType::zero(); b * c * w * h];
        src.chunks_mut(w * h).for_each(|p| {
            p.copy_from_slice(template);
        });
        let mut img: Image<TestType, _> =
            Image::try_from_slice(&src, 20, 3, 3, 3, &mut dev).unwrap();
        let mut mb = MakeBorder::new(1);
        let mb_img = mb.process::<Constant, _, _>(&mut img);
        let dst = mb_img.try_get_data().unwrap();
        dst.chunks(mb_img.width * mb_img.height).for_each(|p| {
            approx::assert_eq(
                &p,
                &[
                    0, 0, 0, 0, 0, 0, 1, 2, 3, 0, 0, 4, 5, 6, 0, 0, 7, 8, 9, 0, 0, 0, 0, 0, 0,
                ],
            );
        })
    }
}
