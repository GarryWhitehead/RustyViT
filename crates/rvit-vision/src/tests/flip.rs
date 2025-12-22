#[rvit_testgen::testgen(flip)]
mod tests {
    use super::*;
    use rvit_vision::vision::flip::RandomFlipHorizontal;

    #[test]
    fn test_flip_horizontal() {
        let src =
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4].map(|x| num_traits::cast(x).unwrap());
        let mut dev = TestDevice::default();
        let mut flipper = RandomFlipHorizontal::new(2.0);
        let mut img: Image<TestType, _> = Image::try_from_slice(&src, 1, 4, 4, 1, &dev).unwrap();
        flipper.process(&mut img);
        approx::assert_eq(
            &img.try_get_data().unwrap(),
            &[4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1],
        );
    }

    #[test]
    fn test_flip_horizontal_batch() {
        let (b, c, w, h) = (10usize, 3, 4, 4);
        let template =
            [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4].map(|x| num_traits::cast(x).unwrap());
        let mut src = vec![TestType::zero(); b * c * w * h];
        src.chunks_mut(w * h)
            .for_each(|chunk| chunk.copy_from_slice(&template));

        let mut dev = TestDevice::default();
        let mut flipper = RandomFlipHorizontal::new(2.0);
        let mut img: Image<TestType, _> = Image::try_from_slice(&src, b, w, h, c, &dev).unwrap();
        flipper.process(&mut img);
        let dst = img.try_get_data().unwrap();
        dst.chunks(w * h).for_each(|chunk| {
            approx::assert_eq(&chunk, &[4, 4, 4, 4, 3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1]);
        })
    }

    /*#[test]
    #[cfg(not(feature = "cuda"))]
    fn test_flip_vertical() {
        let src: Vec<u8> = vec![1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4];
        let mut dst = vec![0u8; src.len()];
        flip_vertical(&src, 4, 4, &mut dst);
        assert_eq!(&dst, &[4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1, 4, 3, 2, 1]);
    }*/
}
