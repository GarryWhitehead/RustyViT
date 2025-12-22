#[rvit_testgen::testgen(crop)]
mod tests {
    use super::*;
    use rvit_vision::vision::crop::RandomCrop;

    #[test]
    fn test_crop() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|x| num_traits::cast(x).unwrap());
        let mut dev = TestDevice::default();
        let mut cropper = RandomCrop::new(3, 3, 3, 3);
        let mut img: Image<TestType, _> = Image::try_from_slice(src, 1, 3, 3, 1, &mut dev).unwrap();
        let dst = cropper.process(&mut img);
        approx::assert_eq(&dst.try_get_data().unwrap(), src);
    }

    #[test]
    fn test_crop_batched() {
        let mut dev = TestDevice::default();
        let (b, c, w, h) = (20, 3, 3, 3);
        let template = &[1, 2, 3, 4, 5, 6, 7, 8, 9].map(|x| num_traits::cast(x).unwrap());
        let mut src = vec![TestType::zero(); b * c * w * h];
        src.chunks_mut(w * h).for_each(|slice| {
            slice.copy_from_slice(template);
        });
        let mut cropper = RandomCrop::new(3, 3, 3, 3);
        let mut img: Image<TestType, _> =
            Image::try_from_slice(&src, b, w, h, c, &mut dev).unwrap();
        let crop_img = cropper.process(&mut img);
        let crop_img = crop_img.try_get_data().unwrap();
        crop_img.chunks(w * h).for_each(|slice| {
            approx::assert_eq(slice, template);
        })
    }
}
/*#[test]
    fn test_crop_1() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0u8; 3 * 3];
        crop(src, 4, 3, 3, 0, 0, &mut dst);
        assert_eq!(dst, &[1, 2, 3, 5, 6, 7, 9, 10, 11]);
    }

    #[test]
    fn test_crop_3() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0u8; 3 * 3];
        crop(src, 4, 3, 3, 1, 1, &mut dst);
        assert_eq!(dst, &[6, 7, 8, 10, 11, 12, 14, 15, 16]);
    }

    #[test]
    fn test_crop_2() {
        let src = &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let mut dst = vec![0u8; 2 * 2];
        crop(src, 4, 2, 2, 2, 2, &mut dst);
        assert_eq!(dst, &[11, 12, 15, 16]);
    }

    #[test]
    fn test_crop_process() {
        let src_image = Image::from_slice(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            4,
            1,
        );
        let cropper = RandomCrop::new(4, 3, 3);
        let out_image = cropper.process(&src_image);
        assert_eq!(out_image.dim, 3);
        assert_eq!(out_image.channels, 1);
    }

    #[test]
    fn test_crop_batched() {
        let src_image = ImageArray::from_slice(
            &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            2,
            1,
            4,
        );
        let mut cropper = RandomCrop::new(2, 2, 2);
        let out_image = cropper.process_batched(&src_image);
        assert_eq!(out_image.dim, 2);
        assert_eq!(out_image.channels, 1);
    }
}*/
