#[rvit_testgen::testgen(sep_filter_convolution)]
mod tests {
    use super::*;
    use rvit_vision::vision::sep_filters::GaussianBlur;

    #[test]
    fn test_gaussian_blur() {
        let mut dev = TestDevice::default();
        let data = vec![TestType::from(3u8); 32 * 32];
        let mut blur = GaussianBlur::<TestType, _>::try_new(2.0, 3, &mut dev).unwrap();
        let mut img = Image::try_from_slice(&data, 1, 32, 32, 1, &mut dev).unwrap();
        blur.process(&mut img);
        approx::assert_eq(
            &img.try_get_data().unwrap(),
            &[2, 2, 2, 1, 2, 3, 2, 1, 3, 4, 3, 1, 4, 4, 2, 1],
        );
    }
}
