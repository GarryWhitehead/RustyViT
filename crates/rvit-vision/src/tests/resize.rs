#[rvit_testgen::testgen(resize)]
mod tests {
    use super::*;
    use rvit_core::pixel_traits::Bilinear;
    use rvit_vision::vision::resize::Resize;

    #[test]
    fn test_resize_upsample() {
        let mut dev = TestDevice::default();
        let mut resizer = Resize::new(3, 4);
        let src = &[255, 0, 255, 255].map(|x| num_traits::cast(x).unwrap());
        let mut img = Image::try_from_slice(src, 1, 2, 2, 1, &mut dev).unwrap();
        let rz_img = resizer.process::<TestType, Bilinear, _>(&mut img);
        approx::assert_eq(
            &rz_img.try_get_data().unwrap(),
            &[255, 128, 0, 255, 160, 64, 255, 223, 191, 255, 255, 255],
        );
    }

    #[test]
    fn test_resize_downsample() {
        let mut dev = TestDevice::default();
        let mut resizer = Resize::new(2, 2);
        let src =
            &[255, 255, 0, 255, 255, 0, 255, 255, 0, 0, 0, 0].map(|x| num_traits::cast(x).unwrap());
        let mut img = Image::try_from_slice(src, 1, 3, 4, 1, &mut dev).unwrap();
        let rz_img = resizer.process::<TestType, Bilinear, _>(&mut img);
        approx::assert_eq(&rz_img.try_get_data().unwrap(), &[255, 96, 146, 55]);
    }

    #[test]
    fn test_resize_y_axis_only() {
        let mut dev = TestDevice::default();
        let mut resizer = Resize::new(3, 4);
        let src = &[255, 255, 0, 255, 255, 0, 0, 0, 0].map(|x| num_traits::cast(x).unwrap());
        let mut img = Image::try_from_slice(src, 1, 3, 3, 1, &mut dev).unwrap();
        let rz_img = resizer.process::<TestType, Bilinear, _>(&mut img);
        approx::assert_eq(
            &rz_img.try_get_data().unwrap(),
            &[255, 255, 0, 255, 255, 0, 159, 159, 0, 0, 0, 0],
        );
    }
}
