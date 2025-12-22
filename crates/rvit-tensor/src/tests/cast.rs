#[rvit_testgen::testgen(cast)]
mod tests {
    use super::*;

    #[test]
    fn test_cast_from_f32() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let data_a: Vec<f32> = (0u8..100u8).map(|x| f32::from(x)).collect();
        let a = FloatTensor::from_array(&[10, 10], &data_a, &dev.storage).cast::<TestType>();
        approx::assert_approx_eq(&a.into_vec(), &data_a, Default::default());
    }
}
