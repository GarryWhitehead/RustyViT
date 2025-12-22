#[rvit_testgen::testgen(convolution2d)]
mod tests {
    use super::*;
    use rvit_tensor::convolution::{permute_nchw_to_nhwc, permute_nhwc_to_nchw};
    use rvit_tensor::ops::convolution::Conv2d;

    #[test]
    fn test_permute_nchw_to_nhwc() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let t = FloatTensor::new(&[3, 5, 4, 4], &dev.storage);
        let pt = permute_nchw_to_nhwc(t);
        assert_eq!(pt.shape, [3, 4, 4, 5]);
        assert_eq!(pt.strides, [80, 4, 1, 16]);

        let pt = permute_nhwc_to_nchw(pt);
        assert_eq!(pt.shape, [3, 5, 4, 4]);
        assert_eq!(pt.strides, [80, 16, 4, 1]);
    }

    #[test]
    fn test_to_nhwc() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let data = [[
            [[2.0, 3.0, 4.0], [1.0, 2.0, 3.0]],
            [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
            [[2.0, 3.0, 4.0], [5.0, 6.0, 7.0]],
        ]];
        let mut t = FloatTensor::from_data_shape4(&data, &dev.storage);
        let p = Conv2d::default();
        let _conv = p.init(&t, &mut dev).unwrap();
        let t = t.nchw_to_nhwc();
        approx::assert_eq(
            &t.into_vec(),
            &[
                2.0, 4.0, 2.0, 3.0, 5.0, 3.0, 4.0, 6.0, 4.0, 1.0, 7.0, 5.0, 2.0, 8.0, 6.0, 3.0,
                9.0, 7.0,
            ],
        );
    }

    #[test]
    fn test_to_nchw() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let data = [[
            [[2.0, 4.0, 2.0], [3.0, 5.0, 3.0], [4.0, 6.0, 4.0]],
            [[1.0, 7.0, 5.0], [2.0, 8.0, 6.0], [3.0, 9.0, 7.0]],
        ]];
        let mut t = FloatTensor::from_data_convert_shape4(&data, &dev.storage);
        let p = Conv2d::default();
        let _conv = p.init(&t, &mut dev).unwrap();
        let t = t.nhwc_to_nchw();
        approx::assert_eq(
            &t.into_vec(),
            &[
                2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 2.0, 3.0, 4.0, 5.0,
                6.0, 7.0,
            ],
        );
    }

    #[test]
    fn test_im2col_nchw() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        #[rustfmt::skip]
        let data = [[
            [[2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0]],

            [[4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]],

            [[2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]],
        ]];
        let mut t = FloatTensor::from_data_shape4(&data, &dev);
        let p = Conv2d {
            out_channels: 3,
            ..Default::default()
        };
        let conv = p.init(&t, &mut dev).unwrap();
        let v = t.im2col(&conv);
        let res = v.into_vec();
        assert_eq!(res.len(), 24); // B * OC * KH * KW * OH * OW;
        approx::assert_eq(
            &res,
            &[
                2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0,
                2.0, 3.0, 5.0, 6.0, 3.0, 4.0, 6.0, 7.0,
            ],
        );
    }

    #[test]
    fn test_im2col_nchw_padding() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        #[rustfmt::skip]
        let data = [[
            [[2.0, 3.0, 4.0],
            [1.0, 2.0, 3.0]],

           [[4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]],

            [[2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]],
        ]];
        let mut t = FloatTensor::from_data_shape4(&data, &dev.storage);
        let p = Conv2d {
            out_channels: 3,
            padding: 1,
            ..Default::default()
        };
        let conv = p.init(&t, &mut dev).unwrap();
        let v = t.im2col(&conv);
        let res = v.into_vec();
        assert_eq!(res.len(), 144); // B * OC * KH * KW * OH * OW;
        approx::assert_eq(
            &res,
            &[
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0,
                0.0, 2.0, 0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 0.0, 3.0, 0.0,
                0.0, 1.0, 0.0, 0.0, 1.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 4.0, 5.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 6.0, 0.0,
                0.0, 4.0, 0.0, 7.0, 4.0, 5.0, 7.0, 8.0, 5.0, 6.0, 8.0, 9.0, 6.0, 0.0, 9.0, 0.0,
                0.0, 7.0, 0.0, 0.0, 7.0, 8.0, 0.0, 0.0, 8.0, 9.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 4.0, 0.0,
                0.0, 2.0, 0.0, 5.0, 2.0, 3.0, 5.0, 6.0, 3.0, 4.0, 6.0, 7.0, 4.0, 0.0, 7.0, 0.0,
                0.0, 5.0, 0.0, 0.0, 5.0, 6.0, 0.0, 0.0, 6.0, 7.0, 0.0, 0.0, 7.0, 0.0, 0.0, 0.0,
            ],
        );
    }

    #[test]
    fn test_im2col_nchw_stride() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        #[rustfmt::skip]
        let data = [[
            [[3.0, 5.0, 1.0, 2.0],
            [4.0, 2.0, 1.0, 9.0]],

            [[3.0, 3.0, 8.0, 3.0],
            [4.0, 2.0, 1.0, 3.0]],

            [[6.0, 5.0, 7.0, 3.0],
            [2.0, 2.0, 1.0, 0.0]],
        ]];
        let mut t = FloatTensor::from_data_shape4(&data, &dev.storage);
        let p = Conv2d {
            out_channels: 3,
            stride: 2,
            ..Default::default()
        };
        let conv = p.init(&t, &mut dev).unwrap();
        let v = t.im2col(&conv);
        let res = v.into_vec();
        assert_eq!(res.len(), 24); // B * OC * KH * KW * OH * OW;
        approx::assert_eq(
            &res,
            &[
                3.0, 5.0, 4.0, 2.0, 1.0, 2.0, 1.0, 9.0, 3.0, 3.0, 4.0, 2.0, 8.0, 3.0, 1.0, 3.0,
                6.0, 5.0, 2.0, 2.0, 7.0, 3.0, 1.0, 0.0,
            ],
        );
    }

    #[test]
    fn test_conv2d() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let weight = [-0.049, -0.43, 0.019, 0.097];
        let data = [[[[-0.867, 0.527, -0.952], [-0.645, 0.778, -0.490]]]];
        let t = FloatTensor::from_data_shape4(&data, &dev.storage);
        let p = Conv2d::default()
            .init(&t, &mut dev)
            .unwrap()
            .update_weights(&weight);
        let t = t.conv2d(&p);
        approx::assert_approx_eq(&t.into_vec(), &[-0.120916, 0.350789], Default::default());
    }

    #[test]
    fn test_conv2d_with_padding() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let weight = [-0.049, -0.43, 0.019, 0.097];
        let data = [[[[-0.867, 0.527, -0.952], [-0.645, 0.778, -0.490]]]];
        let t = FloatTensor::from_data_shape4(&data, &dev.storage);
        let p = Conv2d {
            padding: 1,
            ..Default::default()
        }
        .init(&t, &mut dev)
        .unwrap()
        .update_weights(&weight);
        let t = t.conv2d(&p);
        #[rustfmt::skip]
        approx::assert_approx_eq(
            &t.into_vec(),
            &[
                -0.084099, 0.034646004, -0.082331, -0.018088,
                0.310245, -0.120916, 0.35078904, 0.037338,
                0.27735, -0.302935, 0.172578, 0.02401
            ],
            Default::default()
        );
    }

    #[test]
    fn test_conv2d_with_stride() {
        let mut dev: DeviceRuntime<_, TestType, i32> = Runtime::new(Default::default());
        let weight = [0.67, 0.13, -0.002, 0.001];
        let data = [
            [[[0.563, 0.112, 0.812, -0.445], [-0.999, -0.02, 0.147, 0.671]]],
            [[
                [-0.310, -0.022, 0.136, -0.111],
                [0.527, 0.007, 0.317, -0.094],
            ]],
        ];
        let t = FloatTensor::from_data_shape4(&data, &dev.storage);
        let p = Conv2d {
            stride: 2,
            ..Default::default()
        }
        .init(&t, &mut dev)
        .unwrap()
        .update_weights(&weight);
        let t = t.conv2d(&p);
        #[rustfmt::skip]
        approx::assert_approx_eq(
            &t.into_vec(),
            &[
                0.39374804, 0.48656702, -0.21160701, 0.07596201
            ],
            Default::default()
        );
    }
}
