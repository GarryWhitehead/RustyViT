use image::{EncodableLayout, ImageReader};
#[cfg(feature = "cuda")]
use rusty_vit::device::cuda::Cuda;
use rusty_vit::device::vulkan::Vulkan;
use rusty_vit::image::Image;
use rusty_vit::vision::sep_filters::GaussianBlur;
use rusty_vk::public_types::DeviceType;
use show_image::event;

fn image_to_planar(src: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
    if channels > 4 {
        panic!("Four channels (i.e. RGBA) is the most channels supported.");
    }
    let mut out = vec![0u8; width * height * channels];
    for row in 0..height {
        for col in 0..width {
            for c in 0..channels {
                out[c * width * height + row * width + col] =
                    src[c + col * channels + channels * width * row];
            }
        }
    }
    out
}

fn image_to_interleaved(src: &[u8], width: usize, height: usize, channels: usize) -> Vec<u8> {
    if channels > 4 {
        panic!("Four channels (i.e. RGBA) is the most channels supported.");
    }
    let mut out = vec![0u8; width * height * channels];
    for row in 0..height {
        for col in 0..width {
            for c in 0..channels {
                out[c + col * channels + channels * width * row] =
                    src[c * width * height + row * width + col];
            }
        }
    }
    out
}

#[show_image::main]
fn main() {
    let img = ImageReader::open(format!(
        "{}/../../assets/bike.JPG",
        env!("CARGO_MANIFEST_DIR")
    ))
    .unwrap()
    .decode()
    .unwrap();
    let p_img = image_to_planar(
        img.as_rgb8().unwrap().as_bytes(),
        img.width() as usize,
        img.height() as usize,
        3,
    );

    //let dev = Cpu::default();
    //let dev = Cuda::try_new(0).unwrap();
    let mut dev = Vulkan::new(DeviceType::DiscreteGpu).unwrap();
    let mut conv: GaussianBlur<f32, u8, _> = GaussianBlur::try_new(1.0, 9, &dev).unwrap();
    //let flipper = RandomFlipHorizontal::new(0.9);
    let mut image = Image::try_from_slice(
        &p_img,
        1,
        img.width() as usize,
        img.height() as usize,
        3,
        &dev,
    )
    .unwrap();
    conv.process(&mut image, &mut dev);
    //flipper.flip(&mut image, &mut dev);

    let i_img = image_to_interleaved(
        &image.try_get_data().unwrap(),
        img.width() as usize,
        img.height() as usize,
        3,
    );

    let d_img = show_image::ImageView::new(
        show_image::ImageInfo::rgb8(img.width(), img.height()),
        &i_img,
    );
    let win = show_image::create_window(
        "Image processing demo",
        show_image::WindowOptions::default(),
    )
    .unwrap();

    win.set_image("demo", d_img).unwrap();

    for event in win.event_channel().unwrap() {
        if let event::WindowEvent::KeyboardInput(event) = &event {
            if event.input.key_code == Some(event::VirtualKeyCode::Escape)
                && event.input.state.is_pressed()
            {
                break;
            }
        }
        if let event::WindowEvent::CloseRequested(_event) = &event {
            break;
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_to_planar() {
        let src = &[
            1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
        ];
        let res = crate::image_to_planar(src, 3, 3, 3);
        assert_eq!(
            &res,
            &[
                1, 1, 1, 4, 4, 4, 7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8, 3, 3, 3, 6, 6, 6, 9, 9, 9
            ]
        );
    }

    #[test]
    fn test_to_interleaved() {
        let src = &[
            1, 1, 1, 4, 4, 4, 7, 7, 7, 2, 2, 2, 5, 5, 5, 8, 8, 8, 3, 3, 3, 6, 6, 6, 9, 9, 9,
        ];
        let res = crate::image_to_interleaved(src, 3, 3, 3);
        assert_eq!(
            &res,
            &[
                1, 2, 3, 1, 2, 3, 1, 2, 3, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 8, 9, 7, 8, 9, 7, 8, 9,
            ]
        );
    }
}
