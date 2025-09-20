use rusty_vit::device::cuda::Cuda;
use rusty_vit::image::Image;
use rusty_vit::vision::flip::RandomFlipHorizontal;

fn main() {
    let dev = Cuda::try_new(0).unwrap();
    let flipper = RandomFlipHorizontal::new(0.5);
    let mut image = Image::try_new(1, 640, 400, 3, &dev).unwrap();
    flipper.flip(&mut image);
}
