use rusty_llm::device::cuda::Cuda;
use rusty_llm::image::Image;
use rusty_llm::vision::flip::RandomFlipHorizontal;

fn main() {
    let dev = Cuda::try_new(0).unwrap();
    let flipper = RandomFlipHorizontal::new(0.5);
    let image = Image::try_new(1, 640, 400, 3, dev).unwrap();
    flipper.flip_horizontal()
}
