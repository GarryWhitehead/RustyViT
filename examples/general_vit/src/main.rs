/*use clap::Parser;
use rusty_vit::loaders::cifar;
use rusty_vit::tensor::Tensor;
use rusty_vit::vision::convolution::Convolution;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(short, long)]
    data_folder_path: String,
}

struct Embeddings {
    convolution: Convolution,
    patch_size: usize,
    embed_dim: usize,
}

impl Embeddings {
    fn new(patch_size: usize, embed_dim: usize, stride: usize) -> Self {
        let conv = Convolution::new(3, embed_dim, 1, stride, 0, patch_size);
        Self {
            convolution: conv,
            patch_size,
            embed_dim,
        }
    }

    fn embeddings_forward(&self, x: &Matrix) {
        let x = self.convolution.process(x);

        let num_embeddings = self.patch_size + 1;
        let pos_embedding = get_sinusoid_encoding(num_embeddings, self.embed_dim);
    }
}

fn main() {
    let args = Args::parse();
    load_cifar_data(args.data_folder_path.as_str(), 0);
}*/
