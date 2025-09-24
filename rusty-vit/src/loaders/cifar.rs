use std::error::Error;
use std::fs::File;
use std::io::{BufReader, ErrorKind, Read, Seek, SeekFrom};

pub(crate) trait DataSetBytes {
    fn from_bytes(v: &[u8]) -> Self;
}

impl DataSetBytes for u8 {
    fn from_bytes(v: &[u8]) -> Self {
        v[0]
    }
}

impl DataSetBytes for f32 {
    fn from_bytes(v: &[u8]) -> Self {
        f32::from_le_bytes(v.try_into().unwrap())
    }
}

impl DataSetBytes for u16 {
    fn from_bytes(v: &[u8]) -> Self {
        u16::from_le_bytes(v.try_into().unwrap())
    }
}

fn read_buffer_bytes<T: DataSetBytes>(
    f: &mut BufReader<File>,
    stride: usize,
    out: &mut [T],
) -> Result<(), Box<dyn Error>> {
    let mut buffer = vec![0u8; size_of::<T>()];
    for i in 0..stride {
        let res = f.read_exact(&mut buffer);
        match res {
            Err(err) => {
                if err.kind() == ErrorKind::UnexpectedEof {
                    break;
                }
            }
            _ => {}
        };
        out[i] = T::from_bytes(&buffer);
    }
    Ok(())
}

pub struct Cifar10Loader {}
impl super::DataSetFormat for Cifar10Loader {
    type Type = u8;

    const IMAGE_FORMAT_TOTAL_IMAGE_SIZE: usize = 10000 * 5;
    const IMAGE_FORMAT_IMAGES_PER_FILE: usize = 10000;
    const IMAGE_FORMAT_DIM: usize = 32;
    const IMAGE_FORMAT_CHANNELS: usize = 3;
    const IMAGE_FORMAT_LABEL_SIZE: usize = 1;

    fn get_training_file(idx: usize) -> &'static str {
        let files = [
            "data_batch_1.bin",
            "data_batch_2.bin",
            "data_batch_3.bin",
            "data_batch_4.bin",
            "data_batch_5.bin",
        ];
        if idx >= files.len() {
            panic!("File index out of bounds");
        }
        &files[idx]
    }

    fn get_test_file(idx: usize) -> &'static str {
        let files = ["data_batch.bin"];
        if idx >= files.len() {
            panic!("File index out of bounds");
        }
        &files[idx]
    }

    fn read_bytes_from_buffer(
        file: &str,
        image_idx: usize,
        out: &mut [Self::Type],
    ) -> Result<u8, Box<dyn Error>> {
        let stride = Self::IMAGE_FORMAT_DIM * Self::IMAGE_FORMAT_DIM * Self::IMAGE_FORMAT_CHANNELS;
        let chunk_size = stride + Self::IMAGE_FORMAT_LABEL_SIZE;

        let mut f = BufReader::new(File::open(file)?);
        f.seek(SeekFrom::Start((image_idx * chunk_size) as u64))?;
        let mut label = [0_u8];
        f.read_exact(&mut label)?;
        let label = label[0];

        read_buffer_bytes(&mut f, stride, out)?;

        Ok(label)
    }
}
