use num::Zero;
use std::ops::AddAssign;

pub fn transpose<TYPE>(input: &[TYPE], width: usize, height: usize) -> Vec<TYPE>
where
    TYPE: Zero + Clone + Copy,
{
    assert_eq!(input.len(), width * height);
    let mut dst = vec![TYPE::zero(); width * height];
    for (n, d) in dst.iter_mut().enumerate() {
        let i = n / height;
        let j = n % height;
        *d = input[width * j + i];
    }
    dst
}

pub fn transpose_block<TYPE, const BLOCK_SIZE: usize>(
    input: &[TYPE],
    width: usize,
    height: usize,
) -> Vec<TYPE>
where
    TYPE: Zero + Clone + Copy,
{
    assert_eq!(input.len(), width * height);
    let mut dst = vec![TYPE::zero(); width * height];
    for y in (0..height).step_by(BLOCK_SIZE) {
        for x in (0..width).step_by(BLOCK_SIZE) {
            for yy in y..y + BLOCK_SIZE {
                for xx in x..x + BLOCK_SIZE {
                    dst[xx + yy * width] = input[yy + xx * width];
                }
            }
        }
    }
    dst
}

pub fn add<TYPE>(m1: &mut [TYPE], m2: &[TYPE], width: usize, height: usize)
where
    TYPE: Zero + Clone + Copy + AddAssign,
{
    assert_eq!(m1.len(), width * height);
    assert_eq!(m2.len(), width * height);
    for y in 0..height {
        for x in 0..width {
            m1[x + y * width] += m2[x + y * width];
        }
    }
}

pub fn add_block<TYPE, const BLOCK_SIZE: usize>(
    m1: &mut [TYPE],
    m2: &[TYPE],
    width: usize,
    height: usize,
) where
    TYPE: Zero + Clone + Copy + AddAssign,
{
    assert_eq!(m1.len(), width * height);
    assert_eq!(m2.len(), width * height);
    //let dst = vec![TYPE::zero(); width * height];
    for y in (0..height).step_by(BLOCK_SIZE) {
        for x in (0..width).step_by(BLOCK_SIZE) {
            for yy in y..y + BLOCK_SIZE {
                for xx in x..x + BLOCK_SIZE {
                    m1[xx + yy * width] += m2[xx + yy * width];
                }
            }
        }
    }
}
