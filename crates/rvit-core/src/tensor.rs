pub fn tensor_size(shape: &[usize]) -> usize {
    shape.iter().copied().reduce(|a, b| a * b).unwrap()
}

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let dims = shape.len();
    let mut strides = vec![1; dims];
    for i in (0..(dims - 1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
