pub(crate) fn inner_shape(lhs_shape: &[usize], rhs_shape: &[usize]) -> (usize, usize, usize) {
    let (m, k, n) = match lhs_shape.len() {
        2 => (lhs_shape[0], lhs_shape[1], rhs_shape[1]),
        3 => (lhs_shape[1], lhs_shape[2], rhs_shape[2]),
        4 => (lhs_shape[2], lhs_shape[3], rhs_shape[3]),
        _ => panic!("Unsupported shape dimension"),
    };
    (m, k, n)
}

pub(crate) fn compute_shape(lhs_shape: &[usize], m: usize, n: usize) -> Vec<usize> {
    let out_shape: Vec<usize> = match lhs_shape.len() {
        2 => vec![m, n],
        3 => vec![lhs_shape[0], m, n],
        4 => vec![lhs_shape[0], lhs_shape[1], m, n],
        _ => panic!("Unsupported shape dimension"),
    };
    out_shape
}
