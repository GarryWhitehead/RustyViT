use crate::element_traits::Elem;
use num_traits::Float;

pub fn assert_eq<L: Elem + bytemuck::Pod + num::NumCast, R: Elem + bytemuck::Pod + num::NumCast>(
    lhs: &[L],
    rhs: &[R],
) {
    let rhs_c: Vec<_> = rhs.iter().map(|x| num_traits::cast(*x).unwrap()).collect();

    let mut msg = String::new();
    let mut failed = 0;
    for (a, b) in lhs.iter().zip(rhs_c.iter()) {
        if a.ne(b) {
            failed += 1;
        }
    }
    if failed > 0 {
        msg += format!(
            "Tests failed. Number of failures: {}\nlhs: {:?} !=\nrhs: {:?}",
            failed, lhs, rhs_c
        )
        .as_str();
    }
    if !msg.is_empty() {
        panic!("{}", msg);
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Tolerance<T: Elem> {
    pub absolute: T,
    pub relative: T,
}

impl<T: Elem> Default for Tolerance<T> {
    fn default() -> Self {
        Self {
            absolute: T::from(0.005).unwrap(),
            relative: T::from(1e-5).unwrap(),
        }
    }
}

impl<T: Elem> Tolerance<T> {
    pub fn set_relative(&mut self, relative: f32) -> Self {
        self.relative = T::from(relative).unwrap();
        self.clone()
    }
}

pub fn assert_approx_eq<L: Elem + Float, R: Elem + Copy>(lhs: &[L], rhs: &[R], tol: Tolerance<L>) {
    let rhs_c: Vec<_> = rhs
        .iter()
        .map(|x| num_traits::cast::<R, L>(*x).unwrap())
        .collect();

    let mut msg = String::new();
    let mut failed = 0;
    for (a, b) in lhs.iter().zip(rhs_c.iter()) {
        if (a.is_nan() && b.is_nan()) || (a.is_infinite() && b.is_infinite()) {
            continue;
        }

        let diff = (*a - *b).abs();
        let max = L::max(a.abs(), b.abs());
        if diff >= tol.absolute.max(tol.relative * max) {
            failed += 1;
        }
    }
    if failed > 0 {
        msg += format!(
            "Tests failed. Number of failures: {}\nlhs: {:?} !=\nrhs: {:?}",
            failed, lhs, rhs_c
        )
        .as_str();
    }
    if !msg.is_empty() {
        panic!("{}", msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_assert_eq() {
        let data_a = [4.0, 3.0, 1.0];
        let data_b = [4.0, 3.0, 1.0];
        assert_eq(&data_a, &data_b);
    }

    #[test]
    fn test_assert_eq_diff_types() {
        let data_a = [4, 3, 1];
        let data_b = [4.0, 3.0, 1.0];
        assert_eq(&data_a, &data_b);
    }

    #[test]
    #[should_panic]
    fn test_assert_eq_ne() {
        let data_a = [2.0, 1.0, 1.0];
        let data_b = [4.0, 3.0, 1.0];
        assert_eq(&data_a, &data_b);
    }

    #[test]
    fn test_assert_approx_eq() {
        let data_a = [3.00001, 4.1222, 1.0];
        let data_b = [3.0, 4.1223, 1.0];
        assert_approx_eq(&data_a, &data_b, Tolerance::default());
    }

    #[test]
    #[should_panic]
    fn test_assert_approx_eq_ne() {
        let data_a = [3.1, 4.1, 1.0];
        let data_b = [3.0, 4.1, 1.0];
        assert_approx_eq(&data_a, &data_b, Tolerance::default());
    }
}
