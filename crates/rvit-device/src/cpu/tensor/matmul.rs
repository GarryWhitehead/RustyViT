use crate::cpu::device::{Cpu, VecPool};
use crate::op_traits::MatMulKernel;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::*;
use rvit_core::type_traits::FloatType;
use std::ops::Deref;

pub trait MatMul<T: FloatType> {
    #[allow(clippy::too_many_arguments)]
    fn gemm_matmul(
        m: usize,
        k: usize,
        n: usize,
        lhs: &[T],
        lhs_stride: [usize; 2],
        rhs: &[T],
        rhs_stride: [usize; 2],
        dst: &mut [T],
        dst_stride: [usize; 2],
    );
}

impl MatMul<f32> for Cpu {
    #[allow(clippy::too_many_arguments)]
    fn gemm_matmul(
        m: usize,
        k: usize,
        n: usize,
        lhs: &[f32],
        lhs_stride: [usize; 2],
        rhs: &[f32],
        rhs_stride: [usize; 2],
        dst: &mut [f32],
        dst_stride: [usize; 2],
    ) {
        unsafe {
            gemm::gemm(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                dst_stride[1] as isize,
                dst_stride[0] as isize,
                false, // For accumulation (if alpha > 0) - not supported at present
                lhs.as_ptr(),
                lhs_stride[1] as isize,
                lhs_stride[0] as isize,
                rhs.as_ptr(),
                rhs_stride[1] as isize,
                rhs_stride[0] as isize,
                0.0f32,
                1.0f32,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(rayon::current_num_threads()),
            )
        }
    }
}

impl MatMul<half::f16> for Cpu {
    #[allow(clippy::too_many_arguments)]
    fn gemm_matmul(
        m: usize,
        k: usize,
        n: usize,
        lhs: &[half::f16],
        lhs_stride: [usize; 2],
        rhs: &[half::f16],
        rhs_stride: [usize; 2],
        dst: &mut [half::f16],
        dst_stride: [usize; 2],
    ) {
        unsafe {
            gemm::gemm::<half::f16>(
                m,
                n,
                k,
                dst.as_mut_ptr(),
                dst_stride[1] as isize,
                dst_stride[0] as isize,
                false, // For accumulation (if alpha > 0) - not supported at present
                lhs.as_ptr(),
                lhs_stride[1] as isize,
                lhs_stride[0] as isize,
                rhs.as_ptr(),
                rhs_stride[1] as isize,
                rhs_stride[0] as isize,
                half::f16::ZERO,
                half::f16::ONE,
                false,
                false,
                false,
                gemm::Parallelism::Rayon(rayon::current_num_threads()),
            )
        }
    }
}

impl<T: FloatType> MatMulKernel<T> for Cpu
where
    Self: MatMul<T>,
    Self: DeviceStorage<T>,
{
    fn forward(
        &self,
        lhs: &Self::Vec,
        lhs_shape: &[usize],
        lhs_strides: &[usize],
        rhs: &Self::Vec,
        rhs_shape: &[usize],
        rhs_strides: &[usize],
    ) -> (Self::Vec, Vec<usize>, Vec<usize>) {
        let dim = lhs_shape.len();
        let (m, k, n) = crate::tensor::inner_shape(&lhs_shape, &rhs_shape);
        let out_shape = crate::tensor::compute_shape(&lhs_shape, m, n);
        let out_sz = tensor_size(&out_shape);

        let mut out = Self::try_alloc(self, out_sz).unwrap();
        let out_strides = compute_strides(&out_shape);

        // Check for batched matrix multiply (four dimensions).
        if dim == 4 {
            let bsize = lhs_shape[0];
            let csize = lhs_shape[1];
            for b in 0..bsize {
                let a_base = b * lhs_strides[0];
                let b_base = b * rhs_strides[0];
                let c_base = b * out_strides[0];
                for c in 0..csize {
                    let a_start = a_base + c * lhs_strides[1];
                    let a_end = a_start + lhs_strides[1];
                    let a_slice = &lhs[a_start..a_end];

                    let b_start = b_base + c * rhs_strides[1];
                    let b_end = b_start + rhs_strides[1];
                    let b_slice = &rhs[b_start..b_end];

                    let c_start = c_base + c * out_strides[1];
                    let c_end = c_start + out_strides[1];
                    let c_slice = &mut out[c_start..c_end];

                    Self::gemm_matmul(
                        m,
                        k,
                        n,
                        a_slice,
                        [lhs_strides[2], lhs_strides[3]],
                        b_slice,
                        [rhs_strides[2], rhs_strides[3]],
                        c_slice,
                        [out_strides[2], out_strides[3]],
                    );
                }
            }
        }
        // Check for batched matrix multiply (three dimensions).
        else if dim == 3 {
            let bsize = lhs_shape[0];
            for b in 0..bsize {
                let a_start = b * lhs_strides[0];
                let a_end = a_start + lhs_strides[0];
                let a_slice = &lhs[a_start..a_end];

                let b_start = b * rhs_strides[0];
                let b_end = b_start + rhs_strides[0];
                let b_slice = &rhs[b_start..b_end];

                let c_start = b * out_strides[0];
                let c_end = c_start + out_strides[0];
                let c_slice = &mut out[c_start..c_end];

                Self::gemm_matmul(
                    m,
                    k,
                    n,
                    a_slice,
                    [lhs_strides[1], lhs_strides[2]],
                    b_slice,
                    [rhs_strides[1], rhs_strides[2]],
                    c_slice,
                    [out_strides[1], out_strides[2]],
                );
            }
        }
        // Matrix-matrix multiplication
        else {
            Self::gemm_matmul(
                m,
                k,
                n,
                Self::slice(lhs),
                [lhs_strides[0], lhs_strides[1]],
                Self::slice(rhs),
                [rhs_strides[0], rhs_strides[1]],
                Self::slice_mut(&mut out),
                [out_strides[0], out_strides[1]],
            );
        }
        (out, out_shape, out_strides)
    }
}
