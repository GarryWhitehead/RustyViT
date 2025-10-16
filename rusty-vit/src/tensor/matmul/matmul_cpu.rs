use crate::device::cpu::Cpu;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

impl Cpu {
    #[allow(clippy::too_many_arguments)]
    fn gemm_matmul<T: FloatType>(
        &self,
        m: usize,
        k: usize,
        n: usize,
        lhs: &[T],
        lhs_stride: [usize; 2],
        rhs: &[T],
        rhs_stride: [usize; 2],
        dst: &mut [T],
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
                T::zero(),
                T::one(),
                false,
                false,
                false,
                gemm::Parallelism::Rayon(rayon::current_num_threads()),
            )
        }
    }
}

impl<T: FloatType> super::MatMulKernel<T> for Cpu {
    fn matmul(&self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>) -> Tensor<T, Self> {
        let dim = lhs.shape.len();
        let (m, k, n) = match dim {
            2 => (lhs.shape[0], lhs.shape[1], rhs.shape[1]),
            3 => (lhs.shape[1], lhs.shape[2], rhs.shape[2]),
            4 => (lhs.shape[2], lhs.shape[3], rhs.shape[3]),
            _ => panic!("Unsupported shape dimension"),
        };

        let out_shape: Vec<usize> = match dim {
            2 => vec![m, n],
            3 => vec![lhs.shape[0], m, n],
            4 => vec![lhs.shape[0], lhs.shape[1], m, n],
            _ => panic!("Unsupported shape dimension"),
        };

        let mut out = Tensor::try_new(&out_shape, self).unwrap();

        // Check for batched matrix multiply (four dimensions).
        if dim > 2 && dim == 4 {
            let bsize = lhs.shape[0];
            let csize = lhs.shape[1];
            for b in 0..bsize {
                let base = b * lhs.strides[0];
                for c in 0..csize {
                    let start = base + c * lhs.strides[1];
                    let end = start + lhs.strides[1];

                    let a_slice = &lhs.data[start..end];
                    let b_slice = &rhs.data[start..end];
                    let c_slice = &mut out.data[start..end];
                    self.gemm_matmul(
                        m,
                        k,
                        n,
                        a_slice,
                        [lhs.strides[2], lhs.strides[3]],
                        b_slice,
                        [rhs.strides[2], rhs.strides[3]],
                        c_slice,
                        [out.strides[2], out.strides[3]],
                    );
                }
            }
        }
        // Check for batched matrix multiply (three dimensions).
        else if dim > 2 && dim == 3 {
            let bsize = lhs.shape[0];
            for b in 0..bsize {
                let start = b * lhs.strides[1];
                let end = start + lhs.strides[1];

                let a_slice = &lhs.data[start..end];
                let b_slice = &rhs.data[start..end];
                let c_slice = &mut out.data[start..end];

                self.gemm_matmul(
                    m,
                    k,
                    n,
                    a_slice,
                    [lhs.strides[1], lhs.strides[2]],
                    b_slice,
                    [rhs.strides[1], rhs.strides[2]],
                    c_slice,
                    [out.strides[1], out.strides[2]],
                );
            }
        }
        // Matrix-matrix multiplication
        else {
            self.gemm_matmul(
                m,
                k,
                n,
                &lhs.data,
                [lhs.strides[0], lhs.strides[1]],
                &rhs.data,
                [rhs.strides[0], rhs.strides[1]],
                &mut out.data,
                [out.strides[0], out.strides[1]],
            );
        }
        out
    }
}
