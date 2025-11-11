use crate::device::cpu::Cpu;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;

trait MatMul<T: FloatType> {
    #[allow(clippy::too_many_arguments)]
    fn gemm_matmul(
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
    );
}

impl MatMul<f32> for Cpu {
    #[allow(clippy::too_many_arguments)]
    fn gemm_matmul(
        &self,
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
        &self,
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

impl<T: FloatType> super::MatMulKernel<T> for Cpu
where
    Cpu: MatMul<T>,
{
    fn matmul(&mut self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>) -> Tensor<T, Self> {
        let dim = lhs.shape.len();
        let (m, k, n) = super::inner_shape(&lhs.shape, &rhs.shape);
        let out_shape = super::compute_shape(&lhs.shape, m, n);

        let mut out = Tensor::try_new(&out_shape, self).unwrap();

        // Check for batched matrix multiply (four dimensions).
        if dim == 4 {
            let bsize = lhs.shape[0];
            let csize = lhs.shape[1];
            for b in 0..bsize {
                let a_base = b * lhs.strides[0];
                let b_base = b * rhs.strides[0];
                let c_base = b * out.strides[0];
                for c in 0..csize {
                    let a_start = a_base + c * lhs.strides[1];
                    let a_end = a_start + lhs.strides[1];
                    let a_slice = &lhs.data[a_start..a_end];

                    let b_start = b_base + c * rhs.strides[1];
                    let b_end = b_start + rhs.strides[1];
                    let b_slice = &rhs.data[b_start..b_end];

                    let c_start = c_base + c * out.strides[1];
                    let c_end = c_start + out.strides[1];
                    let c_slice = &mut out.data[c_start..c_end];

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
        else if dim == 3 {
            let bsize = lhs.shape[0];
            for b in 0..bsize {
                let a_start = b * lhs.strides[0];
                let a_end = a_start + lhs.strides[0];
                let a_slice = &lhs.data[a_start..a_end];

                let b_start = b * rhs.strides[0];
                let b_end = b_start + rhs.strides[0];
                let b_slice = &rhs.data[b_start..b_end];

                let c_start = b * out.strides[0];
                let c_end = c_start + out.strides[0];
                let c_slice = &mut out.data[c_start..c_end];

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
