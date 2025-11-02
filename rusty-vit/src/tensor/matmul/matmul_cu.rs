use crate::device::cuda::Cuda;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
use cudarc::cublas;
use cudarc::cublas::sys::cublasOperation_t;
use cudarc::cublas::{CudaBlas, Gemm, StridedBatchedConfig};
use cudarc::driver::{CudaView, CudaViewMut};
use std::ffi::{c_int, c_longlong};

impl Cuda {
    fn gemm<T: FloatType>(
        &self,
        m: usize,
        n: usize,
        k: usize,
        lhs: &CudaView<T>,
        rhs: &CudaView<T>,
        dst: &mut CudaViewMut<T>,
    ) where
        CudaBlas: Gemm<T>,
    {
        // Note: Cublas is col major so we do (B^T *A^T) instead though we
        // don't have to transpose B and A as they are row major.
        // Thus, we switch m and n values as we now have B * A.
        let cfg = cublas::GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as c_int,
            n: m as c_int,
            k: k as c_int,
            alpha: T::one(),
            beta: T::zero(),
            // These leading values are for square matrices. Will need
            // to check strides if vector-matrix and vector-vector support
            // is added.
            lda: n as i32, // n == m as A and B switched
            ldb: k as i32, // k (no trans) or n (trans)
            ldc: n as i32, // n == m as A and B switched
        };
        unsafe { self.blas.gemm(cfg, rhs, lhs, dst).unwrap() };
    }

    fn batched_gemm<T: FloatType>(
        &self,
        m: usize,
        n: usize,
        k: usize,
        batch_size: usize,
        lhs_batch_stride: usize,
        rhs_batch_stride: usize,
        dst_batch_stride: usize,
        lhs: &CudaView<T>,
        rhs: &CudaView<T>,
        dst: &mut CudaViewMut<T>,
    ) where
        CudaBlas: Gemm<T>,
    {
        let gemm_cfg = cublas::GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: m as c_int,
            n: n as c_int,
            k: k as c_int,
            alpha: T::one(),
            beta: T::zero(),
            // These leading values are for square matrices. Will need
            // to check strides if vector-matrix and vector-vector support
            // is added.
            lda: k as i32,
            ldb: n as i32,
            ldc: n as i32,
        };

        let cfg = StridedBatchedConfig {
            gemm: gemm_cfg,
            batch_size: batch_size as c_int,
            stride_a: lhs_batch_stride as c_longlong,
            stride_b: rhs_batch_stride as c_longlong,
            stride_c: dst_batch_stride as c_longlong,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, lhs, rhs, dst).unwrap() };
    }
}

impl<T: FloatType> super::MatMulKernel<T> for Cuda
where
    CudaBlas: Gemm<T>,
{
    fn matmul(&mut self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>) -> Tensor<T, Self> {
        let dim = lhs.shape.len();

        let (m, k, n) = super::inner_shape(&lhs.shape, &rhs.shape);
        let out_shape = super::compute_shape(&lhs.shape, m, n);

        let mut out = Tensor::try_new(&out_shape, self).unwrap();

        if dim == 4 {
            for b in 0..lhs.shape[0] {
                let inner_batch_size = lhs.shape[1];
                let base = b * lhs.strides[0];
                let lhs_slice = lhs.data.slice(base..base + lhs.strides[0]);
                let rhs_slice = rhs.data.slice(base..base + rhs.strides[0]);
                let mut out_slice = out.data.slice_mut(base..base + out.strides[0]);
                self.batched_gemm(
                    m,
                    n,
                    k,
                    inner_batch_size,
                    lhs.strides[1],
                    rhs.strides[1],
                    out.strides[1],
                    &lhs_slice,
                    &rhs_slice,
                    &mut out_slice,
                )
            }
        } else if dim == 3 {
            let batch_size = lhs.shape[0];
            self.batched_gemm(
                m,
                n,
                k,
                batch_size,
                lhs.strides[0],
                rhs.strides[0],
                out.strides[0],
                &lhs.data.as_view(),
                &rhs.data.as_view(),
                &mut out.data.as_view_mut(),
            )
        } else {
            self.gemm(
                m,
                n,
                k,
                &lhs.data.as_view(),
                &rhs.data.as_view(),
                &mut out.data.as_view_mut(),
            );
        }
        out
    }
}
