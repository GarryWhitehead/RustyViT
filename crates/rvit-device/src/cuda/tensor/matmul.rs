use crate::cuda::device::Cuda;
use crate::op_traits::MatMulKernel;
use cudarc::cublas;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType, cudaDataType_t,
};
use cudarc::cublas::{CudaBlas, Gemm, StridedBatchedConfig};
use cudarc::driver::{CudaView, CudaViewMut, DevicePtr};
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::*;
use rvit_core::type_traits::FloatType;
use std::ffi::{c_int, c_longlong};

pub(crate) trait MatMul<T: FloatType> {
    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        lhs: &CudaView<T>,
        rhs: &CudaView<T>,
        dst: &mut CudaViewMut<T>,
    );
}

pub(crate) trait BatchedMatMul<T: FloatType> {
    fn batched_gemm(
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
    );
}

impl MatMul<f32> for Cuda {
    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        lhs: &CudaView<f32>,
        rhs: &CudaView<f32>,
        dst: &mut CudaViewMut<f32>,
    ) where
        CudaBlas: Gemm<f32>,
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
            alpha: 1.0f32,
            beta: 0.0f32,
            // These leading values are for square matrices. Will need
            // to check strides if vector-matrix and vector-vector support
            // is added.
            lda: n as i32, // n == m as A and B switched
            ldb: k as i32, // k (no trans) or n (trans)
            ldc: n as i32, // n == m as A and B switched
        };
        unsafe { self.blas.gemm(cfg, rhs, lhs, dst).unwrap() };
    }
}

impl MatMul<half::f16> for Cuda {
    fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        lhs: &CudaView<half::f16>,
        rhs: &CudaView<half::f16>,
        dst: &mut CudaViewMut<half::f16>,
    ) where
        CudaBlas: Gemm<half::f16>,
    {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            cublas::result::gemm_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                (&alpha) as *const f32 as *const _,
                rhs.device_ptr(&self.stream0).0 as *const _,
                cudaDataType::CUDA_R_16F,
                n as c_int,
                lhs.device_ptr(&self.stream0).0 as *const _,
                cudaDataType::CUDA_R_16F,
                k as c_int,
                (&beta) as *const f32 as *const _,
                dst.device_ptr(&self.stream0).0 as *mut _,
                cudaDataType::CUDA_R_16F,
                n as c_int,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .unwrap()
        }
    }
}

impl BatchedMatMul<f32> for Cuda {
    fn batched_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        batch_size: usize,
        lhs_batch_stride: usize,
        rhs_batch_stride: usize,
        dst_batch_stride: usize,
        lhs: &CudaView<f32>,
        rhs: &CudaView<f32>,
        dst: &mut CudaViewMut<f32>,
    ) where
        CudaBlas: Gemm<f32>,
    {
        let gemm_cfg = cublas::GemmConfig {
            transa: cublasOperation_t::CUBLAS_OP_N,
            transb: cublasOperation_t::CUBLAS_OP_N,
            m: n as c_int,
            n: m as c_int,
            k: k as c_int,
            alpha: 1.0f32,
            beta: 0.0f32,
            // These leading values are for square matrices. Will need
            // to check strides if vector-matrix and vector-vector support
            // is added.
            lda: n as i32,
            ldb: k as i32,
            ldc: n as i32,
        };

        let cfg = StridedBatchedConfig {
            gemm: gemm_cfg,
            batch_size: batch_size as c_int,
            stride_a: rhs_batch_stride as c_longlong,
            stride_b: lhs_batch_stride as c_longlong,
            stride_c: dst_batch_stride as c_longlong,
        };
        unsafe { self.blas.gemm_strided_batched(cfg, rhs, lhs, dst).unwrap() };
    }
}

impl BatchedMatMul<half::f16> for Cuda {
    fn batched_gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        batch_size: usize,
        lhs_batch_stride: usize,
        rhs_batch_stride: usize,
        dst_batch_stride: usize,
        lhs: &CudaView<half::f16>,
        rhs: &CudaView<half::f16>,
        dst: &mut CudaViewMut<half::f16>,
    ) where
        CudaBlas: Gemm<half::f16>,
    {
        let alpha = 1.0f32;
        let beta = 0.0f32;
        unsafe {
            cublas::result::gemm_strided_batched_ex(
                *self.blas.handle(),
                cublasOperation_t::CUBLAS_OP_N,
                cublasOperation_t::CUBLAS_OP_N,
                n as c_int,
                m as c_int,
                k as c_int,
                (&alpha) as *const f32 as *const _,
                rhs.device_ptr(&self.stream0).0 as *const _,
                cudaDataType_t::CUDA_R_16F,
                n as c_int,
                rhs_batch_stride as c_longlong,
                lhs.device_ptr(&self.stream0).0 as *const _,
                cudaDataType_t::CUDA_R_16F,
                k as c_int,
                lhs_batch_stride as c_longlong,
                (&beta) as *const f32 as *const _,
                dst.device_ptr(&self.stream0).0 as *mut _,
                cudaDataType_t::CUDA_R_16F,
                n as c_int,
                dst_batch_stride as c_longlong,
                batch_size as c_int,
                cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )
            .unwrap()
        }
    }
}

impl<T: FloatType> MatMulKernel<T> for Cuda
where
    CudaBlas: Gemm<T>,
    Self: BatchedMatMul<T>,
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
    ) -> Self::Vec {
        let dim = lhs_shape.len();

        let (m, k, n) = crate::tensor::inner_shape(&lhs_shape, &rhs_shape);
        let out_shape = crate::tensor::compute_shape(&lhs_shape, m, n);
        let out_sz = tensor_size(&out_shape);

        let mut out = Self::try_alloc(self, out_sz).unwrap();
        let out_strides = compute_strides(&out_shape);

        if dim == 4 {
            for b in 0..lhs_shape[0] {
                let inner_batch_size = lhs_shape[1];
                let a_base = b * lhs_strides[0];
                let b_base = b * rhs_strides[0];
                let c_base = b * out_strides[0];
                let lhs_slice = lhs.slice(a_base..a_base + lhs_strides[0]);
                let rhs_slice = rhs.slice(b_base..b_base + rhs_strides[0]);
                let mut out_slice = out.slice_mut(c_base..c_base + out_strides[0]);
                self.batched_gemm(
                    m,
                    n,
                    k,
                    inner_batch_size,
                    lhs_strides[1],
                    rhs_strides[1],
                    out_strides[1],
                    &lhs_slice,
                    &rhs_slice,
                    &mut out_slice,
                )
            }
        } else if dim == 3 {
            let batch_size = lhs_shape[0];
            self.batched_gemm(
                m,
                n,
                k,
                batch_size,
                lhs_strides[0],
                rhs_strides[0],
                out_strides[0],
                &lhs.as_view(),
                &rhs.as_view(),
                &mut out.as_view_mut(),
            )
        } else {
            self.gemm(
                m,
                n,
                k,
                &lhs.as_view(),
                &rhs.as_view(),
                &mut out.as_view_mut(),
            );
        }
        out
    }
}
