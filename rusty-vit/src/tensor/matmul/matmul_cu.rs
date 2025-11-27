use crate::device::cuda::Cuda;
use crate::tensor::Tensor;
use crate::type_traits::FloatType;
use cudarc::cublas;
use cudarc::cublas::sys::{
    cublasComputeType_t, cublasGemmAlgo_t, cublasOperation_t, cudaDataType, cudaDataType_t,
};
use cudarc::cublas::{CudaBlas, Gemm, StridedBatchedConfig};
use cudarc::driver::{CudaView, CudaViewMut, DevicePtr};
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

impl<T: FloatType> super::MatMulKernel<T> for Cuda
where
    CudaBlas: Gemm<T>,
    Cuda: BatchedMatMul<T>,
    Cuda: MatMul<T>,
{
    fn matmul(&mut self, lhs: &Tensor<T, Self>, rhs: &Tensor<T, Self>) -> Tensor<T, Self> {
        let dim = lhs.shape.len();

        let (m, k, n) = super::inner_shape(&lhs.shape, &rhs.shape);
        let out_shape = super::compute_shape(&lhs.shape, m, n);

        let mut out = Tensor::try_new(&out_shape, self).unwrap();

        if dim == 4 {
            for b in 0..lhs.shape[0] {
                let inner_batch_size = lhs.shape[1];
                let a_base = b * lhs.strides[0];
                let b_base = b * rhs.strides[0];
                let c_base = b * out.strides[0];
                let lhs_slice = lhs.data.slice(a_base..a_base + lhs.strides[0]);
                let rhs_slice = rhs.data.slice(b_base..b_base + rhs.strides[0]);
                let mut out_slice = out.data.slice_mut(c_base..c_base + out.strides[0]);
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
