use crate::device::DeviceStorage;
use crate::image::{Image, PixelType};
use crate::tensor::Tensor;
use log::{debug, info};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fmt::Debug;
use std::sync::Arc;

use crate::device::cu_utils::*;
use crate::type_traits::{BType, FloatType};
use cudarc::cublas::CudaBlas;
use cudarc::driver::{
    CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct Cuda {
    pub(crate) ctx: Arc<CudaContext>,
    pub(crate) blas: Arc<CudaBlas>,
    pub(crate) stream0: Arc<CudaStream>,
    pub(crate) stream1: Arc<CudaStream>,
    pub(crate) kernel_funcs: HashMap<String, CudaFunction>,
}

impl Cuda {
    pub fn try_new(ordinal: usize) -> Result<Cuda, Box<dyn Error>> {
        let ctx = CudaContext::new(ordinal)?;
        debug!("Created new Cuda context: {:?}", ctx);
        let blas = CudaBlas::new(ctx.default_stream())?;
        let stream0 = ctx.default_stream();
        let stream1 = ctx.default_stream().fork()?;
        Ok(Cuda {
            ctx,
            blas: Arc::new(blas),
            stream0,
            stream1,
            kernel_funcs: HashMap::new(),
        })
    }

    pub fn is_kernel_registered(&self, name: &str) -> bool {
        self.kernel_funcs.contains_key(name)
    }

    pub fn get_kernel_func(&self, name: &str) -> Result<CudaFunction, Box<dyn Error>> {
        if let Some(func) = self.kernel_funcs.get(name) {
            return Ok(func.clone());
        }
        Err("Kernel function not found".into())
    }

    pub fn try_register_kernel(
        &mut self,
        cu_str: &str,
        kernel_name: &str,
    ) -> Result<CudaFunction, Box<dyn Error>> {
        let f = self.compile_kernel(cu_str, kernel_name)?;
        if self
            .kernel_funcs
            .insert(kernel_name.to_string(), f.clone())
            .is_some()
        {
            return Err(format!("Kernel already registered: {}", kernel_name).into());
        }
        Ok(f)
    }

    pub fn register_kernel(&mut self, ptx_str: &str, kernel_name: &str) -> CudaFunction {
        if !self.is_kernel_registered(kernel_name) {
            let func = self.compile_kernel(ptx_str, kernel_name);
            match func {
                Ok(f) => {
                    let _ = self.kernel_funcs.insert(kernel_name.to_string(), f.clone());
                    return f;
                }
                Err(e) => {
                    panic!("Failed to compile kernel: {}", e);
                }
            }
        }
        self.get_kernel_func(kernel_name).unwrap()
    }

    pub fn compile_kernel(
        &self,
        ptx_str: &str,
        kernel_name: &str,
    ) -> Result<CudaFunction, Box<dyn Error>> {
        info!("Compiling kernel: {}", kernel_name);
        let module = self.ctx.load_module(Ptx::from_src(ptx_str))?;
        let kernel_func = module.load_function(kernel_name)?;
        Ok(kernel_func)
    }
}

impl<T: BType> DeviceStorage<T> for Cuda {
    type Vec = CudaSlice<T>;
    fn try_alloc(&self, sz: usize) -> Result<Self::Vec, Box<dyn Error>> {
        assert!(sz > 0);
        let mut cslice = unsafe { self.stream0.alloc(sz)? };
        self.stream0.memset_zeros(&mut cslice)?;
        Ok(cslice)
    }

    fn try_alloc_with_slice(&self, slice: &[T]) -> Result<Self::Vec, Box<dyn Error>> {
        let mut v = self.try_alloc(slice.len())?;
        self.stream0.memcpy_htod(slice, &mut v)?;
        Ok(v)
    }

    fn try_from_device_vec(&self, src: &Self::Vec) -> Result<Vec<T>, Box<dyn Error>> {
        let mut v = vec![T::zero(); src.len()];
        self.stream0.memcpy_dtoh(src, &mut v)?;
        Ok(v)
    }

    fn len(vec: &Self::Vec) -> usize {
        vec.len()
    }

    fn try_sync(&self) -> Result<(), Box<dyn Error>> {
        self.stream0.synchronize()?;
        Ok(())
    }
}

trait KernelOp<P: PixelType, F: FloatType> {
    const KERNEL_NAME: &'static str;
}
impl KernelOp<u8, f32> for Cuda {
    const KERNEL_NAME: &'static str = "to_tensor_u8_f32_kernel";
}
impl KernelOp<u8, f64> for Cuda {
    const KERNEL_NAME: &'static str = "to_tensor_u8_f64_kernel";
}
impl KernelOp<u16, f32> for Cuda {
    const KERNEL_NAME: &'static str = "to_tensor_u16_f32_kernel";
}
impl KernelOp<u16, f64> for Cuda {
    const KERNEL_NAME: &'static str = "to_tensor_u16_f64_kernel";
}
impl KernelOp<f32, f32> for Cuda {
    const KERNEL_NAME: &'static str = "to_tensor_f32_f32_kernel";
}
impl KernelOp<f32, f64> for Cuda {
    const KERNEL_NAME: &'static str = "to_tensor_f32_f64_kernel";
}

impl<P: PixelType, F: FloatType> super::ToTensor<P, Self, F, Self> for Cuda
where
    Self: KernelOp<P, F>,
{
    fn to_tensor(
        &mut self,
        image: &Image<P, Self>,
        norm: (&[F], &[F]),
    ) -> Result<Tensor<F, Self>, Box<dyn Error>> {
        if norm.0.len() != norm.1.len() {
            panic!("Mean and std arrays are different lengths");
        }
        if norm.0.len() != image.channels || norm.1.len() != image.channels {
            panic!("There must be a mean/std dev value for each channel");
        }

        let k_path = format!(
            "{}/{}.cu",
            env::current_dir().unwrap().to_str().unwrap(),
            "flip_image"
        );
        let k_func = self.register_kernel(k_path.as_str(), Self::KERNEL_NAME);

        let block_dim = (16, 16, 1);
        let grid_dim = (
            div_up(image.width as u32, block_dim.0),
            div_up(image.height as u32, block_dim.1),
            1,
        );

        let tensor = Tensor::try_new(
            &[image.batch_size, image.channels, image.width, image.height],
            self,
        )?;
        for b in 0..image.batch_size {
            let slice_base = b * image.channels * image.width * image.height;
            for c in 0..image.channels {
                let start = slice_base + c * image.width * image.height;
                let end = start + image.width * image.height;
                let is = image.data.slice(start..end);
                let ms = tensor.data.slice(start..end);

                let mut builder = self.stream0.launch_builder(&k_func);
                builder
                    .arg(&is)
                    .arg(&image.width)
                    .arg(&image.height)
                    .arg(&norm.0[c])
                    .arg(&norm.1[c])
                    .arg(&ms);
                let cfg = LaunchConfig {
                    block_dim,
                    grid_dim,
                    shared_mem_bytes: 0,
                };
                unsafe { builder.launch(cfg) }.unwrap();
            }
        }
        Ok(tensor)
    }
}
