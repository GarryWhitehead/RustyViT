use crate::tensor::op_traits::CastKernel;
use crate::{DAlloc, Runtime};
use half::f16;
use rvit_core::element_traits::{DataElem, DataType};
use rvit_core::memory::storage::DeviceStorage;

impl<O: DataElem> CastKernel<O> for Runtime {
    fn cast_fwd(&mut self, x: &DAlloc<Self>) -> DAlloc<Self> {
        if x.dtype() == O::DTYPE {
            return x.clone();
        }

        let c: Vec<O> = match O::DTYPE {
            DataType::F32 => x.iter().map(|x: f32| O::from(x).unwrap()).collect(),
            DataType::F16 => x.iter().map(|x: f16| O::from(x).unwrap()).collect(),
            DataType::U8 => x.iter().map(|x: u8| O::from(x).unwrap()).collect(),
            DataType::U16 => x.iter().map(|x: u16| O::from(x).unwrap()).collect(),
            DataType::U32 => x.iter().map(|x: u32| O::from(x).unwrap()).collect(),
            DataType::I8 => x.iter().map(|x: i8| O::from(x).unwrap()).collect(),
            DataType::I16 => x.iter().map(|x: i16| O::from(x).unwrap()).collect(),
            DataType::I32 => x.iter().map(|x: i32| O::from(x).unwrap()).collect(),
        };
        self.storage.try_alloc_with_data::<O>(&c).unwrap()
    }
}
