use crate::cpu::device::{Cpu, VecPool};
use crate::op_traits::CastKernel;
use rvit_core::storage::DeviceStorage;
use rvit_core::tensor::*;
use rvit_core::type_traits::FloatType;

impl<T: FloatType, O: FloatType> CastKernel<T, O> for Cpu {
    fn forward(
        &self,
        x: &<Self as DeviceStorage<T>>::Vec,
        shape: &[usize],
    ) -> <Self as DeviceStorage<O>>::Vec {
        let sz = tensor_size(shape);
        let mut out = Self::try_alloc(self, sz).unwrap();
        out.data = x.iter().map(|x| O::from(*x).unwrap()).collect();
        out
    }
}
