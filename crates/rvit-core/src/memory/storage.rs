use crate::element_traits::{DataElem, DataType, Elem, FloatElem, IntElem};
use crate::memory::arena::*;
use std::fmt::Debug;

pub trait DeviceStorage: Clone {
    type Alloc: Clone + Debug + 'static;

    fn try_alloc(&mut self, count: usize, dtype: DataType) -> Result<Self::Alloc, ArenaError>;

    fn try_alloc_with_bytes(
        &mut self,
        data: *const u8,
        size: usize,
        dtype: DataType,
    ) -> Result<Self::Alloc, ArenaError>;

    fn try_alloc_with_data<T: DataElem>(&mut self, data: &[T]) -> Result<Self::Alloc, ArenaError>;

    fn try_alloc_zeros(&mut self, count: usize, dtype: DataType)
    -> Result<Self::Alloc, ArenaError>;

    fn try_into_vec<T: DataElem>(&self, src: &Self::Alloc) -> Result<Vec<T>, ArenaError>;

    fn len(ptr: &Self::Alloc) -> usize;

    fn try_sync(&self) -> Result<(), ArenaError>;
}
