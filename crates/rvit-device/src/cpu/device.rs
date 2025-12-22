use crate::cpu::memory::*;
use half::f16;
use rvit_core::element_traits::{DataElem, DataType};
use rvit_core::memory::arena::*;
use rvit_core::memory::storage::DeviceStorage;
use std::cell::RefCell;
use std::fmt::Debug;
use std::rc::Rc;

type CpuArena = Arena<AlignedBuffer>;

#[derive(Debug)]
pub struct Cpu {
    arena: Rc<RefCell<CpuArena>>,
}

impl Clone for Cpu {
    fn clone(&self) -> Self {
        Self {
            arena: self.arena.clone(),
        }
    }
}

const INIT_ARENA_SIZE: usize = 1024 * 1024 * 1024;
impl Default for Cpu {
    fn default() -> Self {
        Self {
            arena: Rc::new(RefCell::new(Arena::new(INIT_ARENA_SIZE, 1))),
        }
    }
}

impl DeviceStorage for Cpu {
    type Alloc = ArenaPtr;

    fn try_alloc(&mut self, count: usize, dtype: DataType) -> Result<Self::Alloc, ArenaError> {
        self.arena.borrow_mut().alloc_bytes(
            computes_byte_size(count, dtype),
            compute_alignment(dtype),
            dtype,
        )
    }

    fn try_alloc_with_bytes(
        &mut self,
        data: *const u8,
        size: usize,
        dtype: DataType,
    ) -> Result<Self::Alloc, ArenaError> {
        let mut ptr = self
            .arena
            .borrow_mut()
            .alloc_bytes(size, compute_alignment(dtype), dtype)?;
        ptr.copy_bytes(data, 0, size)?;
        Ok(ptr)
    }

    fn try_alloc_with_data<T: DataElem>(&mut self, data: &[T]) -> Result<Self::Alloc, ArenaError> {
        let mut ptr = self.try_alloc(data.len(), T::DTYPE)?;
        ptr.copy_bytes(data.as_ptr() as *const u8, 0, data.len() * size_of::<T>())?;
        Ok(ptr)
    }

    fn try_alloc_zeros(
        &mut self,
        count: usize,
        dtype: DataType,
    ) -> Result<Self::Alloc, ArenaError> {
        let mut ptr = self.try_alloc(computes_byte_size(count, dtype), dtype)?;
        ptr.fill_zeros();
        Ok(ptr)
    }

    fn try_into_vec<T: DataElem>(&self, src: &ArenaPtr) -> Result<Vec<T>, ArenaError> {
        Ok(src.as_vec()?)
    }

    fn len(ptr: &Self::Alloc) -> usize {
        ptr.len()
    }

    fn try_sync(&self) -> Result<(), ArenaError> {
        Ok(())
    }
}

fn computes_byte_size(count: usize, dtype: DataType) -> usize {
    match dtype {
        DataType::F32 => count * size_of::<f32>(),
        DataType::F16 => count * size_of::<f16>(),
        DataType::U8 => count * size_of::<u8>(),
        DataType::U16 => count * size_of::<u16>(),
        DataType::U32 => count * size_of::<u32>(),
        DataType::I8 => count * size_of::<i8>(),
        DataType::I16 => count * size_of::<i16>(),
        DataType::I32 => count * size_of::<i32>(),
    }
}

fn compute_alignment(dtype: DataType) -> usize {
    match dtype {
        DataType::F32 => align_of::<f32>(),
        DataType::F16 => align_of::<f16>(),
        DataType::U8 => align_of::<u8>(),
        DataType::U16 => align_of::<u16>(),
        DataType::U32 => align_of::<u32>(),
        DataType::I8 => align_of::<i8>(),
        DataType::I16 => align_of::<i16>(),
        DataType::I32 => align_of::<i32>(),
    }
}
