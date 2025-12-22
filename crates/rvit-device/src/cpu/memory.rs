use rvit_core::memory::arena::{ArenaAllocator, ArenaError};
use std::ptr::NonNull;

#[derive(Clone, Debug)]
pub struct AlignedBuffer {
    alignment: usize,
    data_size: usize,
    ptr: NonNull<u8>,
}

impl ArenaAllocator for AlignedBuffer {
    fn new(data_size: usize, alignment: usize) -> Result<Self, ArenaError> {
        if data_size == 0 {
            panic!("size must be a non-zero value.");
        }
        let layout = std::alloc::Layout::from_size_align(data_size, alignment).map_err(|e| {
            ArenaError::MemoryAlign {
                reason: e.to_string(),
            }
        })?;
        let ptr = NonNull::new(unsafe { std::alloc::alloc(layout) });
        match ptr {
            Some(ptr) => Ok(Self {
                alignment,
                data_size,
                ptr,
            }),
            None => Err(ArenaError::Allocation {
                reason: "Error whilst allocating memory from the heap.".to_string(),
            }),
        }
    }

    fn resize(&mut self, new_size: usize) -> Result<(), ArenaError> {
        if new_size <= self.data_size {
            return Ok(());
        }

        let layout =
            std::alloc::Layout::from_size_align(self.data_size, self.alignment).map_err(|e| {
                ArenaError::MemoryAlign {
                    reason: e.to_string(),
                }
            })?;
        let ptr = NonNull::new(unsafe { std::alloc::realloc(self.ptr.as_ptr(), layout, new_size) });
        match ptr {
            Some(ptr) => {
                self.ptr = ptr;
                self.data_size = new_size;
                Ok(())
            }
            None => Err(ArenaError::Allocation {
                reason: "Error whilst re-allocating memory.".to_string(),
            }),
        }
    }

    fn release(&self) -> Result<(), ArenaError> {
        let layout =
            std::alloc::Layout::from_size_align(self.data_size, self.alignment).map_err(|e| {
                ArenaError::MemoryAlign {
                    reason: e.to_string(),
                }
            })?;
        unsafe {
            std::alloc::dealloc(self.ptr.cast::<u8>().as_ptr(), layout);
        }
        Ok(())
    }

    fn get_ptr(&self) -> NonNull<u8> {
        self.ptr
    }

    fn size(&self) -> usize {
        self.data_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rvit_core::element_traits::DataType;
    use rvit_core::memory::arena::Arena;

    #[test]
    fn test_iter() {
        let mut arena = Arena::<AlignedBuffer>::try_new(256, 1).unwrap();
        let mut ptr1 = arena.alloc_bytes(12, 1, DataType::U8).unwrap();
        let mut ptr2 = arena
            .alloc_bytes(12, align_of::<u32>(), DataType::U32)
            .unwrap();

        for (i, p1) in ptr1.iter_mut::<u8>().enumerate() {
            *p1 = i as u8;
        }
        for (i, p2) in ptr2.iter_mut::<u32>().enumerate() {
            *p2 = i as u32;
        }

        for (i, p1) in ptr1.iter::<u8>().enumerate() {
            assert_eq!(p1, i as u8);
        }
        for (i, p1) in ptr2.iter::<u32>().enumerate() {
            assert_eq!(p1, i as u32);
        }
    }

    #[test]
    fn test_copy_bytes() {
        let data: [f32; 6] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let mut arena = Arena::<AlignedBuffer>::try_new(256, 1).unwrap();
        let mut ptr = arena.alloc_bytes(data.len() * 4, 4, DataType::F32).unwrap();
        ptr.copy_bytes(data.as_ptr() as *const u8, 0, data.len() * 4)
            .unwrap();
        for (i, p) in ptr.iter::<f32>().enumerate() {
            assert_eq!(i as f32, p);
        }
    }
}
