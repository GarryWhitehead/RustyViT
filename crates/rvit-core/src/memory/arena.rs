use crate::element_traits::{DataElem, DataType};
use core::ptr::NonNull;
use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::{BitAnd, Not};
use std::slice;
use std::sync::Arc;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum ArenaError {
    #[error("An error occurred whilst allocating memory pool chunk: {reason}")]
    Allocation { reason: String },

    #[error("Memory alignment error: {reason}")]
    MemoryAlign { reason: String },

    #[error("Mismatch in tensor data types: {reason}a type")]
    TypeMismatch { reason: String },

    #[error("Invalid parameters: {reason}")]
    InvalidParameters { reason: String },
}

pub trait ArenaAllocator {
    fn new(alignment: usize, data_size: usize) -> Result<Self, ArenaError>
    where
        Self: Sized;

    fn resize(&mut self, new_size: usize) -> Result<(), ArenaError>
    where
        Self: Sized;

    fn release(&self) -> Result<(), ArenaError>;

    fn get_ptr(&self) -> NonNull<u8>;

    fn size(&self) -> usize;
}

#[derive(Clone, Copy, Debug)]
struct Entry {
    offset: usize,
    size: usize,
    alignment: usize,
    ptr: NonNull<u8>,
}

#[derive(Debug, Default, Clone)]
pub struct FreeList {
    active_pools: VecDeque<Entry>,
    free_pools: VecDeque<Entry>,
    current_alloc_size: usize,
}

impl FreeList {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(
        &mut self,
        size: usize,
        alignment: usize,
        ptr: NonNull<u8>,
        begin_ptr: NonNull<u8>,
    ) {
        let entry = self
            .free_pools
            .iter()
            .enumerate()
            .find(|(_i, entry)| size <= entry.size && alignment == entry.alignment);
        match entry {
            Some(e) => {
                let e = self.free_pools.remove(e.0).unwrap();
                self.active_pools.push_front(e.clone());
            }
            None => {
                let offset = ptr.as_ptr() as usize - begin_ptr.as_ptr() as usize;
                let e = Entry {
                    offset,
                    size,
                    alignment,
                    ptr,
                };
                self.active_pools.push_front(e.clone());
                self.current_alloc_size += size;
            }
        }
    }

    pub fn free(&mut self, size: usize, ptr: NonNull<u8>) {
        let entry = self
            .active_pools
            .iter()
            .enumerate()
            .find(|(_i, entry)| size <= entry.size && ptr == entry.ptr)
            .expect("Unable to free entry as not found in active list");
        let e = self.active_pools.remove(entry.0).unwrap();
        self.free_pools.push_back(e.clone());
    }

    pub fn query_free_pools(&mut self, size: usize, alignment: usize) -> bool {
        self.free_pools
            .iter()
            .enumerate()
            .find(|(_i, entry)| size <= entry.size && alignment == entry.alignment)
            .is_some()
    }
}

#[derive(Debug)]
pub struct ArenaPtr {
    ptr: NonNull<u8>,
    layout: std::alloc::Layout,
    dtype: DataType,
    free_list: Arc<RefCell<FreeList>>,
}

impl Clone for ArenaPtr {
    fn clone(&self) -> Self {
        let new_ptr = NonNull::new(unsafe { std::alloc::alloc(self.layout.clone()) }).unwrap();
        unsafe {
            std::ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_ptr.as_ptr(), self.layout.align())
        };
        Self {
            ptr: new_ptr,
            layout: self.layout,
            dtype: self.dtype,
            free_list: self.free_list.clone(),
        }
    }
}

impl ArenaPtr {
    pub fn new(
        layout: std::alloc::Layout,
        ptr: NonNull<u8>,
        dtype: DataType,
        free_list: Arc<RefCell<FreeList>>,
    ) -> Self {
        Self {
            ptr,
            layout,
            dtype,
            free_list,
        }
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.ptr.as_ptr()
    }

    pub fn as_slice<T: DataElem>(&self) -> Result<&[T], ArenaError> {
        if self.dtype != T::DTYPE {
            return Err(ArenaError::TypeMismatch {
                reason: "Slice return type differs to tensor.".to_string(),
            });
        }
        Ok(unsafe {
            slice::from_raw_parts(
                self.ptr.as_ptr() as *const T,
                self.layout.size() / size_of::<T>(),
            )
        })
    }

    pub fn as_mut_slice<T: DataElem>(&mut self) -> Result<&mut [T], ArenaError> {
        if self.dtype != T::DTYPE {
            return Err(ArenaError::TypeMismatch {
                reason: "Slice return type differs to tensor.".to_string(),
            });
        }
        Ok(unsafe {
            slice::from_raw_parts_mut(
                self.ptr.as_ptr() as *mut T,
                self.layout.size() / size_of::<T>(),
            )
        })
    }

    pub fn as_vec<T: DataElem>(&self) -> Result<Vec<T>, ArenaError> {
        if self.dtype != T::DTYPE {
            return Err(ArenaError::TypeMismatch {
                reason: "Vector type differs from tensor.".to_string(),
            });
        }
        Ok(self.as_slice()?.to_vec())
    }

    pub fn copy_bytes(
        &mut self,
        data: *const u8,
        offset: usize,
        size: usize,
    ) -> Result<(), ArenaError> {
        if size > self.layout.size() {
            return Err(ArenaError::InvalidParameters {
                reason: "Data size of {size} exceeds the allocated size of {self.size}".to_string(),
            });
        }
        if size + offset > self.layout.size() {
            return Err(ArenaError::InvalidParameters {
                reason: "Offset of {offset} exceeds the allocated size of {self.size}".to_string(),
            });
        }
        Ok(unsafe {
            std::ptr::copy_nonoverlapping::<u8>(
                data,
                self.ptr.cast::<u8>().as_ptr().add(offset),
                size,
            )
        })
    }

    pub fn fill_zeros(&mut self) {
        unsafe { self.ptr.write_bytes(0, self.layout.size()) }
    }

    pub fn iter<T: DataElem>(&self) -> Box<dyn Iterator<Item = T> + '_> {
        if self.dtype != T::DTYPE {
            panic!("Iter type differs from tensor.")
        }
        Box::new(self.as_slice::<T>().unwrap().iter().copied())
    }

    pub fn iter_mut<T: DataElem>(&mut self) -> Box<dyn Iterator<Item = &mut T> + '_> {
        if self.dtype != T::DTYPE {
            panic!("Iter type differs from tensor.")
        }
        Box::new(self.as_mut_slice::<T>().unwrap().iter_mut())
    }

    pub fn len(&self) -> usize {
        self.layout.size()
    }

    pub fn dtype(&self) -> DataType {
        self.dtype
    }
}

impl Drop for ArenaPtr {
    fn drop(&mut self) {
        self.free_list
            .borrow_mut()
            .free(self.layout.size(), self.ptr);
    }
}

#[derive(Debug)]
pub struct Arena<B: ArenaAllocator> {
    buffer: B,
    free_list: Arc<RefCell<FreeList>>,
    current_ptr: NonNull<u8>,
}

impl<B: ArenaAllocator> Arena<B> {
    pub fn try_new(size: usize, alignment: usize) -> Result<Self, ArenaError> {
        let buffer = B::new(size, alignment)?;
        Ok(Self {
            current_ptr: buffer.get_ptr(),
            free_list: Arc::new(RefCell::new(FreeList::new())),
            buffer,
        })
    }

    pub fn new(size: usize, alignment: usize) -> Self {
        Self::try_new(size, alignment).unwrap()
    }

    fn align_to(alignment: usize, ptr: *const u8) -> Result<*mut u8, ArenaError> {
        if alignment > 0 && (alignment & (alignment - 1)) != 0 {
            return Err(ArenaError::Allocation {
                reason: "Alignment value must be a power of two".to_string(),
            });
        }
        let ptr_val = ptr as usize;
        let align_ptr = (ptr_val + (alignment - 1)) & (alignment - 1).not();
        Ok(align_ptr as *mut u8)
    }

    pub fn alloc_bytes(
        &mut self,
        size: usize,
        alignment: usize,
        dtype: DataType,
    ) -> Result<ArenaPtr, ArenaError> {
        if size == 0 {
            return Err(ArenaError::Allocation {
                reason: "Zero sized memory allocation".to_string(),
            });
        }

        let new_size = self.free_list.borrow().current_alloc_size + size;

        if !self
            .free_list
            .borrow_mut()
            .query_free_pools(size, alignment)
            == false
            && new_size > self.buffer.size()
        {
            self.buffer.resize(new_size);
        }

        let new_ptr = NonNull::new(Self::align_to(alignment, self.current_ptr.as_ptr())?).unwrap();
        self.current_ptr = unsafe { new_ptr.add(size) };
        self.free_list
            .borrow_mut()
            .push(size, alignment, new_ptr, self.buffer.get_ptr());

        Ok(ArenaPtr::new(
            unsafe { std::alloc::Layout::from_size_align_unchecked(size, alignment) },
            new_ptr,
            dtype,
            self.free_list.clone(),
        ))
    }

    pub fn alloc<T: DataElem>(&mut self, count: usize) -> Result<ArenaPtr, ArenaError> {
        self.alloc_bytes(count * size_of::<T>(), align_of::<T>(), T::DTYPE)
    }

    pub fn free(&mut self, arena_ptr: &ArenaPtr) {
        self.free_list
            .borrow_mut()
            .free(arena_ptr.layout.size(), arena_ptr.ptr);
    }
}

impl<B: ArenaAllocator> Drop for Arena<B> {
    fn drop(&mut self) {
        self.buffer
            .release()
            .expect("Error releasing allocated memory")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_free_list_push() {
        let begin_ptr = NonNull::new(unsafe {
            std::alloc::alloc(std::alloc::Layout::from_size_align_unchecked(256, 1))
        })
        .unwrap();
        let mut list = FreeList::new();
        list.push(12, 1, begin_ptr, begin_ptr);
        assert_eq!(list.active_pools.len(), 1);

        list.free(12, begin_ptr);
        assert_eq!(list.active_pools.len(), 0);
        assert_eq!(list.free_pools.len(), 1);

        list.push(24, 1, begin_ptr, begin_ptr);
        assert_eq!(list.active_pools.len(), 1);
        assert_eq!(list.free_pools.len(), 1);

        list.push(8, 1, begin_ptr, begin_ptr);
        assert_eq!(list.active_pools.len(), 2);
        assert_eq!(list.free_pools.len(), 0);
    }
}
