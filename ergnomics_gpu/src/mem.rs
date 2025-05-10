use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut, Index, IndexMut};

use bytemuck::Pod;

pub fn malloc<T: Pod>(len: usize) -> Vec<T> {
    let mut v = Vec::with_capacity(len);
    unsafe {
        v.set_len(len);
    }
    // v.into_boxed_slice()
    v
}

#[repr(C)]
pub struct UnsafePtr<T> {
    ptr: *const T,
    len: usize,
}

impl<T> UnsafePtr<T> {
    #[inline(always)]
    pub fn null() -> Self {
        Self {
            ptr: core::ptr::null(),
            len: 0,
        }
    }

    #[inline(always)]
    pub fn new(ptr: *const T, len: usize) -> Self {
        Self { ptr, len }
    }

    #[inline(always)]
    pub fn add(&self, offset: impl Into<usize>) -> Self {
        let offset = offset.into();
        debug_assert!(offset < self.len);
        Self {
            ptr: unsafe { self.ptr.add(offset.into()) },
            len: self.len - offset,
        }
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.ptr as *mut T
    }

    #[inline(always)]
    pub fn slice(&self) -> &[T] {
        unsafe {
            core::slice::from_raw_parts(self.ptr, self.len)
        }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn cast<U>(&self) -> UnsafePtr<U> {
        if core::mem::size_of::<T>() > core::mem::size_of::<U>() {
            debug_assert!(core::mem::size_of::<T>() % core::mem::size_of::<U>() == 0);
            UnsafePtr {
                ptr: self.ptr as *const U,
                len: self.len * core::mem::size_of::<T>() / core::mem::size_of::<U>(),
            }
        } else {
            debug_assert!(core::mem::size_of::<U>() % core::mem::size_of::<T>() == 0);
            UnsafePtr {
                ptr: self.ptr as *const U,
                len: self.len * core::mem::size_of::<U>() / core::mem::size_of::<T>(),
            }
        }
    }
}

impl<T, U: Into<u32>> Index<U> for UnsafePtr<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: U) -> &Self::Output {
        let index = index.into();
        debug_assert!(index < self.len);
        unsafe { &*self.ptr.add(index as usize) }
    }
}

impl<T, U: Into<u32>> IndexMut<U> for UnsafePtr<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: U) -> &mut Self::Output {
        let index = index.into();
        debug_assert!(index < self.len);
        unsafe { &mut *(self.ptr.add(index as usize) as *mut T) }
    }
}

impl<T> Deref for UnsafePtr<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        debug_assert!(self.len > 0);
        unsafe { &*self.ptr }
    }
}

impl<T> DerefMut for UnsafePtr<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        debug_assert!(self.len > 0);
        unsafe { &mut *(self.ptr as *mut T) }
    }
}

impl<T> Clone for UnsafePtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr, len: self.len }
    }
}

impl<T> Copy for UnsafePtr<T> {}

unsafe impl<T> Send for UnsafePtr<T> {}
unsafe impl<T> Sync for UnsafePtr<T> {}
