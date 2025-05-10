use alloc::collections::VecDeque;
use alloc::vec::Vec;

use bytemuck::Pod;

use crate::helpers::{ptr_as_slice, ptr_as_slice_mut};

pub trait StaticSize {
    /// Returns the size of a type in bytes.
    fn size() -> usize;
}

impl<T: Sized> StaticSize for T {
    #[inline(always)]
    fn size() -> usize {
        core::mem::size_of::<Self>()
    }
}

pub trait VecExt {
    type Item;
    fn find_swap_remove(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item>;
    unsafe fn find_swap_remove_unchecked(
        &mut self,
        f: impl FnMut(&Self::Item) -> bool,
    ) -> Self::Item;
    fn find_remove(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item>;
    unsafe fn find_remove_unchecked(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Self::Item;
}

impl<T> VecExt for Vec<T> {
    type Item = T;

    fn find_swap_remove(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item> {
        let i = self.iter().position(f)?;
        Some(self.swap_remove(i))
    }

    unsafe fn find_swap_remove_unchecked(
        &mut self,
        f: impl FnMut(&Self::Item) -> bool,
    ) -> Self::Item {
        let i = self.iter().position(f).unwrap_unchecked();
        self.swap_remove(i)
    }

    fn find_remove(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item> {
        let i = self.iter().position(f)?;
        Some(self.remove(i))
    }

    unsafe fn find_remove_unchecked(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Self::Item {
        let i = self.iter().position(f).unwrap_unchecked();
        self.remove(i)
    }
}

impl<T> VecExt for VecDeque<T> {
    type Item = T;

    fn find_swap_remove(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item> {
        let i = self.iter().position(f)?;
        self.swap_remove_back(i)
    }

    unsafe fn find_swap_remove_unchecked(
        &mut self,
        f: impl FnMut(&Self::Item) -> bool,
    ) -> Self::Item {
        let i = self.iter().position(f).unwrap_unchecked();
        self.swap_remove_back(i).unwrap_unchecked()
    }

    fn find_remove(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Option<Self::Item> {
        let i = self.iter().position(f)?;
        self.remove(i)
    }

    unsafe fn find_remove_unchecked(&mut self, f: impl FnMut(&Self::Item) -> bool) -> Self::Item {
        let i = self.iter().position(f).unwrap_unchecked();
        self.remove(i).unwrap_unchecked()
    }
}

pub trait SliceExt {
    fn cast_slice<'a, T>(&self) -> &'a [T];
    fn cast_slice_mut<'a, T>(&mut self) -> &'a mut [T];
}

impl<U: Copy> SliceExt for [U] {
    fn cast_slice<'a, T>(&self) -> &'a [T] {
        unsafe {
            let size = self.len() * U::size();
            debug_assert!(size % T::size() == 0);
            ptr_as_slice(self.as_ptr(), size / T::size())
        }
    }

    fn cast_slice_mut<'a, T>(&mut self) -> &'a mut [T] {
        unsafe {
            let size = self.len() * U::size();
            debug_assert!(size % T::size() == 0);
            ptr_as_slice_mut(self.as_mut_ptr(), size / T::size())
        }
    }
}

pub trait OptionExtGpu {
    type Item;
    fn on_none<U>(&self, callback: impl FnOnce() -> U);
    fn on_mut<'a, U>(&'a mut self, callback: impl FnOnce(&'a mut Self::Item) -> U);
    fn on_ref<'a, U>(&'a self, callback: impl FnOnce(&'a Self::Item) -> U);
}

impl<T> OptionExtGpu for Option<T> {
    type Item = T;

    #[inline(always)]
    fn on_mut<'a, U>(&'a mut self, callback: impl FnOnce(&'a mut Self::Item) -> U) {
        if let Some(x) = self.as_mut() {
            callback(x);
        }
    }

    #[inline(always)]
    fn on_ref<'a, U>(&'a self, callback: impl FnOnce(&'a Self::Item) -> U) {
        if let Some(x) = self.as_ref() {
            callback(x);
        }
    }

    #[inline(always)]
    fn on_none<U>(&self, callback: impl FnOnce() -> U) {
        if self.is_none() {
            callback();
        }
    }
}

pub trait Ptr {
    type Item;
    unsafe fn as_mut<'a, 'b>(&'a self) -> &'b mut Self::Item;
    unsafe fn write(&self, value: Self::Item);
}

impl<T> Ptr for *const T {
    type Item = T;

    #[inline(always)]
    unsafe fn as_mut<'a, 'b>(&'a self) -> &'b mut Self::Item {
        &mut *(*self as *mut T)
    }

    #[inline(always)]
    unsafe fn write(&self, value: Self::Item) {
        *self.as_mut() = value;
    }
}

pub trait Cmov {
    fn cmov(&mut self, cond: bool, other: Self);
}

impl<T: Copy> Cmov for T {
    #[inline(always)]
    // Force a conditional move. CUDA compiler never does this.
    fn cmov(&mut self, cond: bool, other: Self) {
        *self = if cond { other } else { *self };
    }
}

pub trait UninitializedCollection {
    unsafe fn uninitialized_collection(len: usize) -> Self;
    unsafe fn zeroed(len: usize) -> Self;
}

impl<T> UninitializedCollection for Vec<T> {
    #[inline(always)]
    unsafe fn uninitialized_collection(len: usize) -> Self {
        let mut vec = Vec::with_capacity(len);
        vec.set_len(len);
        vec
    }

    #[inline(always)]
    unsafe fn zeroed(len: usize) -> Self {
        let mut vec = Self::uninitialized_collection(len);
        core::ptr::write_bytes(vec.as_mut_ptr(), 0, len);
        vec
    }
}
