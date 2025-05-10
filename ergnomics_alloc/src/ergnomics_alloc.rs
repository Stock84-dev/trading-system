#![no_std]
extern crate alloc;
use alloc::boxed::Box;
use alloc::vec::Vec;
use core::ops::{Deref, DerefMut, Index, IndexMut};

pub mod ptr;

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnsafeBuffer<T> {
    inner: Box<[T]>,
}

impl<T> UnsafeBuffer<T> {
    #[inline(always)]
    pub fn new(inner: Box<[T]>) -> Self {
        Self { inner }
    }
}

impl<T, I: TryInto<usize>> Index<I> for UnsafeBuffer<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        #[cfg(debug_assertions)]
        {
            let Ok(index) = index.try_into() else {
                panic!("could not convert index to usize");
            };
            if index >= self.inner.len() {
                panic!("index out of bounds");
            }
            unsafe { self.inner.get_unchecked(index) }
        }
        #[cfg(not(debug_assertions))]
        {
            unsafe { self.inner.get_unchecked(index.try_into().unwrap_unchecked()) }
        }
    }
}

impl<T, I: TryInto<usize>> IndexMut<I> for UnsafeBuffer<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        #[cfg(debug_assertions)]
        {
            let Ok(index) = index.try_into() else {
                panic!("could not convert index to usize");
            };
            if index >= self.inner.len() {
                panic!("index out of bounds");
            }
            unsafe { self.inner.get_unchecked_mut(index) }
        }
        #[cfg(not(debug_assertions))]
        {
            unsafe { self.inner.get_unchecked_mut(index.try_into().unwrap_unchecked()) }
        }
    }
}

impl<T> Deref for UnsafeBuffer<T> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for UnsafeBuffer<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnsafeVec<T> {
    inner: Vec<T>,
}

impl<T> UnsafeVec<T> {
    #[inline(always)]
    pub fn uninitialized(capacity: usize) -> Self {
        let mut vec = Vec::with_capacity(capacity);
        unsafe {
            vec.set_len(capacity);
        }
        Self { inner: vec }
    }

    #[inline(always)]
    pub fn set(&self, index: usize, value: T) {
        unsafe {
            (self.inner.as_ptr().add(index) as *mut T).write(value);
        }
    }

    #[inline(always)]
    pub fn get_mut(&self, index: usize) -> &mut T {
        #[allow(invalid_reference_casting)]
        unsafe {
            &mut *(self.index(index) as *const T as *mut T)
        }
    }

    #[inline(always)]
    pub fn new(inner: Vec<T>) -> Self {
        Self { inner }
    }
}

impl<T, I: TryInto<usize>> Index<I> for UnsafeVec<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        #[cfg(debug_assertions)]
        {
            let Ok(index) = index.try_into() else {
                panic!("could not convert index to usize");
            };
            if index >= self.inner.len() {
                panic!("index out of bounds");
            }
            unsafe { self.inner.get_unchecked(index) }
        }
        #[cfg(not(debug_assertions))]
        {
            unsafe { self.inner.get_unchecked(index.try_into().unwrap_unchecked()) }
        }
    }
}

impl<T, I: TryInto<usize>> IndexMut<I> for UnsafeVec<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        #[cfg(debug_assertions)]
        {
            let Ok(index) = index.try_into() else {
                panic!("could not convert index to usize");
            };
            if index >= self.inner.len() {
                panic!("index out of bounds");
            }
            unsafe { self.inner.get_unchecked_mut(index) }
        }
        #[cfg(not(debug_assertions))]
        {
            unsafe { self.inner.get_unchecked_mut(index.try_into().unwrap_unchecked()) }
        }
    }
}

impl<T> Deref for UnsafeVec<T> {
    type Target = Vec<T>;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for UnsafeVec<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
