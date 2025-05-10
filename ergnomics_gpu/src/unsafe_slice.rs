use core::ops::{Index, IndexMut, RangeTo};
// use crate::debug_assert;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct UnsafeSlice<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a, T> UnsafeSlice<'a, T> {
    #[inline(always)]
    pub unsafe fn from_slice(slice: &[T]) -> Self {
        Self::from_raw_parts(slice.as_ptr(), slice.len())
    }

    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn iter(&self) -> core::slice::Iter<'a, T> {
        core::slice::from_raw_parts(self.ptr, self.len).iter()
    }

    #[inline(always)]
    pub unsafe fn first(&self) -> &T {
        debug_assert!(self.len > 0);
        &*self.ptr
    }

    #[inline(always)]
    pub unsafe fn last(&self) -> &T {
        debug_assert!(self.len > 0);
        &*self.ptr.add(self.len - 1)
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl<'a, T> AsRef<[T]> for UnsafeSlice<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

macro_rules! index {
    ($($t:ty),*) => {
        $(
            impl<'a, T> Index<$t> for UnsafeSlice<'a, T> {
                type Output = T;

                #[inline(always)]
                fn index(&self, index: $t) -> &Self::Output {
                    let index = index as usize;
                    debug_assert!(index < self.len);
                    unsafe { &*self.ptr.add(index) }
                }
            }

            impl<'a, T> Index<$t> for UnsafeSliceMut<'a, T> {
                type Output = T;

                #[inline(always)]
                fn index(&self, index: $t) -> &Self::Output {
                    let index = index as usize;
                    debug_assert!(index < self.len);
                    unsafe { &*self.ptr.add(index) }
                }
            }

            impl<'a, T> IndexMut<$t> for UnsafeSliceMut<'a, T> {
                #[inline(always)]
                fn index_mut(&mut self, index: $t) -> &mut Self::Output {
                    debug_assert!(index < self.len);
                    unsafe { &mut *(self.ptr.add(index as usize) as *mut T) }
                }
            }
        )*
    };
}
index!(i8, u8, i16, u16, i32, u32, i64, u64, i128, u128, usize, isize);

impl<'a, T> Index<RangeTo<usize>> for UnsafeSlice<'a, T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        debug_assert!(index.end < self.len);
        unsafe { core::slice::from_raw_parts(self.ptr, index.end) }
    }
}

impl<'a, T: 'a> IntoIterator for UnsafeSlice<'a, T> {
    type IntoIter = core::slice::Iter<'a, T>;
    type Item = &'a T;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len).iter() }
    }
}

unsafe impl<'a, T: Send> Send for UnsafeSlice<'a, T> {}
unsafe impl<'a, T: Sync> Sync for UnsafeSlice<'a, T> {}

#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct UnsafeSliceMut<'a, T> {
    ptr: *const T,
    len: usize,
    _marker: core::marker::PhantomData<&'a ()>,
}

impl<'a, T> Index<RangeTo<usize>> for UnsafeSliceMut<'a, T> {
    type Output = [T];

    #[inline(always)]
    fn index(&self, index: RangeTo<usize>) -> &Self::Output {
        debug_assert!(index.end < self.len);
        unsafe { core::slice::from_raw_parts(self.ptr, index.end) }
    }
}

impl<'a, T> IndexMut<RangeTo<usize>> for UnsafeSliceMut<'a, T> {
    #[inline(always)]
    fn index_mut(&mut self, index: RangeTo<usize>) -> &mut Self::Output {
        debug_assert!(index.end < self.len);
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut T, index.end) }
    }
}

impl<'a, T> UnsafeSliceMut<'a, T> {
    #[inline(always)]
    pub unsafe fn from_slice(slice: &[T]) -> Self {
        Self::from_raw_parts(slice.as_ptr(), slice.len())
    }

    #[inline(always)]
    pub unsafe fn from_raw_parts(ptr: *const T, len: usize) -> Self {
        Self {
            ptr,
            len,
            _marker: core::marker::PhantomData,
        }
    }

    #[inline(always)]
    pub unsafe fn iter(&self) -> core::slice::Iter<'a, T> {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len).iter() }
    }

    #[inline(always)]
    pub unsafe fn iter_mut(&mut self) -> core::slice::IterMut<'a, T> {
        core::slice::from_raw_parts_mut(self.ptr as *mut T, self.len).iter_mut()
    }

    #[inline(always)]
    pub unsafe fn first(&self) -> &T {
        debug_assert!(self.len > 0);
        &*self.ptr
    }

    #[inline(always)]
    pub unsafe fn last(&self) -> &T {
        debug_assert!(self.len > 0);
        &*self.ptr.add(self.len - 1)
    }

    #[inline(always)]
    pub unsafe fn first_mut(&mut self) -> &mut T {
        debug_assert!(self.len > 0);
        &mut *(self.ptr as *mut T)
    }

    #[inline(always)]
    pub unsafe fn last_mut(&mut self) -> &mut T {
        debug_assert!(self.len > 0);
        &mut *(self.ptr.add(self.len - 1) as *mut T)
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.ptr
    }

    #[inline(always)]
    pub fn as_ptr_mut(&self) -> *mut T {
        self.ptr as *mut T
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &mut [T] {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut T, self.len) }
    }
}

impl<'a, T> AsRef<[T]> for UnsafeSliceMut<'a, T> {
    #[inline(always)]
    fn as_ref(&self) -> &[T] {
        unsafe { core::slice::from_raw_parts(self.ptr, self.len) }
    }
}

impl<'a, T: 'a> IntoIterator for UnsafeSliceMut<'a, T> {
    type IntoIter = core::slice::IterMut<'a, T>;
    type Item = &'a mut T;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        unsafe { core::slice::from_raw_parts_mut(self.ptr as *mut T, self.len).iter_mut() }
    }
}
unsafe impl<'a, T: Send> Send for UnsafeSliceMut<'a, T> {}
unsafe impl<'a, T: Sync> Sync for UnsafeSliceMut<'a, T> {}

pub trait UnsafeSliceExt {
    type Item;
    unsafe fn as_unsafe_slice<'a>(&'a self) -> UnsafeSlice<'a, Self::Item>;
    unsafe fn as_unsafe_slice_mut<'a>(&'a mut self) -> UnsafeSliceMut<'a, Self::Item>;
    unsafe fn as_unsafe_slice_cast_mut<'a>(&'a self) -> UnsafeSliceMut<'a, Self::Item>;
    unsafe fn cast<U>(&self) -> UnsafeSlice<'_, U>;
    unsafe fn cast_mut<U>(&self) -> UnsafeSliceMut<'_, U>;
}

impl<T> UnsafeSliceExt for [T] {
    type Item = T;

    #[inline(always)]
    unsafe fn as_unsafe_slice<'a>(&'a self) -> UnsafeSlice<'a, Self::Item> {
        UnsafeSlice::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_mut<'a>(&'a mut self) -> UnsafeSliceMut<'a, Self::Item> {
        UnsafeSliceMut::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_cast_mut<'a>(&'a self) -> UnsafeSliceMut<'a, Self::Item> {
        UnsafeSliceMut::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn cast<U>(&self) -> UnsafeSlice<'_, U> {
        UnsafeSlice::from_raw_parts(
            self.as_ptr() as *const U,
            self.len() * core::mem::size_of::<T>() / core::mem::size_of::<U>(),
        )
    }

    #[inline(always)]
    unsafe fn cast_mut<U>(&self) -> UnsafeSliceMut<'_, U> {
        UnsafeSliceMut::from_raw_parts(
            self.as_ptr() as *const U,
            self.len() * core::mem::size_of::<T>() / core::mem::size_of::<U>(),
        )
    }
}

impl<T> UnsafeSliceExt for UnsafeSlice<'_, T> {
    type Item = T;

    #[inline(always)]
    unsafe fn as_unsafe_slice<'a>(&'a self) -> UnsafeSlice<'a, Self::Item> {
        UnsafeSlice::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_mut<'a>(&'a mut self) -> UnsafeSliceMut<'a, Self::Item> {
        UnsafeSliceMut::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_cast_mut<'a>(&'a self) -> UnsafeSliceMut<'a, Self::Item> {
        UnsafeSliceMut::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn cast<U>(&self) -> UnsafeSlice<'_, U> {
        UnsafeSlice::from_raw_parts(
            self.as_ptr() as *const U,
            self.len() * core::mem::size_of::<T>() / core::mem::size_of::<U>(),
        )
    }

    #[inline(always)]
    unsafe fn cast_mut<U>(&self) -> UnsafeSliceMut<'_, U> {
        UnsafeSliceMut::from_raw_parts(
            self.as_ptr() as *const U,
            self.len() * core::mem::size_of::<T>() / core::mem::size_of::<U>(),
        )
    }
}

impl<T> UnsafeSliceExt for UnsafeSliceMut<'_, T> {
    type Item = T;

    #[inline(always)]
    unsafe fn as_unsafe_slice<'a>(&'a self) -> UnsafeSlice<'a, Self::Item> {
        UnsafeSlice::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_mut<'a>(&'a mut self) -> UnsafeSliceMut<'a, Self::Item> {
        UnsafeSliceMut::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_cast_mut<'a>(&'a self) -> UnsafeSliceMut<'a, Self::Item> {
        UnsafeSliceMut::from_raw_parts(self.as_ptr(), self.len())
    }

    #[inline(always)]
    unsafe fn cast<U>(&self) -> UnsafeSlice<'_, U> {
        UnsafeSlice::from_raw_parts(
            self.as_ptr() as *const U,
            self.len() * core::mem::size_of::<T>() / core::mem::size_of::<U>(),
        )
    }

    #[inline(always)]
    unsafe fn cast_mut<U>(&self) -> UnsafeSliceMut<'_, U> {
        UnsafeSliceMut::from_raw_parts(
            self.as_ptr() as *const U,
            self.len() * core::mem::size_of::<T>() / core::mem::size_of::<U>(),
        )
    }
}

#[cfg(feature = "cust")]
impl<T: DeviceCopy> UnsafeSliceExt for cust::memory::UnifiedBuffer<T> {
    type Item = T;

    #[inline(always)]
    unsafe fn as_unsafe_slice<'a>(&'a self) -> UnsafeSlice<'a, Self::Item> {
        self.as_slice().as_unsafe_slice()
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_mut<'a>(&'a mut self) -> UnsafeSliceMut<'a, Self::Item> {
        self.as_mut_slice().as_unsafe_slice_mut()
    }

    #[inline(always)]
    unsafe fn as_unsafe_slice_cast_mut<'a>(&'a self) -> UnsafeSliceMut<'a, Self::Item> {
        self.as_slice().as_unsafe_slice_cast_mut()
    }

    #[inline(always)]
    unsafe fn cast<U>(&self) -> UnsafeSlice<'_, U> {
        self.as_slice().cast()
    }

    #[inline(always)]
    unsafe fn cast_mut<U>(&self) -> UnsafeSliceMut<'_, U> {
        self.as_slice().cast_mut()
    }
}
