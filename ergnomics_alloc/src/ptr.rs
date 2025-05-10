use core::convert::TryInto;
use core::ops::{Index, IndexMut};

pub struct UnsafePtr<T> {
    ptr: *mut T,
}

impl<T> UnsafePtr<T> {
    #[inline(always)]
    pub const fn new(ptr: *mut T) -> Self {
        Self { ptr }
    }

    #[inline(always)]
    pub fn alloc(n: usize) -> Self {
        unsafe {
            let layout = core::alloc::Layout::array::<T>(n).unwrap_unchecked();
            let ptr = alloc::alloc::alloc(layout) as *mut T;
            Self { ptr }
        }
    }

    #[inline(always)]
    pub fn add(&self, offset: usize) -> Self {
        Self {
            ptr: unsafe { self.ptr.add(offset) },
        }
    }

    #[inline(always)]
    pub fn read(&self) -> T {
        unsafe { core::ptr::read(self.ptr) }
    }

    #[inline(always)]
    pub fn write(&self, value: T) {
        unsafe {
            core::ptr::write(self.ptr, value);
        }
    }

    #[inline(always)]
    pub fn cast<U>(self) -> UnsafePtr<U> {
        UnsafePtr {
            ptr: self.ptr as *mut U,
        }
    }
}

impl<T> core::ops::Deref for UnsafePtr<T> {
    type Target = T;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.ptr }
    }
}

impl<T> core::ops::DerefMut for UnsafePtr<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { &mut *self.ptr }
    }
}

impl<T, I: TryInto<u64>> Index<I> for UnsafePtr<T> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        unsafe { &*self.ptr.add(index.try_into().unwrap_unchecked() as usize) }
    }
}

impl<T, I: TryInto<u64>> IndexMut<I> for UnsafePtr<T> {
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        unsafe { &mut *self.ptr.add(index.try_into().unwrap_unchecked() as usize) }
    }
}

unsafe impl<T> Send for UnsafePtr<T> {}
unsafe impl<T> Sync for UnsafePtr<T> {}

impl<T> Clone for UnsafePtr<T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        Self { ptr: self.ptr }
    }
}

impl<T> Copy for UnsafePtr<T> {}
