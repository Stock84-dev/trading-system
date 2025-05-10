use core::ops::{Deref, DerefMut, Index, IndexMut};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UnsafeArray<T, const N: usize> {
    inner: [T; N],
}

impl<T: Default, const N: usize> Default for UnsafeArray<T, N>
where
    [T; N]: Default,
{
    #[inline(always)]
    fn default() -> Self {
        Self {
            inner: Default::default(),
        }
    }
}

impl<T, const N: usize> UnsafeArray<T, N> {
    #[inline(always)]
    pub fn uninitialized() -> Self {
        unsafe {
            Self {
                inner: core::mem::MaybeUninit::uninit().assume_init(),
            }
        }
    }

    #[inline(always)]
    pub fn set(&self, index: usize, value: T) {
        unsafe {
            (self.inner.as_ptr().add(index) as *mut T).write(value);
        }
    }

    #[inline(always)]
    pub fn new(inner: [T; N]) -> Self {
        Self { inner }
    }
}

impl<T, I: TryInto<usize>, const N: usize> Index<I> for UnsafeArray<T, N> {
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

impl<T, I: TryInto<usize>, const N: usize> IndexMut<I> for UnsafeArray<T, N> {
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

impl<T, const N: usize> Deref for UnsafeArray<T, N> {
    type Target = [T; N];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T, const N: usize> DerefMut for UnsafeArray<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
