#![no_std]
#![feature(const_mut_refs)]
#![feature(const_ptr_write)]
pub use derive_new::new;

pub mod futures;

pub mod items {
    #[repr(align(64))]
    pub struct Align<T>(pub T);
}

pub trait Cmov {
    fn cmov(&mut self, cond: bool, other: Self);
}

impl<T: Copy> Cmov for T {
    #[inline(always)]
    /// Force a conditional move. CUDA compiler never does this.
    fn cmov(&mut self, cond: bool, other: Self) {
        *self = if cond { other } else { *self };
    }
}

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

pub struct UnsafeArray<T, const LEN: usize> {
    data: [T; LEN],
}

impl<T, const LEN: usize> UnsafeArray<T, LEN> {
    #[inline(always)]
    pub const fn splat(value: T) -> Self
    where
        T: Copy,
    {
        Self { data: [value; LEN] }
    }

    #[inline(always)]
    pub const fn zeroed() -> Self {
        Self {
            data: unsafe { core::mem::zeroed() },
        }
    }

    #[inline(always)]
    pub const fn uninit() -> Self {
        Self {
            data: unsafe { core::mem::MaybeUninit::uninit().assume_init() },
        }
    }

    #[inline(always)]
    pub const fn write(&mut self, index: usize, value: T) {
        unsafe {
            core::ptr::write(self.data.as_mut_ptr().add(index), value);
        }
    }

    #[inline(always)]
    pub const fn as_mut_ptr(&mut self) -> *mut T {
        self.data.as_mut_ptr()
    }

    #[inline(always)]
    pub const fn as_ptr(&self) -> *const T {
        self.data.as_ptr()
    }

    #[inline(always)]
    pub const fn as_slice(&self) -> &[T] {
        &self.data
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        LEN
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T, const N: usize> core::ops::Deref for UnsafeArray<T, N> {
    type Target = [T];

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<T, const N: usize> core::ops::DerefMut for UnsafeArray<T, N> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<T, I: TryInto<usize>, const N: usize> core::ops::Index<I> for UnsafeArray<T, N> {
    type Output = T;

    #[inline(always)]
    fn index(&self, index: I) -> &Self::Output {
        unsafe { self.data.get_unchecked(index.try_into().unwrap_unchecked()) }
    }
}

impl<T, I: TryInto<usize>, const N: usize> core::ops::IndexMut<I> for UnsafeArray<T, N> {
    #[inline(always)]
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        unsafe { self.data.get_unchecked_mut(index.try_into().unwrap_unchecked()) }
    }
}

impl<T, const N: usize> IntoIterator for UnsafeArray<T, N> {
    type IntoIter = core::array::IntoIter<T, N>;
    type Item = T;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a UnsafeArray<T, N> {
    type IntoIter = core::slice::Iter<'a, T>;
    type Item = &'a T;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter()
    }
}

impl<'a, T, const N: usize> IntoIterator for &'a mut UnsafeArray<T, N> {
    type IntoIter = core::slice::IterMut<'a, T>;
    type Item = &'a mut T;

    #[inline(always)]
    fn into_iter(self) -> Self::IntoIter {
        self.data.iter_mut()
    }
}

