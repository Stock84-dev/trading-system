#[derive(Default)]
pub struct UnsafeCell<T> {
    inner: core::cell::UnsafeCell<T>,
}

impl<T> UnsafeCell<T> {
    pub fn new(data: T) -> Self {
        Self {
            inner: core::cell::UnsafeCell::new(data),
        }
    }

    pub fn inner_mut(&mut self) -> &mut T {
        unsafe { &mut *self.inner.get() }
    }

    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.inner.get() }
    }

    pub fn get(&self) -> &T {
        unsafe { &*self.inner.get() }
    }

    pub fn into_inner(self) -> T {
        self.inner.into_inner()
    }
}

#[derive(Default)]
pub struct SyncUnsafeCell<T> {
    inner: core::cell::UnsafeCell<T>,
}

impl<T> SyncUnsafeCell<T> {
    pub fn new(data: T) -> Self {
        Self {
            inner: core::cell::UnsafeCell::new(data),
        }
    }

    pub fn inner_mut(&mut self) -> &mut T {
        unsafe { &mut *self.inner.get() }
    }

    pub fn get_mut(&self) -> &mut T {
        unsafe { &mut *self.inner.get() }
    }

    pub fn get(&self) -> &T {
        unsafe { &*self.inner.get() }
    }

    pub fn into_inner(self) -> T {
        self.inner.into_inner()
    }
}

// unsafe impl<T: Send> Send for SyncUnsafeCell<T> {}
unsafe impl<T: Sync> Sync for SyncUnsafeCell<T> {}
