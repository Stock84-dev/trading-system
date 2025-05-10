#![feature(const_collections_with_hasher)]
#[cfg(feature = "trace")]
use slotmap::DefaultKey;
#[cfg(feature = "trace")]
use traced::STATE;
#[cfg(feature = "trace")]
pub use traced::{
    __print, lock, mutex_id, read, rwlock_id, rwlockguard_id, upgradable_read, write, Instance,
};
// use tracing_mutex::parking_lot;
pub use triomphe::*;
// pub mod byte_mpsc;

#[cfg(feature = "trace")]
mod traced {
    use std::io::Write;

    use ahash::HashMap;
    use slotmap::{DefaultKey, SlotMap};

    // use tracing_mutex::parking_lot;
    use crate::{
        Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockUpgradableReadGuard, RwLockWriteGuard,
    };

    const RANDOM_STATE: ahash::RandomState = ahash::RandomState::with_seeds(
        12695871016965803447,
        16407641583885871815,
        16678508885079677638,
        18347295202708696857,
    );

    pub static STATE: SyncStates = SyncStates {
        mutexes: parking_lot::Mutex::new(HashMap::with_hasher(RANDOM_STATE)),
        rwlocks: parking_lot::Mutex::new(HashMap::with_hasher(RANDOM_STATE)),
    };

    pub fn __print(blocked_id: &str, blocked: &Instance) {
        let mutexes = STATE.mutexes.lock();
        let mut out = std::io::stdout().lock();
        for (id, instance) in mutexes.iter() {
            if let Some(instance) = instance {
                std::writeln!(
                    &mut out,
                    "{} locked {:?}:\"{}\":{}:{}",
                    id,
                    instance.thread.id(),
                    instance.thread.name().unwrap_or_default(),
                    instance.file,
                    instance.line
                );
            }
        }
        drop(mutexes);
        let rwlocks = STATE.rwlocks.lock();
        for (id, state) in rwlocks.iter() {
            state.read.iter().for_each(|(_, instance)| {
                writeln!(
                    &mut out,
                    "{} read {:?}:\"{}\":{}:{}",
                    id,
                    instance.thread.id(),
                    instance.thread.name().unwrap_or_default(),
                    instance.file,
                    instance.line
                );
            });
            if let Some(x) = &state.upgradable_read {
                writeln!(
                    &mut out,
                    "{} upgradable read {:?}:\"{}\":{}:{}",
                    id,
                    x.thread.id(),
                    x.thread.name().unwrap_or_default(),
                    x.file,
                    x.line
                );
            }
            if let Some(x) = &state.write {
                writeln!(
                    &mut out,
                    "{} write {:?}:\"{}\":{}:{}",
                    id,
                    x.thread.id(),
                    x.thread.name().unwrap_or_default(),
                    x.file,
                    x.line
                );
            }
            writeln!(
                &mut out,
                "{} blocked {:?}:\"{}\":{}:{}\n",
                blocked_id,
                blocked.thread.id(),
                blocked.thread.name().unwrap_or_default(),
                blocked.file,
                blocked.line
            );
        }
    }

    pub struct Instance {
        pub thread: std::thread::Thread,
        pub file: &'static str,
        pub line: u32,
    }

    pub struct SyncStates {
        pub mutexes: parking_lot::Mutex<HashMap<String, Option<Instance>>>,
        pub rwlocks: parking_lot::Mutex<HashMap<String, RwLockState>>,
    }

    pub struct RwLockState {
        pub read: SlotMap<DefaultKey, Instance>,
        pub upgradable_read: Option<Instance>,
        pub write: Option<Instance>,
    }

    impl<'a, T: ?Sized> Drop for MutexGuard<'a, T> {
        fn drop(&mut self) {
            let mut state = STATE.mutexes.lock();
            state.get_mut(self.mutex_id).unwrap().take();
        }
    }

    #[inline]
    pub fn mutex_id<T>(lock: &Mutex<T>) -> &str {
        &lock.id
    }

    #[inline]
    pub fn lock<T>(guard: &MutexGuard<T>, instance: Instance) {
        let mut state = STATE.mutexes.lock();
        match state.get_mut(guard.mutex_id) {
            Some(x) => {
                *x = Some(instance);
            },
            None => {
                state.insert(guard.mutex_id.to_string(), Some(instance));
            },
        }
    }

    #[macro_export]
    macro_rules! lock {
        ($lock:expr) => {{
            let lock_instance = $crate::Instance {
                thread: std::thread::current(),
                file: file!(),
                line: line!(),
            };
            let inner = match $lock.__try_lock() {
                Some(inner) => inner,
                None => {
                    $crate::__print($crate::mutex_id(&$lock), &lock_instance);
                    $lock.__lock()
                },
            };
            $crate::lock(&inner, lock_instance);
            inner
        }};
    }

    impl<'a, T: ?Sized> Drop for RwLockReadGuard<'a, T> {
        fn drop(&mut self) {
            let mut state = STATE.rwlocks.lock();
            let state = state.get_mut(self.mutex_id).unwrap();
            state.read.remove(self.instance_id);
        }
    }

    #[inline]
    pub fn rwlock_id<T>(guard: &RwLock<T>) -> &str {
        &guard.id
    }

    #[inline]
    pub fn rwlockguard_id<'a, T>(guard: &RwLockReadGuard<'a, T>) -> &'a str {
        guard.mutex_id
    }

    #[inline]
    pub fn read<T>(guard: &mut RwLockReadGuard<T>, instance: Instance) {
        let mut state = STATE.rwlocks.lock();
        match state.get_mut(guard.mutex_id) {
            Some(x) => {
                guard.instance_id = x.read.insert(instance);
            },
            None => {
                let mut map = SlotMap::new();
                guard.instance_id = map.insert(instance);
                state.insert(
                    guard.mutex_id.to_string(),
                    RwLockState {
                        read: map,
                        upgradable_read: None,
                        write: None,
                    },
                );
            },
        };
    }

    #[macro_export]
    macro_rules! read {
        ($rwlock:expr) => {{
            let lock_instance = $crate::Instance {
                thread: std::thread::current(),
                file: file!(),
                line: line!(),
            };
            let mut inner = match $rwlock.__try_read() {
                Some(mut inner) => inner,
                None => {
                    $crate::__print($crate::rwlock_id(&$rwlock), &lock_instance);
                    $rwlock.__read()
                },
            };
            $crate::read(&mut inner, lock_instance);
            inner
        }};
    }

    impl<'a, T: ?Sized> Drop for RwLockWriteGuard<'a, T> {
        fn drop(&mut self) {
            let mut state = STATE.rwlocks.lock();
            let state = state.get_mut(self.mutex_id).unwrap();
            state.write.take();
        }
    }

    #[inline]
    pub fn write<T>(guard: &RwLockWriteGuard<T>, instance: Instance) {
        let mut state = STATE.rwlocks.lock();
        match state.get_mut(guard.mutex_id) {
            Some(x) => {
                x.write = Some(instance);
            },
            None => {
                state.insert(
                    guard.mutex_id.to_string(),
                    RwLockState {
                        read: SlotMap::new(),
                        upgradable_read: None,
                        write: Some(instance),
                    },
                );
            },
        };
    }

    #[macro_export]
    macro_rules! write {
        ($rwlock:expr) => {{
            let lock_instance = $crate::Instance {
                thread: std::thread::current(),
                file: file!(),
                line: line!(),
            };
            let inner = match $rwlock.__try_write() {
                Some(mut inner) => inner,
                None => {
                    $crate::__print($crate::rwlock_id(&$rwlock), &lock_instance);
                    $rwlock.__write()
                },
            };
            $crate::write(&inner, lock_instance);
            inner
        }};
    }

    impl<'a, T: ?Sized> Drop for RwLockUpgradableReadGuard<'a, T> {
        fn drop(&mut self) {
            Self::drop_impl(self.mutex_id);
        }
    }

    #[inline]
    pub fn upgradable_read<T>(guard: &RwLockUpgradableReadGuard<T>, instance: Instance) {
        let mut state = STATE.rwlocks.lock();
        match state.get_mut(guard.mutex_id) {
            Some(x) => {
                x.upgradable_read = Some(instance);
            },
            None => {
                state.insert(
                    guard.mutex_id.to_string(),
                    RwLockState {
                        read: SlotMap::new(),
                        upgradable_read: Some(instance),
                        write: None,
                    },
                );
            },
        };
    }

    #[macro_export]
    macro_rules! upgradable_read {
        ($rwlock:expr) => {{
            let lock_instance = $crate::Instance {
                thread: std::thread::current(),
                file: file!(),
                line: line!(),
            };
            let inner = match $rwlock.__try_upgradable_read() {
                Some(mut inner) => inner,
                None => {
                    $crate::__print($crate::rwlock_id(&$rwlock, &lock_instance));
                    $rwlock.__upgradable_read()
                },
            };
            $crate::upgradable_read(&inner, lock_instance);
            inner
        }};
    }

    #[macro_export]
    macro_rules! upgrade {
    ($guard:expr) => {{
        let lock_instance = $crate::Instance {
            thread: std::thread::current(),
            file: file!(),
            line: line!(),
        };
        let inner = match $crate::RwLockUpgradableReadGuard::__try_upgrade($guard) {
            Ok(inner) => inner,
            Err(mut inner) => {
                $crate::__print($crate::rwlockguard_id(&$guard), &lock_instance);
                $crate::RwLockUpgradableReadGuard::__upgrade(inner)
            },
        };
        $crate::write(&mut inner, lock_instance);
        inner,
    }};
}
}

#[cfg(not(feature = "trace"))]
mod normal {
    #[macro_export]
    macro_rules! lock {
        ($lock:expr) => {
            $lock.__lock()
        };
    }

    #[macro_export]
    macro_rules! read {
        ($lock:expr) => {
            $lock.__read()
        };
    }

    #[macro_export]
    macro_rules! write {
        ($lock:expr) => {
            $lock.__write()
        };
    }

    #[macro_export]
    macro_rules! upgradable_read {
        ($lock:expr) => {
            $lock.__upgradable_read()
        };
    }

    #[macro_export]
    macro_rules! upgrade {
        ($lock:expr) => {
            $lock.__upgrade()
        };
    }
}

pub struct Mutex<T: ?Sized> {
    #[cfg(feature = "trace")]
    id: String,
    inner: parking_lot::Mutex<T>,
}

impl<T> Mutex<T> {
    pub fn new<I: ToString>(t: T, _id: impl FnOnce() -> I) -> Self {
        Self {
            inner: parking_lot::Mutex::new(t),
            #[cfg(feature = "trace")]
            id: _id().to_string(),
        }
    }
}

impl<T: ?Sized> Mutex<T> {
    pub fn __lock(&self) -> MutexGuard<T> {
        MutexGuard {
            guard: self.inner.lock(),
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
        }
    }

    pub fn __try_lock(&self) -> Option<MutexGuard<T>> {
        self.inner.try_lock().map(|guard| MutexGuard {
            guard,
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
        })
    }
}

pub struct MutexGuard<'a, T: ?Sized> {
    guard: parking_lot::MutexGuard<'a, T>,
    #[cfg(feature = "trace")]
    mutex_id: &'a str,
}

impl<'a, T: ?Sized> std::ops::Deref for MutexGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.guard
    }
}

impl<'a, T: ?Sized> std::ops::DerefMut for MutexGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.guard
    }
}

pub struct RwLock<T: ?Sized> {
    #[cfg(feature = "trace")]
    id: String,
    inner: parking_lot::RwLock<T>,
}
impl<T> RwLock<T> {
    pub fn new<I: ToString>(t: T, _id: impl FnOnce() -> I) -> Self {
        Self {
            #[cfg(feature = "trace")]
            id: _id().to_string(),
            inner: parking_lot::RwLock::new(t),
        }
    }
}

impl<T: ?Sized> RwLock<T> {
    pub fn __read(&self) -> RwLockReadGuard<T> {
        RwLockReadGuard {
            guard: self.inner.read(),
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
            #[cfg(feature = "trace")]
            instance_id: DefaultKey::default(),
        }
    }

    pub fn __try_read(&self) -> Option<RwLockReadGuard<T>> {
        self.inner.try_read().map(|guard| RwLockReadGuard {
            guard,
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
            #[cfg(feature = "trace")]
            instance_id: DefaultKey::default(),
        })
    }

    pub fn __try_write(&self) -> Option<RwLockWriteGuard<T>> {
        self.inner.try_write().map(|guard| RwLockWriteGuard {
            guard,
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
        })
    }

    pub fn __write(&self) -> RwLockWriteGuard<T> {
        RwLockWriteGuard {
            guard: self.inner.write(),
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
        }
    }

    pub fn __try_upgradable_read(&self) -> Option<RwLockUpgradableReadGuard<T>> {
        self.inner.try_upgradable_read().map(|guard| RwLockUpgradableReadGuard {
            guard,
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
        })
    }

    pub fn __upgradable_read(&self) -> RwLockUpgradableReadGuard<T> {
        RwLockUpgradableReadGuard {
            guard: self.inner.upgradable_read(),
            #[cfg(feature = "trace")]
            mutex_id: &self.id,
        }
    }
}

pub struct RwLockReadGuard<'a, T: ?Sized> {
    guard: parking_lot::RwLockReadGuard<'a, T>,
    #[cfg(feature = "trace")]
    mutex_id: &'a str,
    #[cfg(feature = "trace")]
    instance_id: DefaultKey,
}

impl<'a, T: ?Sized> std::ops::Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.guard
    }
}

#[cfg(feature = "trace")]
impl<'a, T: ?Sized> RwLockWriteGuard<'a, T> {
    pub fn downgrade(s: Self) -> RwLockReadGuard<'a, T> {
        let mut state = STATE.rwlocks.lock();
        let state = state.get_mut(s.mutex_id).unwrap();
        let instance = state.write.take().unwrap();
        let instance_id = state.read.insert(instance);
        let s = core::mem::ManuallyDrop::new(s);
        let guard = unsafe { core::ptr::read(&s.guard) };

        RwLockReadGuard {
            guard: parking_lot::RwLockWriteGuard::downgrade(guard),
            mutex_id: s.mutex_id,
            instance_id,
        }
    }

    pub fn downgrade_to_upgradable(s: Self) -> RwLockUpgradableReadGuard<'a, T> {
        let mut state = STATE.rwlocks.lock();
        let state = state.get_mut(s.mutex_id).unwrap();
        let instance = state.write.take().unwrap();
        state.upgradable_read = Some(instance);
        let s = core::mem::ManuallyDrop::new(s);
        let guard = unsafe { core::ptr::read(&s.guard) };

        RwLockUpgradableReadGuard {
            guard: parking_lot::RwLockWriteGuard::downgrade_to_upgradable(guard),
            mutex_id: s.mutex_id,
        }
    }
}

#[cfg(not(feature = "trace"))]
impl<'a, T: ?Sized> RwLockWriteGuard<'a, T> {
    pub fn downgrade(s: Self) -> RwLockReadGuard<'a, T> {
        let s = core::mem::ManuallyDrop::new(s);
        let guard = unsafe { core::ptr::read(&s.guard) };

        RwLockReadGuard {
            guard: parking_lot::RwLockWriteGuard::downgrade(guard),
        }
    }

    pub fn downgrade_to_upgradable(s: Self) -> RwLockUpgradableReadGuard<'a, T> {
        let s = core::mem::ManuallyDrop::new(s);
        let guard = unsafe { core::ptr::read(&s.guard) };

        RwLockUpgradableReadGuard {
            guard: parking_lot::RwLockWriteGuard::downgrade_to_upgradable(guard),
        }
    }
}

pub struct RwLockWriteGuard<'a, T: ?Sized> {
    guard: parking_lot::RwLockWriteGuard<'a, T>,
    #[cfg(feature = "trace")]
    mutex_id: &'a str,
}

impl<'a, T: ?Sized> std::ops::Deref for RwLockWriteGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.guard
    }
}

impl<'a, T: ?Sized> std::ops::DerefMut for RwLockWriteGuard<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut *self.guard
    }
}

pub struct RwLockUpgradableReadGuard<'a, T: ?Sized> {
    guard: parking_lot::RwLockUpgradableReadGuard<'a, T>,
    #[cfg(feature = "trace")]
    mutex_id: &'a str,
}

impl<'a, T: ?Sized> RwLockUpgradableReadGuard<'a, T> {
    pub fn __try_upgrade(s: Self) -> Result<RwLockWriteGuard<'a, T>, Self> {
        let s = core::mem::ManuallyDrop::new(s);
        let guard = unsafe { core::ptr::read(&s.guard) };
        parking_lot::RwLockUpgradableReadGuard::try_upgrade(guard)
            .map(|guard| {
                #[cfg(feature = "trace")]
                Self::drop_impl(&s.mutex_id);
                RwLockWriteGuard {
                    guard,
                    #[cfg(feature = "trace")]
                    mutex_id: s.mutex_id,
                }
            })
            .map_err(|guard| RwLockUpgradableReadGuard {
                guard,
                #[cfg(feature = "trace")]
                mutex_id: s.mutex_id,
            })
    }

    pub fn __upgrade(s: RwLockUpgradableReadGuard<'a, T>) -> RwLockWriteGuard<'a, T> {
        let s = core::mem::ManuallyDrop::new(s);
        #[cfg(feature = "trace")]
        Self::drop_impl(s.mutex_id);
        RwLockWriteGuard {
            guard: parking_lot::RwLockUpgradableReadGuard::upgrade(unsafe {
                core::ptr::read(&s.guard)
            }),
            #[cfg(feature = "trace")]
            mutex_id: s.mutex_id,
        }
    }

    #[cfg(feature = "trace")]
    fn drop_impl(mutex_id: &str) {
        let mut state = STATE.rwlocks.lock();
        let state = state.get_mut(mutex_id).unwrap();
        state.upgradable_read.take();
    }
}

impl<'a, T: ?Sized> std::ops::Deref for RwLockUpgradableReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.guard
    }
}
