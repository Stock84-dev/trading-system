use std::thread::ThreadId;

use ahash::HashMap;
use ergnomics::OptionExt as _;
use slotmap::{DefaultKey, SlotMap};
use tracing_mutex::parking_lot;

const RANDOM_STATE: ahash::RandomState = ahash::RandomState::with_seeds(
    12695871016965803447,
    16407641583885871815,
    16678508885079677638,
    18347295202708696857,
);

static STATE: SyncStates = SyncStates {
    mutexes: parking_lot::Mutex::new(HashMap::with_hasher(RANDOM_STATE)),
    rwlocks: parking_lot::Mutex::new(HashMap::with_hasher(RANDOM_STATE)),
};

fn print() {
    let mutexes = STATE.mutexes.lock();
    for (id, instance) in mutexes.iter() {
        if let Some(instance) = instance {
            println!(
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
            println!(
                "{} read {:?}:\"{}\":{}:{}",
                id,
                instance.thread.id(),
                instance.thread.name().unwrap_or_default(),
                instance.file,
                instance.line
            );
        });
        if let Some(x) = &state.upgradable_read {
            println!(
                "{} upgradable read {:?}:\"{}\":{}:{}",
                id,
                x.thread.id(),
                x.thread.name().unwrap_or_default(),
                x.file,
                x.line
            );
        }
        if let Some(x) = &state.write {
            println!(
                "{} write {:?}:\"{}\":{}:{}",
                id,
                x.thread.id(),
                x.thread.name().unwrap_or_default(),
                x.file,
                x.line
            );
        }
    }
}

struct Instance {
    thread: std::thread::Thread,
    file: &'static str,
    line: u32,
}

struct SyncStates {
    mutexes: parking_lot::Mutex<HashMap<String, Option<Instance>>>,
    rwlocks: parking_lot::Mutex<HashMap<String, RwLockState>>,
}

struct RwLockState {
    read: SlotMap<DefaultKey, Instance>,
    upgradable_read: Option<Instance>,
    write: Option<Instance>,
}

pub struct Mutex<T: ?Sized> {
    #[cfg(debug_assertions)]
    id: String,
    inner: tracing_mutex::parkinglot::Mutex<T>,
}

impl<T> Mutex<T> {
    pub fn new<I: ToString>(t: T, id: impl FnOnce() -> I) -> Self {
        Self {
            inner: tracing_mutex::parkinglot::Mutex::new(t),
            #[cfg(debug_assertions)]
            id: id().to_string(),
        }
    }
}

impl<T: ?Sized> Mutex<T> {
    pub fn __lock(&self) -> MutexGuard<T> {
        MutexGuard {
            guard: self.inner.lock(),
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
        }
    }

    pub fn __try_lock(&self) -> Option<MutexGuard<T>> {
        self.inner.try_lock().map(|guard| MutexGuard {
            guard,
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
        })
    }
}

struct MutexGuard<'a, T: ?Sized> {
    guard: tracing_mutex::parkinglot::MutexGuard<'a, T>,
    #[cfg(debug_assertions)]
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

impl<'a, T: ?Sized> Drop for MutexGuard<'a, T> {
    fn drop(&mut self) {
        let mut state = STATE.mutexes.lock();
        state.get_mut(self.mutex_id).unwrap().take();
    }
}

#[inline]
fn lock(lock_id: &str, instance: Instance) {
    let mut state = STATE.mutexes.lock();
    match state.get_mut(lock_id) {
        Some(x) => {
            *x = Some(instance);
        },
        None => {
            state.insert(lock_id.to_string(), Some(instance));
        },
    }
}

#[macro_export]
macro_rules! lock {
    ($lock:expr) => {{
        let lock_instance = Instance {
            thread: std::thread::current(),
            file: file!(),
            line: line!(),
        };
        let inner = match $lock.__try_lock() {
            Some(inner) => inner,
            None => {
                print();
                lock.__lock()
            },
        };
        $crate::lock(&inner.mutex_id, lock_instance);
        inner
    }};
}

pub struct RwLock<T: ?Sized> {
    #[cfg(debug_assertions)]
    id: String,
    inner: parking_lot::RwLock<T>,
}
impl<T> RwLock<T> {
    pub fn new<I: ToString>(t: T, id: impl FnOnce() -> I) -> Self {
        Self {
            #[cfg(debug_assertions)]
            id: id().to_string(),
            inner: parking_lot::RwLock::new(t),
        }
    }
}

impl<T: ?Sized> RwLock<T> {
    pub fn __read(&self) -> RwLockReadGuard<T> {
        RwLockReadGuard {
            guard: self.inner.read(),
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
            #[cfg(debug_assertions)]
            instance_id: DefaultKey::default(),
        }
    }

    pub fn __try_read(&self) -> Option<RwLockReadGuard<T>> {
        self.inner.try_read().map(|guard| RwLockReadGuard {
            guard,
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
            #[cfg(debug_assertions)]
            instance_id: DefaultKey::default(),
        })
    }

    pub fn __try_write(&self) -> Option<RwLockWriteGuard<T>> {
        self.inner.try_write().map(|guard| RwLockWriteGuard {
            guard,
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
        })
    }

    pub fn __write(&self) -> RwLockWriteGuard<T> {
        RwLockWriteGuard {
            guard: self.inner.write(),
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
        }
    }

    pub fn __try_upgradable_read(&self) -> Option<RwLockUpgradableReadGuard<T>> {
        self.inner.try_upgradable_read().map(|guard| RwLockUpgradableReadGuard {
            guard,
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
        })
    }

    pub fn __upgradable_read(&self) -> RwLockUpgradableReadGuard<T> {
        RwLockUpgradableReadGuard {
            guard: self.inner.upgradable_read(),
            #[cfg(debug_assertions)]
            mutex_id: &self.id,
        }
    }
}

struct RwLockReadGuard<'a, T: ?Sized> {
    guard: parking_lot::RwLockReadGuard<'a, T>,
    #[cfg(debug_assertions)]
    mutex_id: &'a str,
    #[cfg(debug_assertions)]
    instance_id: DefaultKey,
}

impl<'a, T: ?Sized> std::ops::Deref for RwLockReadGuard<'a, T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &*self.guard
    }
}

impl<'a, T: ?Sized> Drop for RwLockReadGuard<'a, T> {
    fn drop(&mut self) {
        let mut state = STATE.rwlocks.lock();
        let state = state.get_mut(self.mutex_id).unwrap();
        state.read.remove(self.instance_id);
    }
}

#[inline]
fn read<T>(guard: &mut RwLockReadGuard<T>, instance: Instance) {
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
        let lock_instance = Instance {
            thread: std::thread::current(),
            file: file!(),
            line: line!(),
        };
        let inner = match $rwlock.__try_read() {
            Some(mut inner) => inner,
            None => {
                print();
                inner = lock.__read()
            },
        };
        $crate::read(&mut inner, lock_instance);
        inner
    }};
}

struct RwLockWriteGuard<'a, T: ?Sized> {
    guard: parking_lot::RwLockWriteGuard<'a, T>,
    #[cfg(debug_assertions)]
    mutex_id: &'a str,
}

#[cfg(debug_assertions)]
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

#[cfg(not(debug_assertions))]
impl<'a, T: ?Sized> RwLockWriteGuard<'a, T> {
    pub fn downgrade(s: Self) -> RwLockReadGuard<'a, T> {
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

impl<'a, T: ?Sized> Drop for RwLockWriteGuard<'a, T> {
    fn drop(&mut self) {
        let mut state = STATE.rwlocks.lock();
        let state = state.get_mut(self.mutex_id).unwrap();
        state.write.take();
    }
}

#[inline]
fn write<T>(guard: &RwLockWriteGuard<T>, instance: Instance) {
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
        let lock_instance = Instance {
            thread: std::thread::current(),
            file: file!(),
            line: line!(),
        };
        let inner = match $rwlock.__try_write() {
            Some(mut inner) => {
                $crate::write(&inner, lock_instance);
                inner
            },
            None => {
                print();
                lock.__write()
            },
        };
        $crate::write(&inner, lock_instance);
        inner
    }};
}

struct RwLockUpgradableReadGuard<'a, T: ?Sized> {
    guard: parking_lot::RwLockUpgradableReadGuard<'a, T>,
    mutex_id: &'a str,
}

impl<'a, T: ?Sized> RwLockUpgradableReadGuard<'a, T> {
    pub fn __try_upgrade(s: Self) -> Result<RwLockWriteGuard<'a, T>, Self> {
        let s = core::mem::ManuallyDrop::new(s);
        let guard = unsafe { core::ptr::read(&s.guard) };
        parking_lot::RwLockUpgradableReadGuard::try_upgrade(guard)
            .map(|mut guard| {
                Self::drop_impl(&s.mutex_id);
                RwLockWriteGuard {
                    guard,
                    mutex_id: s.mutex_id,
                }
            })
            .map_err(|guard| RwLockUpgradableReadGuard {
                guard,
                mutex_id: s.mutex_id,
            })
    }

    pub fn __upgrade(mut s: RwLockUpgradableReadGuard<'a, T>) -> RwLockWriteGuard<'a, T> {
        let s = core::mem::ManuallyDrop::new(s);
        Self::drop_impl(s.mutex_id);
        RwLockWriteGuard {
            guard: parking_lot::RwLockUpgradableReadGuard::upgrade(unsafe {
                core::ptr::read(&s.guard)
            }),
            mutex_id: s.mutex_id,
        }
    }

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

impl<'a, T: ?Sized> Drop for RwLockUpgradableReadGuard<'a, T> {
    fn drop(&mut self) {
        Self::drop_impl(self.mutex_id);
    }
}

#[inline]
fn upgradable_read<T>(guard: &RwLockUpgradableReadGuard<T>, instance: Instance) {
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
        let lock_instance = Instance {
            thread: std::thread::current(),
            file: file!(),
            line: line!(),
        };
        let inner = match $rwlock.__try_upgradable_read() {
            Some(mut inner) => inner,
            None => {
                print();
                lock.__upgradable_read()
            },
        };
        $crate::upgradable_read(&inner, lock_instance);
        inner
    }};
}

#[macro_export]
macro_rules! upgrade {
    ($guard:expr) => {{
        let lock_instance = Instance {
            thread: std::thread::current(),
            file: file!(),
            line: line!(),
        };
        let inner = match $crate::RwLockUpgradableReadGuard::__try_upgrade($guard) {
            Ok(inner) => inner,
            Err(mut inner) => {
                print();
                $crate::RwLockUpgradableReadGuard::__upgrade(inner)
            },
        };
        $crate::write(&mut inner, lock_instance);
        inner,
    }};
}
