#![no_std]
use alloc::string::String;
use alloc::vec::Vec;
use core::iter::FromIterator;

pub use cell::*;
pub use ergnomics_gpu::*;
pub use ergnomics_alloc::*;
use itertools::MultiUnzip;
pub use itertools::{self, chain, iproduct, izip, Itertools};
pub use num::*;
use num_traits::float::FloatCore;
use num_traits::AsPrimitive;
pub use tracing::*;

extern crate alloc;

#[macro_export]
macro_rules! debug_trace {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $crate::trace!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! debug_debug {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $crate::debug!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! debug_info {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $crate::info!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! debug_warn {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $crate::warn!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! debug_error {
    ($($arg:tt)*) => {
        #[cfg(debug_assertions)]
        {
            $crate::error!($($arg)*);
        }
    };
}

#[macro_export]
macro_rules! some_loop {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => continue,
        }
    };
}

#[macro_export]
macro_rules! some {
    ($e:expr) => {
        match $e {
            Some(x) => x,
            None => return,
        }
    };
}

mod cell;
mod helpers;
mod num;

pub trait ResultExt {
    fn ignore(self);
}

impl<T, E> ResultExt for Result<T, E> {
    #[inline(always)]
    fn ignore(self) {}
}

pub trait OptionExt {
    type Item;
    fn expect_or_else<F: FnOnce() -> S, S: Into<String>>(self, f: F) -> Self::Item;
    fn some(self) -> Result<Self::Item, eyre::Error>;
}

impl<T> OptionExt for Option<T> {
    type Item = T;

    #[inline(always)]
    fn expect_or_else<F: FnOnce() -> S, S: Into<String>>(self, f: F) -> Self::Item {
        match self {
            Some(x) => x,
            None => panic!("{}", f().into()),
        }
    }

    #[inline(always)]
    fn some(self) -> Result<Self::Item, eyre::Error> {
        self.ok_or(eyre::eyre!("called `Option::unwrap()` on a `None` value"))
    }
}

pub trait TypeExt {
    fn type_name() -> &'static str;
}

impl<T: ?Sized> TypeExt for T {
    #[inline(always)]
    fn type_name() -> &'static str {
        core::any::type_name::<T>()
    }
}

pub trait TypeIdExt {
    fn type_id() -> core::any::TypeId;
}

impl<T: ?Sized + 'static> TypeIdExt for T {
    #[inline(always)]
    fn type_id() -> core::any::TypeId {
        core::any::TypeId::of::<T>()
    }
}

pub trait BoolExt {
    fn to_f32(self) -> f32;
    fn to_f64(self) -> f64;
}

impl BoolExt for bool {
    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as u8 as f32
    }

    #[inline(always)]
    fn to_f64(self) -> f64 {
        self as u8 as f64
    }
}

pub trait FloatExt {
    fn positive(self) -> Self;
    fn negative(self) -> Self;
}

impl<T: FloatCore + 'static> FloatExt for T
where
    u8: AsPrimitive<T>,
{
    fn positive(self) -> Self {
        // This is twice as fast than below version
        ((self > Self::zero()) as u8).as_()
        // if self.is_sign_positive() {
        //     Self::one()
        // } else {
        //     Self::zero()
        // }
    }

    fn negative(self) -> Self {
        // This is twice as fast than below version
        ((self < Self::zero()) as u8).as_()
        // if self.is_sign_negative() {
        //     Self::one()
        // } else {
        //     Self::zero()
        // }
    }
}

#[inline(always)]
pub fn chunked<const CHUNK_SIZE: usize, F: FnMut(usize, usize)>(n: usize, mut f: F) {
    let mut i = 0;
    while i < n - n % CHUNK_SIZE {
        for j in 0..CHUNK_SIZE {
            f(i, j);
        }
        i += CHUNK_SIZE;
    }
    for j in 0..n % CHUNK_SIZE {
        f(i, j);
    }
}

pub trait IterToolsUnzipExt<T>: Itertools {
    fn unzip_vec(self) -> T;
}

macro_rules! impl_iter_tools_ext {
    ($($name:ident),*) => {
        impl<I: MultiUnzip<($(Vec<$name>,)*)> $(, $name)*> IterToolsUnzipExt<($(Vec<$name>,)*)> for I {
            fn unzip_vec(self) -> ($(Vec<$name>,)*) {
                self.multiunzip()
            }
        }
    };
}

all_tuples::all_tuples!(impl_iter_tools_ext, 2, 7, T);

pub trait IterToolsExt: Itertools {
    fn try_collect_vec<T, U, E>(self) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
        Result<U, E>: FromIterator<Result<T, E>>;
}

impl<I: Itertools> IterToolsExt for I {
    fn try_collect_vec<T, U, E>(self) -> Result<Vec<T>, E>
    where
        Self: Sized + Iterator<Item = Result<T, E>>,
        Result<U, E>: FromIterator<Result<T, E>>,
    {
        self.collect::<Result<Vec<T>, E>>()
    }
}

pub trait Transmutations: Sized {
    fn as_one_slice<'a>(&'a self) -> &'a [Self];
    fn as_one_slice_mut<'a>(&'a mut self) -> &'a mut [Self];
    fn as_u8_slice<'a>(&'a self) -> &'a [u8];
    unsafe fn as_u8_slice_mut<'a>(&'a mut self) -> &'a mut [u8];
    unsafe fn as_static<'a>(&'a self) -> &'static Self;
    unsafe fn as_static_mut<'a>(&'a mut self) -> &'static mut Self;
    unsafe fn as_mut_cast<'a>(&'a self) -> &'a mut Self;
    unsafe fn from_u8_slice<'a>(slice: &'a [u8]) -> &'a Self;
    unsafe fn from_u8_slice_mut<'a>(slice: &'a mut [u8]) -> &'a mut Self;
}

impl<T: Sized> Transmutations for T {
    fn as_one_slice<'a>(&'a self) -> &'a [Self] {
        unsafe { helpers::ptr_as_slice(self as *const Self, 1) }
    }

    fn as_one_slice_mut<'a>(&'a mut self) -> &'a mut [Self] {
        unsafe { helpers::ptr_as_slice_mut(self as *mut Self, 1) }
    }

    fn as_u8_slice<'a>(&'a self) -> &'a [u8] {
        unsafe { helpers::ptr_as_slice(self as *const Self, T::size()) }
    }

    unsafe fn as_u8_slice_mut<'a>(&'a mut self) -> &'a mut [u8] {
        helpers::ptr_as_slice_mut(self as *mut Self, T::size())
    }

    unsafe fn as_static<'a>(&'a self) -> &'static Self {
        core::mem::transmute(self)
    }

    unsafe fn as_static_mut<'a>(&'a mut self) -> &'static mut Self {
        core::mem::transmute(self)
    }

    unsafe fn as_mut_cast<'a>(&'a self) -> &'a mut Self {
        #[allow(invalid_reference_casting)]
        {
            &mut *(self as *const _ as *mut Self)
        }
    }

    unsafe fn from_u8_slice<'a>(slice: &'a [u8]) -> &'a Self {
        &*(slice.as_ptr() as *const Self)
    }

    unsafe fn from_u8_slice_mut<'a>(slice: &'a mut [u8]) -> &'a mut Self {
        &mut *(slice.as_mut_ptr() as *mut Self)
    }
}

pub trait ConstLen {
    const LEN: usize;
}

impl<T, const N: usize> ConstLen for [T; N] {
    const LEN: usize = N;
}

pub trait OptionExtResult<'a, O: Default, E> {
    type Nullable: 'a;
    fn on_mut_result(
        &'a mut self,
        callback: impl FnMut(&'a mut Self::Nullable) -> Result<O, E>,
    ) -> Result<O, E>;
    fn on_ref_result(
        &'a self,
        callback: impl FnMut(&'a Self::Nullable) -> Result<O, E>,
    ) -> Result<O, E>;
}

impl<'a, T: 'a, O: Default, E> OptionExtResult<'a, O, E> for Option<T> {
    type Nullable = T;

    fn on_mut_result(
        &'a mut self,
        mut callback: impl FnMut(&'a mut Self::Nullable) -> Result<O, E>,
    ) -> Result<O, E> {
        if let Some(nullable) = self.as_mut() {
            callback(nullable)
        } else {
            Ok(Default::default())
        }
    }

    fn on_ref_result(
        &'a self,
        mut callback: impl FnMut(&'a Self::Nullable) -> Result<O, E>,
    ) -> Result<O, E> {
        if let Some(nullable) = self.as_ref() {
            callback(nullable)
        } else {
            Ok(Default::default())
        }
    }
}

pub trait Alignment {
    fn align() -> usize;
}

impl<T> Alignment for T {
    fn align() -> usize {
        core::mem::align_of::<T>()
    }
}
