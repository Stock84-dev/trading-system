#[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
use cuda_std::intrinsics;
use num_traits::Float;
pub trait MinMaxExt {
    fn max_mut(&mut self, value: Self);
    fn min_mut(&mut self, value: Self);
}

macro_rules! impl_normal_min_max {
    ($($t:ty),*) => {
        $(
            impl MinMaxExt for $t {
                #[inline(always)]
                fn max_mut(&mut self, value: $t) {
                    *self = value.max(*self);
                }

                #[inline(always)]
                fn min_mut(&mut self, value: $t) {
                    *self = value.min(*self);
                }
            }
        )*
    };
}

impl_normal_min_max!(i8, i16, u8, u16, i128, u128, isize, usize);

macro_rules! impl_cuda_min_max {
    ($t:ty, $min:ident, $max:ident) => {
        impl MinMaxExt for $t {
            #[inline(always)]
            fn max_mut(&mut self, value: $t) {
                #[cfg(not(any(target_arch = "nvptx", target_arch = "nvptx64")))]
                let val = value.max(*self);
                #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
                let val = unsafe { cuda_std::intrinsics::$max(*self, value) };
                *self = val;
            }

            #[inline(always)]
            fn min_mut(&mut self, value: $t) {
                #[cfg(not(any(target_arch = "nvptx", target_arch = "nvptx64")))]
                let val = value.min(*self);
                #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
                let val = unsafe { cuda_std::intrinsics::$min(*self, value) };
                *self = val;
            }
        }
    };
}

impl_cuda_min_max!(i32, min, max);
impl_cuda_min_max!(i64, llmin, llmax);
impl_cuda_min_max!(f32, fminf, fmaxf);
impl_cuda_min_max!(f64, fmin, fmax);

macro_rules! impl_cuda_min_max_unsigned {
    ($u:ty, $i:ty, $min:ident, $max:ident) => {
        impl MinMaxExt for $u {
            #[inline(always)]
            fn max_mut(&mut self, value: $u) {
                #[cfg(not(any(target_arch = "nvptx", target_arch = "nvptx64")))]
                let val = value.max(*self);
                #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
                let val;
                #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
                unsafe {
                    // don't know why types are signed
                    let s: $i = core::mem::transmute(*self);
                    let v: $i = core::mem::transmute(value);
                    let v: $u = core::mem::transmute(cuda_std::intrinsics::$max(s, v));
                    val = v;
                }
                *self = val;
            }

            #[inline(always)]
            fn min_mut(&mut self, value: $u) {
                #[cfg(not(any(target_arch = "nvptx", target_arch = "nvptx64")))]
                let val = value.min(*self);
                #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
                let val;
                #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
                unsafe {
                    let s: $i = core::mem::transmute(*self);
                    let v: $i = core::mem::transmute(value);
                    let v: $u = core::mem::transmute(cuda_std::intrinsics::$min(s, v));
                    val = v;
                }
                *self = val;
            }
        }
    };
}

impl_cuda_min_max_unsigned!(u32, i32, umax, umin);
impl_cuda_min_max_unsigned!(u64, i64, ullmax, ullmin);

// pub trait NumExt<T = Self> {
//     fn max_mut(&mut self, value: T);
//     fn min_mut(&mut self, value: T);
//     fn average(&mut self, new_entry: T, new_count: T);
//     /// NOTE: only works for integers. If it is divisible by specified number and has remainder
// then     /// increases self to the number that is divisible by specified number.
//     fn div_ceil(&self, denominator: T) -> Self;
//     // returns true if number is within <base * (1 - rel), base * (1 + rel)>
//     fn within_percent(&self, base: T, rel: T) -> bool;
// }

// impl NumExt<f32> for f32 {
//     #[inline(always)]
//     fn max_mut(&mut self, value: f32) {
//         #[cfg(not(any(target_arch = "nvptx", target_arch = "nvptx64")))]
//         let val = self.max(value);
//         #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
//         let val = unsafe { cuda_std::intrinsics::fmaxf(*self, value) };
//         *self = val;
//     }
//
//     #[inline(always)]
//     fn min_mut(&mut self, value: f32) {
//         #[cfg(not(any(target_arch = "nvptx", target_arch = "nvptx64")))]
//         let val = self.min(value);
//         #[cfg(any(target_arch = "nvptx", target_arch = "nvptx64"))]
//         let val = unsafe { cuda_std::intrinsics::fminf(*self, value) };
//         *self = val;
//     }
//
//     #[inline(always)]
//     fn average(&mut self, new_entry: f32, new_count: f32) {
//         *self = *self * ((new_count - 1.) / new_count) + new_entry / new_count;
//     }
//
//     #[inline(always)]
//     fn div_ceil(&self, denominator: f32) -> Self {
//         (*self + denominator - 1.) / denominator
//     }
//
//     #[inline(always)]
//     fn within_percent(&self, base: f32, rel: f32) -> bool {
//         *self > base * (1. - rel) && *self < base * (1. + rel)
//     }
// }
