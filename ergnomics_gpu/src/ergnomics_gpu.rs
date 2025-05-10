#![no_std]

// #[cfg(target_os = "cuda")]
// macro_rules! println {
//     ($($arg:tt)*) => {};
// }

// #[macro_export]
// macro_rules! debug_assert_eq {
//     ($a:expr, $b:expr) => {
//         #[cfg(feature = "debug")]
//         {
//             cuda_std::assert_eq!($a, $b);
//         }
//     };
// }
//
#[macro_export]
macro_rules! debug_assert {
    ($a:expr) => {
        #[cfg(feature = "debug")]
        {
            #[cfg(target_os = "cuda")]
            {
                if !$a {
                    let msg = ::alloc::format!("assertion failed: {}", stringify!($a),);

                    unsafe {
                        cuda_std::io::__assertfail(
                            msg.as_ptr(),
                            file!().as_ptr(),
                            line!(),
                            "".as_ptr(),
                            1,
                        )
                    };
                }
            }
            #[cfg(not(target_os = "cuda"))]
            {
                assert!($a);
            }
        }
    };
}
// #[macro_export]
// macro_rules! dbg {
//     // NOTE: We cannot use `concat!` to make a static string as a format argument
//     // of `eprintln!` because `file!` could contain a `{` or
//     // `$val` expression could be a block (`{ .. }`), in which case the `eprintln!`
//     // will be malformed.
//     () => {
//         #[cfg(feature = "debug")]
//         {
//             cuda_std::println!("[{}:{}]", file!(), line!())
//         }
//     };
//     ($val:expr $(,)?) => {
//         #[cfg(feature = "debug")]
//         {
//             // Use of `match` here is intentional because it affects the lifetimes
//             // of temporaries - https://stackoverflow.com/a/48732525/1063961
//             match $val {
//                 tmp => {
//                     cuda_std::println!("[{}:{}] {} = {:#?}",
//                         file!(), line!(), stringify!($val), &tmp);
//                     tmp
//                 }
//             }
//         }
//     };
//     ($($val:expr),+ $(,)?) => {
//         ($($crate::dbg!($val)),+,)
//     };
// }

extern crate alloc;

pub use ext::*;
pub use helpers::*;
pub use mem::*;
pub use num::*;
pub use unsafe_slice::*;
pub use unsafe_array::*;

mod ext;
mod helpers;
mod mem;
mod num;
mod unsafe_slice;
mod unsafe_array;
