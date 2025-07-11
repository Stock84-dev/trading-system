use core::slice;

pub unsafe fn ptr_as_slice<'r, T, R>(ptr: *const T, len: usize) -> &'r [R] {
    let slice: &[R] = slice::from_raw_parts(ptr as *const _ as *const R, len);
    slice
}

pub unsafe fn ptr_as_slice_mut<'r, T, R>(ptr: *mut T, len: usize) -> &'r mut [R] {
    let slice: &mut [R] = slice::from_raw_parts_mut(ptr as *mut _ as *mut R, len);
    slice
}

pub unsafe fn object_as_slice<'r, T, R>(object: &T, len: usize) -> &'r [R] {
    ptr_as_slice(object as *const T, len)
}

pub unsafe fn object_as_slice_mut<'r, T, R>(object: &mut T, len: usize) -> &'r mut [R] {
    ptr_as_slice_mut(object as *mut T, len)
}
