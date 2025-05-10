use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicU8;
use std::sync::atomic::Ordering::*;

use atomic::Atomic;
use slice_ring_buffer::SliceRingBuffer;
struct Channel {
    // Read pointer can be cached to reduce coherency traffic.
    // Use TLB page mapping for wrapping when writing memory with memcpy.
    reading_offset: AtomicU64,
    read_offset: AtomicU64,
    written_offset: AtomicU64,
    writing_offset: AtomicU64,
    buf: slice_ring_buffer::Buffer<u8>,
    // buf: Align<[u8; 256]>,
}

#[repr(align(64))]
struct Align<T>(T);

impl Channel {
    pub fn new() {
        let a = SliceRingBuffer::with_capacity(256);
        let b = slice_ring_buffer::sdeq![1; 256];
    }

    pub fn begin_write(&self, size: usize) -> Option<&mut [u8]> {
        let mut writing_offset = self.writing_offset.load(Relaxed);
        loop {
            let new = writing_offset + size as u64;
            if new <= self.read_offset.load(Relaxed) {
                match self.writing_offset.compare_exchange(writing_offset, new, Relaxed, Relaxed) {
                    Ok(_) => break,
                    Err(x) => writing_offset = x,
                }
            } else {
                return None;
            }
        }
        unsafe {
            let ptr = self.buf.ptr().add(writing_offset as usize % 256);
            Some(std::slice::from_raw_parts_mut(ptr, size))
        }
    }

    pub fn finish_write(&self, write_offset: u8) {
        self.written_offset.store(write_offset, Release);
    }

    pub fn read(&self, last_offset: u8) -> ReadGuard {
        let written_offset = self.written_offset.load(Relaxed);
        let len = 

    }
}

pub struct ReadGuard<'a> {
    channel: &'a Channel,
    // read_offset: u8,
}
