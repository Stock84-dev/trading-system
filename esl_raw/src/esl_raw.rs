#![feature(lazy_cell)]
#![feature(f16)]
#![feature(core_intrinsics)]
#![feature(portable_simd)]
#![feature(slice_swap_unchecked)]
#![allow(invalid_reference_casting)]
// #![feature(downcast_unchecked)]
// #![feature(int_roundings)]
//
// use std::ffi::c_void;
use std::fs::{File, OpenOptions};
use std::hash::BuildHasherDefault;
use std::io::{Read, Seek as _, Write};
use std::ops::{Deref, DerefMut, Range};
// use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use ahash::AHasher;
use binary::BinaryEncoder;
use bitcode::{Buffer, Decode, Encode};
use bytemuck::{Pod, Zeroable};
use chrono::{Datelike, Duration, NaiveDate};
pub use custom_config::*;
use ergnomics::*;
// use cust::context::CurrentContext;
// use cust::memory::{
//     AsyncCopyDestination as _, CopyDestination as _, DeviceBuffer, DeviceCopy,
// DevicePointer,     GpuBuffer, LockedBuffer,
// };
// use cust::stream::Stream;
use ergnomics_gpu::*;
use eyre::Context;
// use error::ToResult as _;
// pub use esl_macros_raw::*;
pub use eyre::{Result, bail, ensure, eyre as anyhow};
use memmap2::{Mmap, MmapOptions};
use parking_lot::Mutex;
use path_no_alloc::with_paths;
// pub use {bytemuck, cust, ergnomics_gpu, eyre};

pub mod binary;
// mod error;
// pub mod parquet;
// pub mod polars_ext;
// pub mod postgres;
pub mod binomial;
pub mod cache;
pub mod cleaning;
pub mod hlc;
pub mod indicators;
pub mod messages;

pub type MarketId = u16;

pub const START_DATE: NaiveDate = NaiveDate::from_ymd_opt(1896, 1, 1).unwrap();
pub const START_YEAR: u16 = 1896;

#[repr(u8)]
#[derive(Debug)]
pub enum ScoreKind {
    LongEntry,
    LongExit,
    LongStop,
    LongAlphaCapture,
    ShortEntry,
    ShortExit,
    ShortStop,
    ShortAlphaCapture,
}

#[repr(u8)]
#[derive(Debug)]
pub enum DiscoveryMethod {
    MAD,
    Mean,
    KDE,
}

#[repr(u8)]
#[derive(Debug)]
pub enum ModelKind {
    YearlySeasonality1,
}

// #[macro_use]
// extern crate paste;

// pub trait FileLoad: Sized {
//     fn load(path: impl AsRef<Path>) -> Result<Self>;
// }
//
// pub trait FileLoadAsync: Sized {
//     fn load(path: impl AsRef<Path>, stream: &Stream) -> Result<Self>;
// }
//
// impl<T: Pod> FileLoad for LockedBuffer<T> {
//     fn load(path: impl AsRef<Path>) -> Result<Self> {
//         let path = path.as_ref();
//         let mut file = OpenOptions::new().read(true).open(path)?;
//         let len = file.metadata()?.len() as usize;
//         let mut data = unsafe { LockedBuffer::uninitialized(len)? };
//         file.read_exact(data.cast_slice_mut())?;
//         Ok(data)
//     }
// }
// impl<T: Pod> FileLoad for DeviceBuffer<T> {
//     fn load(path: impl AsRef<Path>) -> Result<Self> {
//         let path = path.as_ref();
//         let mut file = OpenOptions::new().read(true).open(path)?;
//         unsafe {
//             let map = MmapOptions::new().populate().map(&file)?;
//             let len = file.metadata()?.len() as usize;
//             let mut buf = DeviceBuffer::<u8>::uninitialized(len)?;
//             buf.copy_from(&map);
//             Ok(buf.cast())
//         }
//     }
// }

pub struct Reader<T> {
    map: Mmap,
    _t: std::marker::PhantomData<T>,
}

impl<T: Copy> Reader<T> {
    #[inline(always)]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let file = OpenOptions::new()
            .read(true)
            .open(path)
            .with_context(|| format!("{:?}", path))?;
        let map = unsafe { MmapOptions::new().populate().map(&file)? };
        Ok(Self {
            map,
            _t: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    pub fn slice(&self) -> &[T] {
        self.map.as_ref().cast_slice()
    }

    #[inline(always)]
    pub fn static_slice(&self) -> &'static [T] {
        self.map.as_ref().cast_slice()
    }

    #[inline(always)]
    pub fn range(&self, range: &Range<usize>) -> UnsafeSlice<T> {
        // let start = range.start * std::mem::size_of::<T>();
        // let len = range.end * std::mem::size_of::<T>() - start;
        // let _ = self.map.advise_range(memmap2::Advice::Sequential, start, len);
        unsafe { UnsafeSlice::from_slice(&self.map.as_ref().cast_slice()[range.clone()]) }
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.map.len() / std::mem::size_of::<T>()
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.map.as_ptr() as *const T
    }
}

pub struct Updater<T> {
    data: Box<[u8]>,
    path_part: String,
    pub commit_i: AtomicUsize,
    _t: std::marker::PhantomData<T>,
}

impl<T> Updater<T> {
    #[inline(always)]
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let needle = path.file_stem().unwrap().to_str().unwrap();
        let mut max_n: Option<usize> = None;
        let parent = path.parent().unwrap();
        for entry in std::fs::read_dir(parent)? {
            let entry = entry?;
            let path = entry.path();
            let path = path.file_name().unwrap().to_str().unwrap();
            if path.starts_with(needle) {
                if path.len() == needle.len() + 4 {
                    continue;
                }
                let start = path.rfind('_').unwrap() + 1;
                let n = path[start..].parse::<usize>().unwrap();
                match &mut max_n {
                    Some(max_n) => {
                        max_n.max_mut(n);
                    },
                    None => max_n = Some(n),
                }
            }
        }
        let commit_i;
        let path;
        match max_n {
            Some(max_n) => {
                commit_i = max_n + 1;
                path = format!("{}/{}.bin_{}", parent.to_str().unwrap(), needle, max_n);
            },
            None => {
                commit_i = 0;
                path = format!("{}/{}.bin", parent.to_str().unwrap(), needle);
            },
        }
        let path_part = format!("{}/{}.bin_", parent.to_str().unwrap(), needle);
        let data = std::fs::read(&path)?;
        Ok(Self {
            data: data.into_boxed_slice(),
            path_part,
            commit_i: AtomicUsize::new(commit_i),
            _t: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    pub fn slice(&self) -> &[T] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const T,
                self.data.len() / core::mem::size_of::<T>(),
            )
        }
    }

    #[inline(always)]
    pub fn range(&self, range: &Range<usize>) -> UnsafeSliceMut<T> {
        unsafe {
            UnsafeSliceMut::from_slice(
                &std::slice::from_raw_parts(
                    self.data.as_ptr() as *const T,
                    self.data.len() / core::mem::size_of::<T>(),
                )[range.clone()],
            )
        }
    }

    pub fn commit(&self) -> Result<()> {
        let i = self.commit_i.fetch_add(1, Ordering::Relaxed);
        let path = format!("{}{}", self.path_part, i);
        std::fs::write(&path, &self.data)?;
        println!("committed {i}");
        Ok(())
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.data.len() / std::mem::size_of::<T>()
    }

    #[inline(always)]
    pub fn as_ptr(&self) -> *const T {
        self.data.as_ptr() as *const T
    }
}

impl<T> Deref for Updater<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.slice()
    }
}

impl<T> DerefMut for Updater<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            std::slice::from_raw_parts_mut(
                self.data.as_mut_ptr() as *mut T,
                self.data.len() / core::mem::size_of::<T>(),
            )
        }
    }
}

impl<T> Drop for Updater<T> {
    fn drop(&mut self) {
        self.commit().unwrap();
    }
}

#[macro_export]
macro_rules! work_iter {
    ($ty:ty) => {
        $crate::WorkIterator::<$ty>::new(env!("CARGO_BIN_NAME"))?
    };
}

#[macro_export]
macro_rules! par_work_iter {
    ($ty:ty, $f:expr) => {
        $crate::parallel_work::<$ty>(env!("CARGO_BIN_NAME"), $f)?;
    };
}

#[macro_export]
macro_rules! work_slice_iter {
    ($ty:ty) => {
        $crate::WorkSliceIterator::<$ty>::new(env!("CARGO_BIN_NAME"))?
    };
}

pub struct WorkIterator<T> {
    reader: Reader<T>,
    states_path: String,
    index: usize,
    commit_time: Instant,
    state_file: File,
    start: Instant,
    print_time: Instant,
    start_index: usize,
    commit_len: usize,
}

impl<T: Copy> WorkIterator<T> {
    pub fn new(state_name: &str) -> Result<Self> {
        let reader = Reader::open(format!(
            "{}/{}.bin",
            STATES_PATH,
            core::any::type_name::<T>()
        ))?;
        let states_path = format!("{}/{}.bin", STATES_PATH, state_name);
        let mut state_file =
            OpenOptions::new().read(true).write(true).create(true).open(&states_path)?;
        let mut cursor = 0;
        let len = state_file.metadata()?.len() as usize;
        if len != 0 {
            let mut bytes = unsafe {
                core::slice::from_raw_parts_mut(
                    &mut cursor as *mut _ as *mut u8,
                    std::mem::size_of::<usize>(),
                )
            };
            state_file.read_exact(bytes)?;
        }
        let now = Instant::now();
        Ok(Self {
            reader,
            index: cursor,
            states_path,
            start_index: cursor,
            start: now,
            print_time: now,
            commit_time: now,
            state_file,
            commit_len: 0,
        })
    }

    #[inline(always)]
    pub fn slice(&self) -> &[T] {
        &self.reader.slice()[self.index..]
    }

    #[inline(always)]
    pub fn commit(&mut self, n_items: usize) -> Result<()> {
        self.index += n_items;
        if self.print_time.elapsed().as_secs() >= 1 {
            self.print_time = Instant::now();
            let processed = self.index - self.start_index;
            let elapsed = self.start.elapsed().as_secs_f32();
            let throughput = processed as f32 / elapsed;
            let remaining = self.reader.len() - self.index;
            let eta = remaining as f32 / throughput;
            let eta_duration = std::time::Duration::from_secs(eta as u64);
            let eta_pretty = humantime::format_duration(eta_duration);
            println!(
                "{:.3}%, {:.3} wi/s, ETA: {} {:?}",
                (self.index + 1) as f32 / self.reader.len() as f32 * 100.0,
                throughput,
                eta_pretty,
                eta_duration,
            );
        }
        if self.commit_time.elapsed().as_secs() >= 60 {
            let bytes = unsafe {
                core::slice::from_raw_parts(
                    &mut self.index as *const _ as *const u8,
                    std::mem::size_of::<usize>(),
                )
            };
            self.state_file.seek(std::io::SeekFrom::Start(0))?;
            self.state_file.write_all(bytes)?;
            self.commit_time = Instant::now();
        }
        Ok(())
    }

    pub fn delete_state(&mut self) -> Result<()> {
        std::fs::remove_file(&self.states_path)?;
        Ok(())
    }
}

impl<T: Copy> Iterator for WorkIterator<T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<T> {
        self.commit(self.commit_len as usize).unwrap();
        if self.index >= self.reader.len() {
            self.delete_state().unwrap();
            return None;
        }
        let work_item = unsafe { *self.reader.slice().get_unchecked(self.index) };
        self.commit_len = 1;
        Some(work_item)
    }
}

pub struct WorkSliceIterator<'a, T> {
    inner: WorkIterator<T>,
    _a: std::marker::PhantomData<&'a ()>,
}

impl<'a, T: Copy> WorkSliceIterator<'a, T> {
    pub fn new(state_name: &str) -> Result<Self> {
        Ok(Self {
            inner: WorkIterator::new(state_name)?,
            _a: std::marker::PhantomData,
        })
    }

    #[inline(always)]
    pub fn commit(&mut self, n_items: usize) -> Result<()> {
        self.inner.commit(n_items)?;
        Ok(())
    }
}

impl<'a, T: Copy + 'static> Iterator for WorkSliceIterator<'a, T> {
    type Item = UnsafeSlice<'a, T>;

    #[inline(always)]
    fn next(&mut self) -> Option<UnsafeSlice<'a, T>> {
        let slice = unsafe { UnsafeSlice::from_slice(self.inner.slice()) };
        if slice.is_empty() {
            self.inner.delete_state().unwrap();
            return None;
        }
        Some(slice)
    }
}

pub fn parallel_work<T: Copy + Sync>(
    state_name: &str,
    f: impl Fn(&T) -> Result<()> + Sync,
) -> Result<()> {
    let reader = Reader::open(format!(
        "{}/{}.bin",
        STATES_PATH,
        core::any::type_name::<T>()
    ))?;
    let states_path = format!("{}/{}.bin", STATES_PATH, state_name);
    let mut state_file =
        OpenOptions::new().read(true).write(true).create(true).open(&states_path)?;
    let mut cursor = 0;
    let len = state_file.metadata()?.len() as usize;
    if len != 0 {
        let bytes = unsafe {
            core::slice::from_raw_parts_mut(
                &mut cursor as *mut _ as *mut u8,
                std::mem::size_of::<usize>(),
            )
        };
        state_file.read_exact(bytes)?;
    }
    let start_index = cursor;
    let now = Instant::now();
    let start_time = now;
    let slice = unsafe { reader.slice().as_unsafe_slice() };
    let n_cpus = num_cpus::get();
    let leading_offset = AtomicUsize::new(cursor);
    let commited_offset = AtomicUsize::new(cursor);
    struct State {
        commit_time: Instant,
        state_file: File,
        print_time: Instant,
    }
    let commit_state = Mutex::new(State {
        commit_time: now,
        state_file,
        print_time: now,
    });
    let uncommited_offsets = Mutex::new(Vec::new());
    println!("Spawning {} threads", n_cpus);
    std::thread::scope(|s| {
        let handles = (0..n_cpus)
            .map(|thread_id| {
                let f = &f;
                let reader = &reader;
                let slice = &slice;
                let leading_offset = &leading_offset;
                let commited_offset = &commited_offset;
                let uncommited_offsets = &uncommited_offsets;
                let commit_state = &commit_state;
                s.spawn(move || {
                    loop {
                        let offset = leading_offset.fetch_add(1, Ordering::Relaxed);
                        if offset >= reader.len() {
                            println!("Thread {} finished", thread_id);
                            return;
                        }
                        let wi = &slice[offset];
                        f(wi).unwrap();
                        let mut uncommited_offsets = uncommited_offsets.lock();
                        let co = commited_offset.load(Ordering::Acquire);
                        if offset == co + 1 && !uncommited_offsets.is_empty() {
                            let mut find_co = co + 2;
                            while uncommited_offsets.iter().any(|&x| x == find_co) {
                                find_co += 1;
                            }
                            commited_offset.store(find_co - 1, Ordering::Release);
                            uncommited_offsets.retain(|&x| x >= find_co);
                        } else if offset == co + 1 {
                            commited_offset.store(offset, Ordering::Release);
                        } else {
                            uncommited_offsets.push(offset);
                        }
                        // dbg!(thread_id, offset, commited_offset.load(Ordering::Acquire));
                        drop(uncommited_offsets);
                        let mut commit_state = commit_state.lock();
                        if commit_state.commit_time.elapsed().as_secs() >= 60 {
                            let commited_offset = commited_offset.load(Ordering::Acquire);
                            let bytes = unsafe {
                                core::slice::from_raw_parts(
                                    &commited_offset as *const _ as *const u8,
                                    std::mem::size_of::<usize>(),
                                )
                            };
                            commit_state.state_file.seek(std::io::SeekFrom::Start(0)).unwrap();
                            commit_state.state_file.write_all(bytes).unwrap();
                            commit_state.commit_time = Instant::now();
                        }
                        if commit_state.print_time.elapsed().as_secs() >= 1 {
                            let commited_offset = commited_offset.load(Ordering::Acquire);
                            commit_state.print_time = Instant::now();
                            let processed = commited_offset - start_index;
                            let elapsed = start_time.elapsed().as_secs_f32();
                            let throughput = processed as f32 / elapsed;
                            let remaining = reader.len() - commited_offset;
                            let eta = remaining as f32 / throughput;
                            let eta_duration = std::time::Duration::from_secs(eta as u64);
                            let eta_pretty = humantime::format_duration(eta_duration);
                            println!(
                                "{:.3}%, {:.3} wi/s, ETA: {} {:?}",
                                (commited_offset + 1) as f32 / reader.len() as f32 * 100.0,
                                throughput,
                                eta_pretty,
                                eta_duration,
                            );
                        }
                    }
                })
            })
            .collect::<Vec<_>>();
        for handle in handles {
            handle.join().unwrap();
        }
    });

    Ok(())
}
// pub struct ParallelWorkIterator<'a, T> {
//     index: AtomicUsize,
//     inner: WorkIterator<T>,
//     _a: std::marker::PhantomData<&'a ()>,
// }
//
// impl<'a, T: Copy> ParallelWorkIterator<'a, T> {
//     pub fn new(state_name: &str) -> Result<Self> {
//         Ok(Self {
//             inner: WorkIterator::new(state_name)?,
//             _a: std::marker::PhantomData,
//         })
//     }
//
//     pub fn run(&mut self, f: impl Fn(&T) -> Result<()>) -> Result<()> {
//         let n_cpus = num_cpus::get();
//         let mut index =
//         let mut uncommited_offsets = vec![0; n_cpus];
//         let handles = (0..num_cpus::get())
//             .map(|_| {
//                 let mut inner = self.inner.clone();
//                 std::thread::spawn(move || {
//                     while let Some(work_item) = inner.next() {
//                         f(&work_item).unwrap();
//                     }
//                 })
//             })
//             .collect::<Vec<_>>();
//     }
// }

// impl<T: Pod + DeviceCopy> Reader<T> {
//     pub fn device_buffer_async(&self, stream: &Stream) -> Result<DeviceBuffer<T>> {
//         unsafe {
//             let len = self.map.len();
//             let buf = DeviceBuffer::<u8>::uninitialized(len)?;
//             cust_raw::cuMemcpyHtoDAsync_v2(
//                 buf.as_device_ptr().as_raw(),
//                 self.map.as_ptr() as *const c_void,
//                 len,
//                 stream.as_inner(),
//             )
//             .to_result()?;
//             Ok(buf.cast())
//         }
//     }
//
//     pub fn device_buffer(&self) -> Result<DeviceBuffer<T>> {
//         unsafe {
//             let len = self.map.len();
//             let buf = DeviceBuffer::<u8>::uninitialized(len)?;
//             cust_raw::cuMemcpyHtoD_v2(
//                 buf.as_device_ptr().as_raw(),
//                 self.map.as_ptr() as *const c_void,
//                 len,
//             )
//             .to_result()?;
//             Ok(buf.cast())
//         }
//     }
// }

// impl<T: Pod> FileLoadAsync for DeviceBuffer<T> {
//     fn load(path: impl AsRef<Path>, stream: &Stream) -> Result<Self> {
//         let path = path.as_ref();
//         let mut file = OpenOptions::new().read(true).open(path)?;
//         unsafe {
//             let map = MmapOptions::new().populate().map(&file)?;
//             let len = file.metadata()?.len() as usize;
//             let mut buf = DeviceBuffer::<u8>::uninitialized(len)?;
//             buf.copy_from(&map);
//             Ok(buf.cast())
//         }
//     }
// }

// #[derive(Clone)]
// #[repr(C)]
// pub struct DeviceSlice<T: DeviceCopy> {
//     ptr: DevicePointer<T>,
//     len: usize,
// }
//
// unsafe impl<T: Pod + DeviceCopy> DeviceCopy for DeviceSlice<T> {}
// impl<T: Pod + DeviceCopy> Copy for DeviceSlice<T> {}
// unsafe impl<T: Zeroable + DeviceCopy> Zeroable for DeviceSlice<T> {}
// unsafe impl<T: Pod + DeviceCopy> Pod for DeviceSlice<T> {}
//
// pub trait AsDeviceSlice<T: DeviceCopy> {
//     fn as_device_slice(&self) -> DeviceSlice<T>;
// }
//
// impl<T: Pod + DeviceCopy> AsDeviceSlice<T> for DeviceBuffer<T> {
//     fn as_device_slice(&self) -> DeviceSlice<T> {
//         DeviceSlice {
//             ptr: self.as_device_ptr(),
//             len: self.len(),
//         }
//     }
// }

pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<AHasher>>;
pub type HashSet<T> = std::collections::HashSet<T, BuildHasherDefault<AHasher>>;

pub const N_PATHS: usize = 1771;

#[derive(Debug, Hash, PartialEq, Eq, Encode, Decode)]
pub struct CcvMetadata {
    pub market: String,
    pub wfa_end_ts_d: u16,
    pub duration_y: u16,
}

pub type CcvMetadataMap = HashMap<CcvMetadata, Vec<u32>>;

#[repr(u8)]
#[derive(Copy, Clone, PartialEq, Eq, Debug, Default)]
pub enum CcvSplit {
    #[default]
    Train,
    Validation,
}

impl CcvSplit {
    pub fn as_str(&self) -> &'static str {
        match self {
            CcvSplit::Train => "train",
            CcvSplit::Validation => "validation",
        }
    }
}
const MAX_CCV_SEGMENTS: usize = 6 * 2 + 1;
#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct CcvWorkItem {
    pub duration_y: u8,
    pub n_segments: u8,
    pub wfa_end_ts_d: u16,
    pub market_id: MarketId,
    // work item contains only n_validation * 2 + 1 paths, not thousands
    /// Combined chunks until they reach c chunk with different type
    pub segment_ranges: [SegmentRange; MAX_CCV_SEGMENTS],
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct SegmentRange {
    // PERF: start always equals to previous end
    pub start: u16,
    pub end: u16,
    pub split: CcvSplit,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
/// Iterates by split, wfa_end_ts_d, duration_y, ccv_path, market_id
pub struct PathWorkItem {
    pub split: CcvSplit,
    pub duration_y: u8,
    pub wfa_end_ts_d: u16,
    pub path_id: u16,
    pub market_ranges: [(u32, u32); N_MARKETS],
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
/// Iterates by split, wfa_end_ts_d, duration_y, market_id, ccv_path
pub struct MarketWorkItem {
    pub split: CcvSplit,
    pub duration_y: u8,
    pub wfa_end_ts_d: u16,
    pub market_id: u16,
    pub n_paths: u16,
    pub path_ranges: [(u32, u32); N_PATHS],
}

pub fn create_market_work_items() -> Result<()> {
    let mut buffer = Buffer::new();
    let mut encoder = BinaryEncoder::<MarketWorkItem>::new(format!(
        "/home/stock/nvme/states/{}.bin",
        core::any::type_name::<MarketWorkItem>()
    ))?;
    let market_to_id = market_to_market_id();
    for split in ["train", "validation"] {
        let split_enum = match split {
            "train" => CcvSplit::Train,
            "validation" => CcvSplit::Validation,
            _ => unreachable!(),
        };
        with_paths! {
            path = CCV_DATA_LAKE_PATH / split,
            offsets = path / "offsets.bin",
        };
        if !path.is_dir() {
            continue;
        }
        let map: CcvMetadataMap = buffer.decode(&std::fs::read(offsets)?)?;
        for (k, v) in &map {
            let mut prev_offset = 0;
            let market_id = *market_to_id.get(k.market.as_str()).unwrap();
            let mut work_item = MarketWorkItem {
                split: split_enum,
                duration_y: k.duration_y as u8,
                wfa_end_ts_d: k.wfa_end_ts_d,
                market_id,
                n_paths: 0,
                path_ranges: [(0, 0); N_PATHS],
            };
            for offset in v.iter() {
                work_item.path_ranges[work_item.n_paths as usize] = (prev_offset, *offset);
                work_item.n_paths += 1;
                prev_offset = *offset as u32;
            }
            encoder.push(work_item)?;
        }
    }
    Ok(())
}

pub fn create_path_work_items() -> Result<()> {
    let mut buffer = Buffer::new();
    let mut encoder = BinaryEncoder::<PathWorkItem>::new(format!(
        "{}/{}.bin",
        STATES_PATH,
        core::any::type_name::<PathWorkItem>()
    ))?;
    let market_to_id = market_to_market_id();
    for split in ["train", "validation"] {
        let split_enum = match split {
            "train" => CcvSplit::Train,
            "validation" => CcvSplit::Validation,
            _ => unreachable!(),
        };
        with_paths! {
            path = CCV_DATA_LAKE_PATH / split,
            offsets = path / "offsets.bin",
        };
        if !path.is_dir() {
            continue;
        }
        let map: CcvMetadataMap = buffer.decode(&std::fs::read(offsets)?)?;
        let mut map2: HashMap<u16, HashMap<u16, HashMap<u16, Vec<(MarketId, (u32, u32))>>>> =
            HashMap::default();
        for (k, v) in &map {
            let mut prev_offset = 0;
            for (ccv_path, offset) in v.iter().enumerate() {
                let entry = map2.entry(k.wfa_end_ts_d).or_default();
                let entry = entry.entry(k.duration_y).or_default();
                let entry = entry.entry(ccv_path as u16).or_default();
                let market_id = *market_to_id.get(k.market.as_str()).unwrap();
                entry.push((market_id, (prev_offset, *offset)));
                prev_offset = *offset as u32;
            }
        }

        for (wfa_end_ts_d, map) in map2 {
            for (duration_y, map) in map {
                for (path_id, offsets) in map {
                    let mut work_item = PathWorkItem {
                        split: split_enum,
                        duration_y: duration_y as u8,
                        wfa_end_ts_d,
                        path_id,
                        market_ranges: [(0, 0); N_MARKETS],
                    };
                    for (market_id, range) in offsets {
                        work_item.market_ranges[market_id as usize] = range;
                    }
                    encoder.push(work_item)?;
                }
            }
        }
    }
    Ok(())
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Dividend {
    pub ex_dividend_date_ts_d: u16,
    pub declaration_date_ts_d: u16,
    pub record_date_ts_d: u16,
    pub payment_date_ts_d: u16,
    pub amount: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct Split {
    pub effective_date_ts_d: u16,
    pub ratio: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Earning {
    pub fiscal_date_ending_ts_d: u16,
    pub reported_date_ts_d: u16,
    pub reported_eps: f32,
    pub estimated_eps: f32,
    pub report_time: ReportTime,
}

#[repr(u8)]
#[derive(Copy, Clone, Debug)]
pub enum ReportTime {
    PreMarket,
    PostMarket,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Gdp {
    pub ts_d: u16,
    /// In billions
    pub value: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct GdpPerCapita {
    pub ts_d: u16,
    /// In chained 2012 dollars
    pub value: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct RetailSales {
    pub ts_d: u16,
    /// In millions
    pub value: f32,
}

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct NonFarmPayroll {
    pub ts_d: u16,
    /// In thousands of persons
    pub value: f32,
}

pub fn build_futures_markets_map() -> HashSet<&'static str> {
    let mut map = HashSet::default();
    map.insert("FX_GBPUSD");
    map.insert("FX_IDC_CADUSD");
    map.insert("FX_EURUSD");
    map.insert("FX_IDC_JPYUSD");
    map.insert("FX_NZDUSD");
    map.insert("FX_IDC_CHFUSD");
    map.insert("TVC_SPX");
    map.insert("TVC_GOLD");
    map.insert("FX_AUDUSD");
    map.insert("BNC_BLX");
    map.insert("TVC_NDX");
    map.insert("CAPITALCOM_RTY");
    map.insert("TVC_SILVER");
    map.insert("TVC_DJI");
    map.insert("XETR_DAX");
    map.insert("TVC_SX5E");
    map.insert("CBOT_KE1!");
    // map.insert("CBOT_UB1!");
    map.insert("CBOT_ZC1!");
    // map.insert("CBOT_ZF1!");
    map.insert("CBOT_ZL1!");
    map.insert("CBOT_ZM1!");
    map.insert("CBOT_ZN1!");
    map.insert("CBOT_ZS1!");
    // map.insert("CBOT_ZT1!");
    // map.insert("CBOT_ZW1!");
    map.insert("CME_HE1!");
    map.insert("CME_LE1!");
    map.insert("COMEX_HG1!");
    map.insert("EUREX_FGBL1!");
    // map.insert("EUREX_FGBM1!");
    // map.insert("EUREX_FGBS1!");
    // map.insert("ICEEUR_I1!");
    // map.insert("ICEUS_CC1!");
    // map.insert("ICEUS_KC1!");
    // map.insert("ICEUS_SB1!");
    map.insert("NYMEX_HO1!");
    map.insert("NYMEX_NG1!");
    map.insert("NYMEX_PL1!");
    map.insert("NYMEX_RB1!");
    map
}

pub fn build_stock_markets_map() -> HashSet<&'static str> {
    let mut map = HashSet::default();
    map.insert("AMEX_DIA");
    map.insert("AMEX_EEM");
    map.insert("AMEX_EFA");
    map.insert("AMEX_EWJ");
    map.insert("AMEX_EWT");
    map.insert("AMEX_EWY");
    map.insert("AMEX_EWZ");
    map.insert("AMEX_FDN");
    map.insert("AMEX_FVD");
    map.insert("AMEX_GDX");
    map.insert("AMEX_GDXJ");
    map.insert("AMEX_GLD");
    map.insert("AMEX_IHI");
    map.insert("AMEX_IJH");
    map.insert("AMEX_IJR");
    map.insert("AMEX_IJS");
    map.insert("AMEX_ITOT");
    map.insert("AMEX_IVE");
    map.insert("AMEX_IVW");
    map.insert("AMEX_IWB");
    map.insert("AMEX_IWD");
    map.insert("AMEX_IWF");
    map.insert("AMEX_IWM");
    map.insert("AMEX_IWN");
    map.insert("AMEX_IWO");
    map.insert("AMEX_IWS");
    map.insert("AMEX_IYR");
    map.insert("AMEX_IYW");
    map.insert("AMEX_KRE");
    map.insert("AMEX_MDY");
    map.insert("AMEX_OEF");
    map.insert("AMEX_RSP");
    map.insert("AMEX_SCHB");
    map.insert("AMEX_SCHF");
    map.insert("AMEX_SCHG");
    map.insert("AMEX_SCHV");
    map.insert("AMEX_SLV");
    map.insert("AMEX_SPDW");
    map.insert("AMEX_SPY");
    map.insert("AMEX_SPYG");
    map.insert("AMEX_SPYV");
    map.insert("AMEX_TIP");
    map.insert("AMEX_VDE");
    map.insert("AMEX_VEA");
    map.insert("AMEX_VGT");
    map.insert("AMEX_VHT");
    map.insert("AMEX_VNQ");
    map.insert("AMEX_VOE");
    map.insert("AMEX_VPL");
    map.insert("AMEX_VT");
    map.insert("AMEX_VTI");
    map.insert("AMEX_VWO");
    map.insert("AMEX_VXF");
    map.insert("AMEX_XBI");
    map.insert("AMEX_XLB");
    map.insert("AMEX_XLE");
    map.insert("AMEX_XLF");
    map.insert("AMEX_XLI");
    map.insert("AMEX_XLK");
    map.insert("AMEX_XLP");
    map.insert("AMEX_XLU");
    map.insert("AMEX_XLV");
    map.insert("AMEX_XLY");
    map.insert("CAPITALCOM_RTY");
    map.insert("CBOE_EFG");
    map.insert("CBOE_EFV");
    map.insert("CBOE_EZU");
    map.insert("CBOE_IGV");
    map.insert("FX_AUDCAD");
    map.insert("FX_AUDCHF");
    map.insert("FX_AUDJPY");
    map.insert("FX_AUDNZD");
    map.insert("FX_AUDUSD");
    map.insert("FX_AUS200");
    map.insert("FX_CADCHF");
    map.insert("FX_CADJPY");
    map.insert("FX_CHFJPY");
    map.insert("FX_EURAUD");
    map.insert("FX_EURCAD");
    map.insert("FX_EURCHF");
    map.insert("FX_EURGBP");
    map.insert("FX_EURJPY");
    map.insert("FX_EURNZD");
    map.insert("FX_EURUSD");
    map.insert("FX_GBPAUD");
    map.insert("FX_GBPCAD");
    map.insert("FX_GBPCHF");
    map.insert("FX_GBPJPY");
    map.insert("FX_GBPNZD");
    map.insert("FX_GBPUSD");
    map.insert("FX_IDC_CADUSD");
    map.insert("FX_IDC_CHFUSD");
    map.insert("FX_IDC_EURMXN");
    map.insert("FX_IDC_EURNOK");
    map.insert("FX_IDC_EURSEK");
    map.insert("FX_IDC_GBPMXN");
    map.insert("FX_IDC_GBPNOK");
    map.insert("FX_IDC_GBPSEK");
    map.insert("FX_IDC_JPYUSD");
    map.insert("FX_IDC_USDNOK");
    map.insert("FX_IDC_USDSGD");
    map.insert("FX_NZDCAD");
    map.insert("FX_NZDCHF");
    map.insert("FX_NZDJPY");
    map.insert("FX_NZDUSD");
    map.insert("FX_UK100");
    map.insert("FX_USDCAD");
    map.insert("FX_USDCHF");
    map.insert("FX_USDJPY");
    map.insert("FX_USDMXN");
    map.insert("FX_USDSEK");
    map.insert("NASDAQ_AAL");
    map.insert("NASDAQ_AAPL");
    map.insert("NASDAQ_AAXJ");
    map.insert("NASDAQ_ADBE");
    map.insert("NASDAQ_ADI");
    map.insert("NASDAQ_ADP");
    map.insert("NASDAQ_ADSK");
    map.insert("NASDAQ_AEP");
    map.insert("NASDAQ_AKAM");
    map.insert("NASDAQ_ALGN");
    map.insert("NASDAQ_ALNY");
    map.insert("NASDAQ_AMAT");
    map.insert("NASDAQ_AMD");
    map.insert("NASDAQ_AMED");
    map.insert("NASDAQ_AMGN");
    map.insert("NASDAQ_AMKR");
    map.insert("NASDAQ_AMZN");
    map.insert("NASDAQ_ANSS");
    map.insert("NASDAQ_APA");
    map.insert("NASDAQ_ARCC");
    map.insert("NASDAQ_ARWR");
    map.insert("NASDAQ_AVGO");
    map.insert("NASDAQ_AXON");
    map.insert("NASDAQ_AZPN");
    map.insert("NASDAQ_AZTA");
    map.insert("NASDAQ_BIIB");
    map.insert("NASDAQ_BKNG");
    map.insert("NASDAQ_BKR");
    map.insert("NASDAQ_BMRN");
    map.insert("NASDAQ_BRKR");
    map.insert("NASDAQ_CACC");
    map.insert("NASDAQ_CAR");
    map.insert("NASDAQ_CASY");
    map.insert("NASDAQ_CDNS");
    map.insert("NASDAQ_CGNX");
    map.insert("NASDAQ_CHDN");
    map.insert("NASDAQ_CHRW");
    map.insert("NASDAQ_CHTR");
    map.insert("NASDAQ_CINF");
    map.insert("NASDAQ_CMCSA");
    map.insert("NASDAQ_CME");
    map.insert("NASDAQ_COLM");
    map.insert("NASDAQ_COO");
    map.insert("NASDAQ_COST");
    map.insert("NASDAQ_CPRT");
    map.insert("NASDAQ_CROX");
    map.insert("NASDAQ_CSCO");
    map.insert("NASDAQ_CSGP");
    map.insert("NASDAQ_CSX");
    map.insert("NASDAQ_CTAS");
    map.insert("NASDAQ_CTSH");
    map.insert("NASDAQ_DLTR");
    map.insert("NASDAQ_DOX");
    map.insert("NASDAQ_DVY");
    map.insert("NASDAQ_DXCM");
    map.insert("NASDAQ_EA");
    map.insert("NASDAQ_EBAY");
    map.insert("NASDAQ_EEFT");
    map.insert("NASDAQ_EMB");
    map.insert("NASDAQ_ENTG");
    map.insert("NASDAQ_EVRG");
    map.insert("NASDAQ_EWBC");
    map.insert("NASDAQ_EXC");
    map.insert("NASDAQ_EXEL");
    map.insert("NASDAQ_EXPE");
    map.insert("NASDAQ_FAST");
    map.insert("NASDAQ_FFIV");
    map.insert("NASDAQ_FITB");
    map.insert("NASDAQ_FLEX");
    map.insert("NASDAQ_FSLR");
    map.insert("NASDAQ_FTNT");
    map.insert("NASDAQ_GEN");
    map.insert("NASDAQ_GILD");
    map.insert("NASDAQ_GNTX");
    map.insert("NASDAQ_GOOGL");
    map.insert("NASDAQ_GT");
    map.insert("NASDAQ_HALO");
    map.insert("NASDAQ_HAS");
    map.insert("NASDAQ_HBAN");
    map.insert("NASDAQ_HELE");
    map.insert("NASDAQ_HOLX");
    map.insert("NASDAQ_HON");
    map.insert("NASDAQ_HSIC");
    map.insert("NASDAQ_IBB");
    map.insert("NASDAQ_IBKR");
    map.insert("NASDAQ_ICLN");
    map.insert("NASDAQ_IDXX");
    map.insert("NASDAQ_ILMN");
    map.insert("NASDAQ_INCY");
    map.insert("NASDAQ_INTC");
    map.insert("NASDAQ_INTU");
    map.insert("NASDAQ_IONS");
    map.insert("NASDAQ_IPGP");
    map.insert("NASDAQ_IRDM");
    map.insert("NASDAQ_ISRG");
    map.insert("NASDAQ_IUSG");
    map.insert("NASDAQ_IUSV");
    map.insert("NASDAQ_JBHT");
    map.insert("NASDAQ_JBLU");
    map.insert("NASDAQ_JKHY");
    map.insert("NASDAQ_KDP");
    map.insert("NASDAQ_KLAC");
    map.insert("NASDAQ_LKQ");
    map.insert("NASDAQ_LNT");
    map.insert("NASDAQ_LNW");
    map.insert("NASDAQ_LRCX");
    map.insert("NASDAQ_LSCC");
    map.insert("NASDAQ_LSTR");
    map.insert("NASDAQ_MANH");
    map.insert("NASDAQ_MAR");
    map.insert("NASDAQ_MASI");
    map.insert("NASDAQ_MAT");
    map.insert("NASDAQ_MCHP");
    map.insert("NASDAQ_MDLZ");
    map.insert("NASDAQ_MIDD");
    map.insert("NASDAQ_MKSI");
    map.insert("NASDAQ_MKTX");
    map.insert("NASDAQ_MNST");
    map.insert("NASDAQ_MPWR");
    map.insert("NASDAQ_MRVL");
    map.insert("NASDAQ_MSFT");
    map.insert("NASDAQ_MSTR");
    map.insert("NASDAQ_MU");
    map.insert("NASDAQ_NBIX");
    map.insert("NASDAQ_NDAQ");
    map.insert("NASDAQ_NDSN");
    map.insert("NASDAQ_NDX");
    map.insert("NASDAQ_NFLX");
    map.insert("NASDAQ_NTAP");
    map.insert("NASDAQ_NTRS");
    map.insert("NASDAQ_NVDA");
    map.insert("NASDAQ_NWL");
    map.insert("NASDAQ_NXST");
    map.insert("NASDAQ_ODFL");
    map.insert("NASDAQ_OLED");
    map.insert("NASDAQ_OMCL");
    map.insert("NASDAQ_ON");
    map.insert("NASDAQ_ORLY");
    map.insert("NASDAQ_OZK");
    map.insert("NASDAQ_PARA");
    map.insert("NASDAQ_PAYX");
    map.insert("NASDAQ_PCAR");
    map.insert("NASDAQ_PEGA");
    map.insert("NASDAQ_PENN");
    map.insert("NASDAQ_PEP");
    map.insert("NASDAQ_PFF");
    map.insert("NASDAQ_PFG");
    map.insert("NASDAQ_PNFP");
    map.insert("NASDAQ_PODD");
    map.insert("NASDAQ_POOL");
    map.insert("NASDAQ_PTC");
    map.insert("NASDAQ_QCOM");
    map.insert("NASDAQ_QQQ");
    map.insert("NASDAQ_REGN");
    map.insert("NASDAQ_RGEN");
    map.insert("NASDAQ_RGLD");
    map.insert("NASDAQ_ROP");
    map.insert("NASDAQ_ROST");
    map.insert("NASDAQ_SAIA");
    map.insert("NASDAQ_SBUX");
    map.insert("NASDAQ_SCZ");
    map.insert("NASDAQ_SLAB");
    map.insert("NASDAQ_SLM");
    map.insert("NASDAQ_SMH");
    map.insert("NASDAQ_SNPS");
    map.insert("NASDAQ_SOXX");
    map.insert("NASDAQ_SRPT");
    map.insert("NASDAQ_SSNC");
    map.insert("NASDAQ_STLD");
    map.insert("NASDAQ_SWKS");
    map.insert("NASDAQ_SYNA");
    map.insert("NASDAQ_TECH");
    map.insert("NASDAQ_TER");
    map.insert("NASDAQ_TLT");
    map.insert("NASDAQ_TMUS");
    map.insert("NASDAQ_TRMB");
    map.insert("NASDAQ_TROW");
    map.insert("NASDAQ_TSCO");
    map.insert("NASDAQ_TTEK");
    map.insert("NASDAQ_TTWO");
    map.insert("NASDAQ_TXN");
    map.insert("NASDAQ_TXRH");
    map.insert("NASDAQ_UAL");
    map.insert("NASDAQ_ULTA");
    map.insert("NASDAQ_UTHR");
    map.insert("NASDAQ_VRSK");
    map.insert("NASDAQ_VRSN");
    map.insert("NASDAQ_VRTX");
    map.insert("NASDAQ_VTRS");
    map.insert("NASDAQ_WBA");
    map.insert("NASDAQ_WDC");
    map.insert("NASDAQ_WWD");
    map.insert("NASDAQ_WYNN");
    map.insert("NASDAQ_XEL");
    map.insert("NASDAQ_XRAY");
    map.insert("NASDAQ_ZBRA");
    map.insert("NASDAQ_ZD");
    map.insert("NASDAQ_ZION");
    map.insert("NYSE_A");
    map.insert("NYSE_AA");
    map.insert("NYSE_AAP");
    map.insert("NYSE_ABT");
    map.insert("NYSE_ACM");
    map.insert("NYSE_ACN");
    map.insert("NYSE_ADM");
    map.insert("NYSE_AES");
    map.insert("NYSE_AFG");
    map.insert("NYSE_AFL");
    map.insert("NYSE_AGCO");
    map.insert("NYSE_AIG");
    map.insert("NYSE_AIZ");
    map.insert("NYSE_AJG");
    map.insert("NYSE_ALB");
    map.insert("NYSE_ALK");
    map.insert("NYSE_ALL");
    map.insert("NYSE_AME");
    map.insert("NYSE_AMG");
    map.insert("NYSE_AMP");
    map.insert("NYSE_AMT");
    map.insert("NYSE_AN");
    map.insert("NYSE_AON");
    map.insert("NYSE_AOS");
    map.insert("NYSE_APD");
    map.insert("NYSE_APH");
    map.insert("NYSE_ARW");
    map.insert("NYSE_ASH");
    map.insert("NYSE_AVY");
    map.insert("NYSE_AWI");
    map.insert("NYSE_AWK");
    map.insert("NYSE_AXP");
    map.insert("NYSE_AYI");
    map.insert("NYSE_AZO");
    map.insert("NYSE_BA");
    map.insert("NYSE_BAC");
    map.insert("NYSE_BALL");
    map.insert("NYSE_BAX");
    map.insert("NYSE_BBY");
    map.insert("NYSE_BC");
    map.insert("NYSE_BDX");
    map.insert("NYSE_BEN");
    map.insert("NYSE_BIO");
    map.insert("NYSE_BK");
    map.insert("NYSE_BLDR");
    map.insert("NYSE_BLK");
    map.insert("NYSE_BMY");
    map.insert("NYSE_BR");
    map.insert("NYSE_BRK.A");
    map.insert("NYSE_BRO");
    map.insert("NYSE_BSX");
    map.insert("NYSE_BWA");
    map.insert("NYSE_BX");
    map.insert("NYSE_BYD");
    map.insert("NYSE_C");
    map.insert("NYSE_CACI");
    map.insert("NYSE_CAG");
    map.insert("NYSE_CAH");
    map.insert("NYSE_CAT");
    map.insert("NYSE_CB");
    map.insert("NYSE_CBRE");
    map.insert("NYSE_CCI");
    map.insert("NYSE_CCK");
    map.insert("NYSE_CCL");
    map.insert("NYSE_CE");
    map.insert("NYSE_CF");
    map.insert("NYSE_CFR");
    map.insert("NYSE_CHD");
    map.insert("NYSE_CHE");
    map.insert("NYSE_CI");
    map.insert("NYSE_CIEN");
    map.insert("NYSE_CL");
    map.insert("NYSE_CLF");
    map.insert("NYSE_CLX");
    map.insert("NYSE_CMA");
    map.insert("NYSE_CMG");
    map.insert("NYSE_CMI");
    map.insert("NYSE_CMS");
    map.insert("NYSE_CNC");
    map.insert("NYSE_CNP");
    map.insert("NYSE_COF");
    map.insert("NYSE_COHR");
    map.insert("NYSE_COP");
    map.insert("NYSE_COR");
    map.insert("NYSE_CRL");
    map.insert("NYSE_CRM");
    map.insert("NYSE_CSL");
    map.insert("NYSE_CVS");
    map.insert("NYSE_CVX");
    map.insert("NYSE_D");
    map.insert("NYSE_DAL");
    map.insert("NYSE_DAR");
    map.insert("NYSE_DD");
    map.insert("NYSE_DE");
    map.insert("NYSE_DECK");
    map.insert("NYSE_DFS");
    map.insert("NYSE_DG");
    map.insert("NYSE_DGX");
    map.insert("NYSE_DHI");
    map.insert("NYSE_DHR");
    map.insert("NYSE_DIS");
    map.insert("NYSE_DKS");
    map.insert("NYSE_DLB");
    map.insert("NYSE_DOV");
    map.insert("NYSE_DPZ");
    map.insert("NYSE_DRI");
    map.insert("NYSE_DTE");
    map.insert("NYSE_DUK");
    map.insert("NYSE_DVA");
    map.insert("NYSE_DVN");
    map.insert("NYSE_ECL");
    map.insert("NYSE_EFX");
    map.insert("NYSE_EHC");
    map.insert("NYSE_EIX");
    map.insert("NYSE_EL");
    map.insert("NYSE_ELV");
    map.insert("NYSE_EME");
    map.insert("NYSE_EMN");
    map.insert("NYSE_EMR");
    map.insert("NYSE_EOG");
    map.insert("NYSE_EQT");
    map.insert("NYSE_ES");
    map.insert("NYSE_ETN");
    map.insert("NYSE_ETR");
    map.insert("NYSE_EVR");
    map.insert("NYSE_EW");
    map.insert("NYSE_EXP");
    map.insert("NYSE_EXPD");
    map.insert("NYSE_F");
    map.insert("NYSE_FAF");
    map.insert("NYSE_FCX");
    map.insert("NYSE_FDS");
    map.insert("NYSE_FDX");
    map.insert("NYSE_FE");
    map.insert("NYSE_FHN");
    map.insert("NYSE_FI");
    map.insert("NYSE_FICO");
    map.insert("NYSE_FIS");
    map.insert("NYSE_FLS");
    map.insert("NYSE_FMC");
    map.insert("NYSE_FNF");
    map.insert("NYSE_G");
    map.insert("NYSE_GAP");
    map.insert("NYSE_GD");
    map.insert("NYSE_GE");
    map.insert("NYSE_GGG");
    map.insert("NYSE_GIS");
    map.insert("NYSE_GL");
    map.insert("NYSE_GLW");
    map.insert("NYSE_GNRC");
    map.insert("NYSE_GPC");
    map.insert("NYSE_GPK");
    map.insert("NYSE_GPN");
    map.insert("NYSE_GS");
    map.insert("NYSE_GTLS");
    map.insert("NYSE_GWW");
    map.insert("NYSE_H");
    map.insert("NYSE_HAL");
    map.insert("NYSE_HBI");
    map.insert("NYSE_HD");
    map.insert("NYSE_HEI");
    map.insert("NYSE_HES");
    map.insert("NYSE_HIG");
    map.insert("NYSE_HOG");
    map.insert("NYSE_HPQ");
    map.insert("NYSE_HUBB");
    map.insert("NYSE_HUM");
    map.insert("NYSE_HUN");
    map.insert("NYSE_HXL");
    map.insert("NYSE_IBM");
    map.insert("NYSE_ICE");
    map.insert("NYSE_IEX");
    map.insert("NYSE_IFF");
    map.insert("NYSE_IGT");
    map.insert("NYSE_INGR");
    map.insert("NYSE_IP");
    map.insert("NYSE_IPG");
    map.insert("NYSE_IT");
    map.insert("NYSE_ITT");
    map.insert("NYSE_ITW");
    map.insert("NYSE_IVZ");
    map.insert("NYSE_J");
    map.insert("NYSE_JBL");
    map.insert("NYSE_JCI");
    map.insert("NYSE_JEF");
    map.insert("NYSE_JLL");
    map.insert("NYSE_JNJ");
    map.insert("NYSE_JNPR");
    map.insert("NYSE_JPM");
    map.insert("NYSE_JWN");
    map.insert("NYSE_K");
    map.insert("NYSE_KBR");
    map.insert("NYSE_KEY");
    map.insert("NYSE_KKR");
    map.insert("NYSE_KMB");
    map.insert("NYSE_KMX");
    map.insert("NYSE_KNX");
    map.insert("NYSE_KO");
    map.insert("NYSE_KR");
    map.insert("NYSE_KSS");
    map.insert("NYSE_L");
    map.insert("NYSE_LAD");
    map.insert("NYSE_LDOS");
    map.insert("NYSE_LEA");
    map.insert("NYSE_LEG");
    map.insert("NYSE_LEN");
    map.insert("NYSE_LH");
    map.insert("NYSE_LHX");
    map.insert("NYSE_LII");
    map.insert("NYSE_LLY");
    map.insert("NYSE_LMT");
    map.insert("NYSE_LNC");
    map.insert("NYSE_LOW");
    map.insert("NYSE_LPX");
    map.insert("NYSE_LUMN");
    map.insert("NYSE_LUV");
    map.insert("NYSE_LVS");
    map.insert("NYSE_LYB");
    map.insert("NYSE_LYV");
    map.insert("NYSE_M");
    map.insert("NYSE_MA");
    map.insert("NYSE_MAN");
    map.insert("NYSE_MAS");
    map.insert("NYSE_MCD");
    map.insert("NYSE_MCK");
    map.insert("NYSE_MCO");
    map.insert("NYSE_MDT");
    map.insert("NYSE_MET");
    map.insert("NYSE_MGM");
    map.insert("NYSE_MHK");
    map.insert("NYSE_MKL");
    map.insert("NYSE_MLM");
    map.insert("NYSE_MMC");
    map.insert("NYSE_MMM");
    map.insert("NYSE_MO");
    map.insert("NYSE_MODG");
    map.insert("NYSE_MOH");
    map.insert("NYSE_MOS");
    map.insert("NYSE_MRK");
    map.insert("NYSE_MS");
    map.insert("NYSE_MSCI");
    map.insert("NYSE_MSI");
    map.insert("NYSE_MSM");
    map.insert("NYSE_MTB");
    map.insert("NYSE_MTD");
    map.insert("NYSE_MTN");
    map.insert("NYSE_MTZ");
    map.insert("NYSE_NEE");
    map.insert("NYSE_NEM");
    map.insert("NYSE_NI");
    map.insert("NYSE_NKE");
    map.insert("NYSE_NOC");
    map.insert("NYSE_NOV");
    map.insert("NYSE_NRG");
    map.insert("NYSE_NSC");
    map.insert("NYSE_NUE");
    map.insert("NYSE_NYT");
    map.insert("NYSE_OC");
    map.insert("NYSE_OGE");
    map.insert("NYSE_OKE");
    map.insert("NYSE_OLN");
    map.insert("NYSE_OMC");
    map.insert("NYSE_ORCL");
    map.insert("NYSE_ORI");
    map.insert("NYSE_OSK");
    map.insert("NYSE_OXY");
    map.insert("NYSE_PEG");
    map.insert("NYSE_PFE");
    map.insert("NYSE_PG");
    map.insert("NYSE_PGR");
    map.insert("NYSE_PH");
    map.insert("NYSE_PHM");
    map.insert("NYSE_PII");
    map.insert("NYSE_PKG");
    map.insert("NYSE_PLD");
    map.insert("NYSE_PM");
    map.insert("NYSE_PNC");
    map.insert("NYSE_PNW");
    map.insert("NYSE_PPG");
    map.insert("NYSE_PPL");
    map.insert("NYSE_PRU");
    map.insert("NYSE_PSA");
    map.insert("NYSE_PVH");
    map.insert("NYSE_PWR");
    map.insert("NYSE_R");
    map.insert("NYSE_RCL");
    map.insert("NYSE_RF");
    map.insert("NYSE_RGA");
    map.insert("NYSE_RHI");
    map.insert("NYSE_RJF");
    map.insert("NYSE_RL");
    map.insert("NYSE_RMD");
    map.insert("NYSE_ROK");
    map.insert("NYSE_ROL");
    map.insert("NYSE_RPM");
    map.insert("NYSE_RRX");
    map.insert("NYSE_RS");
    map.insert("NYSE_RSG");
    map.insert("NYSE_RTX");
    map.insert("NYSE_RVTY");
    map.insert("NYSE_SAM");
    map.insert("NYSE_SCHW");
    map.insert("NYSE_SCI");
    map.insert("NYSE_SEE");
    map.insert("NYSE_SF");
    map.insert("NYSE_SHW");
    map.insert("NYSE_SJM");
    map.insert("NYSE_SKX");
    map.insert("NYSE_SLB");
    map.insert("NYSE_SMG");
    map.insert("NYSE_SNA");
    map.insert("NYSE_SNV");
    map.insert("NYSE_SNX");
    map.insert("NYSE_SO");
    map.insert("NYSE_SPG");
    map.insert("NYSE_SPGI");
    map.insert("NYSE_SPR");
    map.insert("NYSE_SRE");
    map.insert("NYSE_ST");
    map.insert("NYSE_STE");
    map.insert("NYSE_STT");
    map.insert("NYSE_STZ");
    map.insert("NYSE_SWK");
    map.insert("NYSE_SYK");
    map.insert("NYSE_SYY");
    map.insert("NYSE_T");
    map.insert("NYSE_TAP");
    map.insert("NYSE_TDG");
    map.insert("NYSE_TDY");
    map.insert("NYSE_TFC");
    map.insert("NYSE_TFX");
    map.insert("NYSE_TGT");
    map.insert("NYSE_THC");
    map.insert("NYSE_THO");
    map.insert("NYSE_TJX");
    map.insert("NYSE_TKR");
    map.insert("NYSE_TMO");
    map.insert("NYSE_TNL");
    map.insert("NYSE_TOL");
    map.insert("NYSE_TPL");
    map.insert("NYSE_TPR");
    map.insert("NYSE_TPX");
    map.insert("NYSE_TREX");
    map.insert("NYSE_TRV");
    map.insert("NYSE_TSN");
    map.insert("NYSE_TTC");
    map.insert("NYSE_TXT");
    map.insert("NYSE_TYL");
    map.insert("NYSE_UAA");
    map.insert("NYSE_UGI");
    map.insert("NYSE_UHS");
    map.insert("NYSE_UNH");
    map.insert("NYSE_UNM");
    map.insert("NYSE_UNP");
    map.insert("NYSE_UPS");
    map.insert("NYSE_URI");
    map.insert("NYSE_USB");
    map.insert("NYSE_V");
    map.insert("NYSE_VFC");
    map.insert("NYSE_VLO");
    map.insert("NYSE_VMC");
    map.insert("NYSE_VYX");
    map.insert("NYSE_VZ");
    map.insert("NYSE_WAB");
    map.insert("NYSE_WAL");
    map.insert("NYSE_WAT");
    map.insert("NYSE_WCC");
    map.insert("NYSE_WEC");
    map.insert("NYSE_WEX");
    map.insert("NYSE_WFC");
    map.insert("NYSE_WHR");
    map.insert("NYSE_WM");
    map.insert("NYSE_WMB");
    map.insert("NYSE_WMT");
    map.insert("NYSE_WOLF");
    map.insert("NYSE_WRB");
    map.insert("NYSE_WSM");
    map.insert("NYSE_WSO");
    map.insert("NYSE_WST");
    map.insert("NYSE_WTI");
    map.insert("NYSE_WTRG");
    map.insert("NYSE_WU");
    map.insert("NYSE_X");
    map.insert("NYSE_XOM");
    map.insert("NYSE_XPO");
    map.insert("NYSE_YUM");
    map.insert("NYSE_ZBH");
    map.insert("TVC_CAC40");
    map.insert("TVC_DJI");
    map.insert("TVC_DXY");
    map.insert("TVC_GOLD");
    map.insert("TVC_IBEX35");
    map.insert("TVC_NI225");
    map.insert("TVC_SILVER");
    map.insert("TVC_SPX");
    map.insert("TVC_SX5E");
    map.insert("TVC_UKOIL");
    map.insert("TVC_USOIL");
    map.insert("XETR_DAX");
    map
}

pub fn market_to_market_id() -> HashMap<&'static str, MarketId> {
    let mut map = HashMap::default();
    map.insert("AMEX_DIA", 0);
    map.insert("AMEX_EEM", 1);
    map.insert("AMEX_EFA", 2);
    map.insert("AMEX_EWJ", 3);
    map.insert("AMEX_EWT", 4);
    map.insert("AMEX_EWY", 5);
    map.insert("AMEX_EWZ", 6);
    map.insert("AMEX_FDN", 7);
    map.insert("AMEX_FVD", 8);
    map.insert("AMEX_GDX", 9);
    map.insert("AMEX_GDXJ", 10);
    map.insert("AMEX_GLD", 11);
    map.insert("AMEX_IHI", 12);
    map.insert("AMEX_IJH", 13);
    map.insert("AMEX_IJR", 14);
    map.insert("AMEX_IJS", 15);
    map.insert("AMEX_ITOT", 16);
    map.insert("AMEX_IVE", 17);
    map.insert("AMEX_IVW", 18);
    map.insert("AMEX_IWB", 19);
    map.insert("AMEX_IWD", 20);
    map.insert("AMEX_IWF", 21);
    map.insert("AMEX_IWM", 22);
    map.insert("AMEX_IWN", 23);
    map.insert("AMEX_IWO", 24);
    map.insert("AMEX_IWS", 25);
    map.insert("AMEX_IYR", 26);
    map.insert("AMEX_IYW", 27);
    map.insert("AMEX_KRE", 28);
    map.insert("AMEX_MDY", 29);
    map.insert("AMEX_OEF", 30);
    map.insert("AMEX_RSP", 31);
    map.insert("AMEX_SCHB", 32);
    map.insert("AMEX_SCHF", 33);
    map.insert("AMEX_SCHG", 34);
    map.insert("AMEX_SCHV", 35);
    map.insert("AMEX_SLV", 36);
    map.insert("AMEX_SPDW", 37);
    map.insert("AMEX_SPY", 38);
    map.insert("AMEX_SPYG", 39);
    map.insert("AMEX_SPYV", 40);
    map.insert("AMEX_TIP", 41);
    map.insert("AMEX_VDE", 42);
    map.insert("AMEX_VEA", 43);
    map.insert("AMEX_VGT", 44);
    map.insert("AMEX_VHT", 45);
    map.insert("AMEX_VNQ", 46);
    map.insert("AMEX_VOE", 47);
    map.insert("AMEX_VPL", 48);
    map.insert("AMEX_VT", 49);
    map.insert("AMEX_VTI", 50);
    map.insert("AMEX_VWO", 51);
    map.insert("AMEX_VXF", 52);
    map.insert("AMEX_XBI", 53);
    map.insert("AMEX_XLB", 54);
    map.insert("AMEX_XLE", 55);
    map.insert("AMEX_XLF", 56);
    map.insert("AMEX_XLI", 57);
    map.insert("AMEX_XLK", 58);
    map.insert("AMEX_XLP", 59);
    map.insert("AMEX_XLU", 60);
    map.insert("AMEX_XLV", 61);
    map.insert("AMEX_XLY", 62);
    map.insert("CAPITALCOM_RTY", 63);
    map.insert("CBOE_EFG", 64);
    map.insert("CBOE_EFV", 65);
    map.insert("CBOE_EZU", 66);
    map.insert("CBOE_IGV", 67);
    map.insert("FX_AUDCAD", 68);
    map.insert("FX_AUDCHF", 69);
    map.insert("FX_AUDJPY", 70);
    map.insert("FX_AUDNZD", 71);
    map.insert("FX_AUDUSD", 72);
    map.insert("FX_AUS200", 73);
    map.insert("FX_CADCHF", 74);
    map.insert("FX_CADJPY", 75);
    map.insert("FX_CHFJPY", 76);
    map.insert("FX_EURAUD", 77);
    map.insert("FX_EURCAD", 78);
    map.insert("FX_EURCHF", 79);
    map.insert("FX_EURGBP", 80);
    map.insert("FX_EURJPY", 81);
    map.insert("FX_EURNZD", 82);
    map.insert("FX_EURUSD", 83);
    map.insert("FX_GBPAUD", 84);
    map.insert("FX_GBPCAD", 85);
    map.insert("FX_GBPCHF", 86);
    map.insert("FX_GBPJPY", 87);
    map.insert("FX_GBPNZD", 88);
    map.insert("FX_GBPUSD", 89);
    map.insert("FX_IDC_CADUSD", 90);
    map.insert("FX_IDC_CHFUSD", 91);
    map.insert("FX_IDC_EURMXN", 92);
    map.insert("FX_IDC_EURNOK", 93);
    map.insert("FX_IDC_EURSEK", 94);
    map.insert("FX_IDC_GBPMXN", 95);
    map.insert("FX_IDC_GBPNOK", 96);
    map.insert("FX_IDC_GBPSEK", 97);
    map.insert("FX_IDC_JPYUSD", 98);
    map.insert("FX_IDC_USDNOK", 99);
    map.insert("FX_IDC_USDSGD", 100);
    map.insert("FX_NZDCAD", 101);
    map.insert("FX_NZDCHF", 102);
    map.insert("FX_NZDJPY", 103);
    map.insert("FX_NZDUSD", 104);
    map.insert("FX_UK100", 105);
    map.insert("FX_USDCAD", 106);
    map.insert("FX_USDCHF", 107);
    map.insert("FX_USDJPY", 108);
    map.insert("FX_USDMXN", 109);
    map.insert("FX_USDSEK", 110);
    map.insert("NASDAQ_AAL", 111);
    map.insert("NASDAQ_AAPL", 112);
    map.insert("NASDAQ_AAXJ", 113);
    map.insert("NASDAQ_ADBE", 114);
    map.insert("NASDAQ_ADI", 115);
    map.insert("NASDAQ_ADP", 116);
    map.insert("NASDAQ_ADSK", 117);
    map.insert("NASDAQ_AEP", 118);
    map.insert("NASDAQ_AKAM", 119);
    map.insert("NASDAQ_ALGN", 120);
    map.insert("NASDAQ_ALNY", 121);
    map.insert("NASDAQ_AMAT", 122);
    map.insert("NASDAQ_AMD", 123);
    map.insert("NASDAQ_AMED", 124);
    map.insert("NASDAQ_AMGN", 125);
    map.insert("NASDAQ_AMKR", 126);
    map.insert("NASDAQ_AMZN", 127);
    map.insert("NASDAQ_ANSS", 128);
    map.insert("NASDAQ_APA", 129);
    map.insert("NASDAQ_ARCC", 130);
    map.insert("NASDAQ_ARWR", 131);
    map.insert("NASDAQ_AVGO", 132);
    map.insert("NASDAQ_AXON", 133);
    map.insert("NASDAQ_AZPN", 134);
    map.insert("NASDAQ_AZTA", 135);
    map.insert("NASDAQ_BIIB", 136);
    map.insert("NASDAQ_BKNG", 137);
    map.insert("NASDAQ_BKR", 138);
    map.insert("NASDAQ_BMRN", 139);
    map.insert("NASDAQ_BRKR", 140);
    map.insert("NASDAQ_CACC", 141);
    map.insert("NASDAQ_CAR", 142);
    map.insert("NASDAQ_CASY", 143);
    map.insert("NASDAQ_CDNS", 144);
    map.insert("NASDAQ_CGNX", 145);
    map.insert("NASDAQ_CHDN", 146);
    map.insert("NASDAQ_CHRW", 147);
    map.insert("NASDAQ_CHTR", 148);
    map.insert("NASDAQ_CINF", 149);
    map.insert("NASDAQ_CMCSA", 150);
    map.insert("NASDAQ_CME", 151);
    map.insert("NASDAQ_COLM", 152);
    map.insert("NASDAQ_COO", 153);
    map.insert("NASDAQ_COST", 154);
    map.insert("NASDAQ_CPRT", 155);
    map.insert("NASDAQ_CROX", 156);
    map.insert("NASDAQ_CSCO", 157);
    map.insert("NASDAQ_CSGP", 158);
    map.insert("NASDAQ_CSX", 159);
    map.insert("NASDAQ_CTAS", 160);
    map.insert("NASDAQ_CTSH", 161);
    map.insert("NASDAQ_DLTR", 162);
    map.insert("NASDAQ_DOX", 163);
    map.insert("NASDAQ_DVY", 164);
    map.insert("NASDAQ_DXCM", 165);
    map.insert("NASDAQ_EA", 166);
    map.insert("NASDAQ_EBAY", 167);
    map.insert("NASDAQ_EEFT", 168);
    map.insert("NASDAQ_EMB", 169);
    map.insert("NASDAQ_ENTG", 170);
    map.insert("NASDAQ_EVRG", 171);
    map.insert("NASDAQ_EWBC", 172);
    map.insert("NASDAQ_EXC", 173);
    map.insert("NASDAQ_EXEL", 174);
    map.insert("NASDAQ_EXPE", 175);
    map.insert("NASDAQ_FAST", 176);
    map.insert("NASDAQ_FFIV", 177);
    map.insert("NASDAQ_FITB", 178);
    map.insert("NASDAQ_FLEX", 179);
    map.insert("NASDAQ_FSLR", 180);
    map.insert("NASDAQ_FTNT", 181);
    map.insert("NASDAQ_GEN", 182);
    map.insert("NASDAQ_GILD", 183);
    map.insert("NASDAQ_GNTX", 184);
    map.insert("NASDAQ_GOOGL", 185);
    map.insert("NASDAQ_GT", 186);
    map.insert("NASDAQ_HALO", 187);
    map.insert("NASDAQ_HAS", 188);
    map.insert("NASDAQ_HBAN", 189);
    map.insert("NASDAQ_HELE", 190);
    map.insert("NASDAQ_HOLX", 191);
    map.insert("NASDAQ_HON", 192);
    map.insert("NASDAQ_HSIC", 193);
    map.insert("NASDAQ_IBB", 194);
    map.insert("NASDAQ_IBKR", 195);
    map.insert("NASDAQ_ICLN", 196);
    map.insert("NASDAQ_IDXX", 197);
    map.insert("NASDAQ_ILMN", 198);
    map.insert("NASDAQ_INCY", 199);
    map.insert("NASDAQ_INTC", 200);
    map.insert("NASDAQ_INTU", 201);
    map.insert("NASDAQ_IONS", 202);
    map.insert("NASDAQ_IPGP", 203);
    map.insert("NASDAQ_IRDM", 204);
    map.insert("NASDAQ_ISRG", 205);
    map.insert("NASDAQ_IUSG", 206);
    map.insert("NASDAQ_IUSV", 207);
    map.insert("NASDAQ_JBHT", 208);
    map.insert("NASDAQ_JBLU", 209);
    map.insert("NASDAQ_JKHY", 210);
    map.insert("NASDAQ_KDP", 211);
    map.insert("NASDAQ_KLAC", 212);
    map.insert("NASDAQ_LKQ", 213);
    map.insert("NASDAQ_LNT", 214);
    map.insert("NASDAQ_LNW", 215);
    map.insert("NASDAQ_LRCX", 216);
    map.insert("NASDAQ_LSCC", 217);
    map.insert("NASDAQ_LSTR", 218);
    map.insert("NASDAQ_MANH", 219);
    map.insert("NASDAQ_MAR", 220);
    map.insert("NASDAQ_MASI", 221);
    map.insert("NASDAQ_MAT", 222);
    map.insert("NASDAQ_MCHP", 223);
    map.insert("NASDAQ_MDLZ", 224);
    map.insert("NASDAQ_MIDD", 225);
    map.insert("NASDAQ_MKSI", 226);
    map.insert("NASDAQ_MKTX", 227);
    map.insert("NASDAQ_MNST", 228);
    map.insert("NASDAQ_MPWR", 229);
    map.insert("NASDAQ_MRVL", 230);
    map.insert("NASDAQ_MSFT", 231);
    map.insert("NASDAQ_MSTR", 232);
    map.insert("NASDAQ_MU", 233);
    map.insert("NASDAQ_NBIX", 234);
    map.insert("NASDAQ_NDAQ", 235);
    map.insert("NASDAQ_NDSN", 236);
    map.insert("NASDAQ_NDX", 237);
    map.insert("NASDAQ_NFLX", 238);
    map.insert("NASDAQ_NTAP", 239);
    map.insert("NASDAQ_NTRS", 240);
    map.insert("NASDAQ_NVDA", 241);
    map.insert("NASDAQ_NWL", 242);
    map.insert("NASDAQ_NXST", 243);
    map.insert("NASDAQ_ODFL", 244);
    map.insert("NASDAQ_OLED", 245);
    map.insert("NASDAQ_OMCL", 246);
    map.insert("NASDAQ_ON", 247);
    map.insert("NASDAQ_ORLY", 248);
    map.insert("NASDAQ_OZK", 249);
    map.insert("NASDAQ_PARA", 250);
    map.insert("NASDAQ_PAYX", 251);
    map.insert("NASDAQ_PCAR", 252);
    map.insert("NASDAQ_PEGA", 253);
    map.insert("NASDAQ_PENN", 254);
    map.insert("NASDAQ_PEP", 255);
    map.insert("NASDAQ_PFF", 256);
    map.insert("NASDAQ_PFG", 257);
    map.insert("NASDAQ_PNFP", 258);
    map.insert("NASDAQ_PODD", 259);
    map.insert("NASDAQ_POOL", 260);
    map.insert("NASDAQ_PTC", 261);
    map.insert("NASDAQ_QCOM", 262);
    map.insert("NASDAQ_QQQ", 263);
    map.insert("NASDAQ_REGN", 264);
    map.insert("NASDAQ_RGEN", 265);
    map.insert("NASDAQ_RGLD", 266);
    map.insert("NASDAQ_ROP", 267);
    map.insert("NASDAQ_ROST", 268);
    map.insert("NASDAQ_SAIA", 269);
    map.insert("NASDAQ_SBUX", 270);
    map.insert("NASDAQ_SCZ", 271);
    map.insert("NASDAQ_SLAB", 272);
    map.insert("NASDAQ_SLM", 273);
    map.insert("NASDAQ_SMH", 274);
    map.insert("NASDAQ_SNPS", 275);
    map.insert("NASDAQ_SOXX", 276);
    map.insert("NASDAQ_SRPT", 277);
    map.insert("NASDAQ_SSNC", 278);
    map.insert("NASDAQ_STLD", 279);
    map.insert("NASDAQ_SWKS", 280);
    map.insert("NASDAQ_SYNA", 281);
    map.insert("NASDAQ_TECH", 282);
    map.insert("NASDAQ_TER", 283);
    map.insert("NASDAQ_TLT", 284);
    map.insert("NASDAQ_TMUS", 285);
    map.insert("NASDAQ_TRMB", 286);
    map.insert("NASDAQ_TROW", 287);
    map.insert("NASDAQ_TSCO", 288);
    map.insert("NASDAQ_TTEK", 289);
    map.insert("NASDAQ_TTWO", 290);
    map.insert("NASDAQ_TXN", 291);
    map.insert("NASDAQ_TXRH", 292);
    map.insert("NASDAQ_UAL", 293);
    map.insert("NASDAQ_ULTA", 294);
    map.insert("NASDAQ_UTHR", 295);
    map.insert("NASDAQ_VRSK", 296);
    map.insert("NASDAQ_VRSN", 297);
    map.insert("NASDAQ_VRTX", 298);
    map.insert("NASDAQ_VTRS", 299);
    map.insert("NASDAQ_WBA", 300);
    map.insert("NASDAQ_WDC", 301);
    map.insert("NASDAQ_WWD", 302);
    map.insert("NASDAQ_WYNN", 303);
    map.insert("NASDAQ_XEL", 304);
    map.insert("NASDAQ_XRAY", 305);
    map.insert("NASDAQ_ZBRA", 306);
    map.insert("NASDAQ_ZD", 307);
    map.insert("NASDAQ_ZION", 308);
    map.insert("NYSE_A", 309);
    map.insert("NYSE_AA", 310);
    map.insert("NYSE_AAP", 311);
    map.insert("NYSE_ABT", 312);
    map.insert("NYSE_ACM", 313);
    map.insert("NYSE_ACN", 314);
    map.insert("NYSE_ADM", 315);
    map.insert("NYSE_AES", 316);
    map.insert("NYSE_AFG", 317);
    map.insert("NYSE_AFL", 318);
    map.insert("NYSE_AGCO", 319);
    map.insert("NYSE_AIG", 320);
    map.insert("NYSE_AIZ", 321);
    map.insert("NYSE_AJG", 322);
    map.insert("NYSE_ALB", 323);
    map.insert("NYSE_ALK", 324);
    map.insert("NYSE_ALL", 325);
    map.insert("NYSE_AME", 326);
    map.insert("NYSE_AMG", 327);
    map.insert("NYSE_AMP", 328);
    map.insert("NYSE_AMT", 329);
    map.insert("NYSE_AN", 330);
    map.insert("NYSE_AON", 331);
    map.insert("NYSE_AOS", 332);
    map.insert("NYSE_APD", 333);
    map.insert("NYSE_APH", 334);
    map.insert("NYSE_ARW", 335);
    map.insert("NYSE_ASH", 336);
    map.insert("NYSE_AVY", 337);
    map.insert("NYSE_AWI", 338);
    map.insert("NYSE_AWK", 339);
    map.insert("NYSE_AXP", 340);
    map.insert("NYSE_AYI", 341);
    map.insert("NYSE_AZO", 342);
    map.insert("NYSE_BA", 343);
    map.insert("NYSE_BAC", 344);
    map.insert("NYSE_BALL", 345);
    map.insert("NYSE_BAX", 346);
    map.insert("NYSE_BBY", 347);
    map.insert("NYSE_BC", 348);
    map.insert("NYSE_BDX", 349);
    map.insert("NYSE_BEN", 350);
    map.insert("NYSE_BIO", 351);
    map.insert("NYSE_BK", 352);
    map.insert("NYSE_BLDR", 353);
    map.insert("NYSE_BLK", 354);
    map.insert("NYSE_BMY", 355);
    map.insert("NYSE_BR", 356);
    map.insert("NYSE_BRK.A", 357);
    map.insert("NYSE_BRO", 358);
    map.insert("NYSE_BSX", 359);
    map.insert("NYSE_BWA", 360);
    map.insert("NYSE_BX", 361);
    map.insert("NYSE_BYD", 362);
    map.insert("NYSE_C", 363);
    map.insert("NYSE_CACI", 364);
    map.insert("NYSE_CAG", 365);
    map.insert("NYSE_CAH", 366);
    map.insert("NYSE_CAT", 367);
    map.insert("NYSE_CB", 368);
    map.insert("NYSE_CBRE", 369);
    map.insert("NYSE_CCI", 370);
    map.insert("NYSE_CCK", 371);
    map.insert("NYSE_CCL", 372);
    map.insert("NYSE_CE", 373);
    map.insert("NYSE_CF", 374);
    map.insert("NYSE_CFR", 375);
    map.insert("NYSE_CHD", 376);
    map.insert("NYSE_CHE", 377);
    map.insert("NYSE_CI", 378);
    map.insert("NYSE_CIEN", 379);
    map.insert("NYSE_CL", 380);
    map.insert("NYSE_CLF", 381);
    map.insert("NYSE_CLX", 382);
    map.insert("NYSE_CMA", 383);
    map.insert("NYSE_CMG", 384);
    map.insert("NYSE_CMI", 385);
    map.insert("NYSE_CMS", 386);
    map.insert("NYSE_CNC", 387);
    map.insert("NYSE_CNP", 388);
    map.insert("NYSE_COF", 389);
    map.insert("NYSE_COHR", 390);
    map.insert("NYSE_COP", 391);
    map.insert("NYSE_COR", 392);
    map.insert("NYSE_CRL", 393);
    map.insert("NYSE_CRM", 394);
    map.insert("NYSE_CSL", 395);
    map.insert("NYSE_CVS", 396);
    map.insert("NYSE_CVX", 397);
    map.insert("NYSE_D", 398);
    map.insert("NYSE_DAL", 399);
    map.insert("NYSE_DAR", 400);
    map.insert("NYSE_DD", 401);
    map.insert("NYSE_DE", 402);
    map.insert("NYSE_DECK", 403);
    map.insert("NYSE_DFS", 404);
    map.insert("NYSE_DG", 405);
    map.insert("NYSE_DGX", 406);
    map.insert("NYSE_DHI", 407);
    map.insert("NYSE_DHR", 408);
    map.insert("NYSE_DIS", 409);
    map.insert("NYSE_DKS", 410);
    map.insert("NYSE_DLB", 411);
    map.insert("NYSE_DOV", 412);
    map.insert("NYSE_DPZ", 413);
    map.insert("NYSE_DRI", 414);
    map.insert("NYSE_DTE", 415);
    map.insert("NYSE_DUK", 416);
    map.insert("NYSE_DVA", 417);
    map.insert("NYSE_DVN", 418);
    map.insert("NYSE_ECL", 419);
    map.insert("NYSE_EFX", 420);
    map.insert("NYSE_EHC", 421);
    map.insert("NYSE_EIX", 422);
    map.insert("NYSE_EL", 423);
    map.insert("NYSE_ELV", 424);
    map.insert("NYSE_EME", 425);
    map.insert("NYSE_EMN", 426);
    map.insert("NYSE_EMR", 427);
    map.insert("NYSE_EOG", 428);
    map.insert("NYSE_EQT", 429);
    map.insert("NYSE_ES", 430);
    map.insert("NYSE_ETN", 431);
    map.insert("NYSE_ETR", 432);
    map.insert("NYSE_EVR", 433);
    map.insert("NYSE_EW", 434);
    map.insert("NYSE_EXP", 435);
    map.insert("NYSE_EXPD", 436);
    map.insert("NYSE_F", 437);
    map.insert("NYSE_FAF", 438);
    map.insert("NYSE_FCX", 439);
    map.insert("NYSE_FDS", 440);
    map.insert("NYSE_FDX", 441);
    map.insert("NYSE_FE", 442);
    map.insert("NYSE_FHN", 443);
    map.insert("NYSE_FI", 444);
    map.insert("NYSE_FICO", 445);
    map.insert("NYSE_FIS", 446);
    map.insert("NYSE_FLS", 447);
    map.insert("NYSE_FMC", 448);
    map.insert("NYSE_FNF", 449);
    map.insert("NYSE_G", 450);
    map.insert("NYSE_GAP", 451);
    map.insert("NYSE_GD", 452);
    map.insert("NYSE_GE", 453);
    map.insert("NYSE_GGG", 454);
    map.insert("NYSE_GIS", 455);
    map.insert("NYSE_GL", 456);
    map.insert("NYSE_GLW", 457);
    map.insert("NYSE_GNRC", 458);
    map.insert("NYSE_GPC", 459);
    map.insert("NYSE_GPK", 460);
    map.insert("NYSE_GPN", 461);
    map.insert("NYSE_GS", 462);
    map.insert("NYSE_GTLS", 463);
    map.insert("NYSE_GWW", 464);
    map.insert("NYSE_H", 465);
    map.insert("NYSE_HAL", 466);
    map.insert("NYSE_HBI", 467);
    map.insert("NYSE_HD", 468);
    map.insert("NYSE_HEI", 469);
    map.insert("NYSE_HES", 470);
    map.insert("NYSE_HIG", 471);
    map.insert("NYSE_HOG", 472);
    map.insert("NYSE_HPQ", 473);
    map.insert("NYSE_HUBB", 474);
    map.insert("NYSE_HUM", 475);
    map.insert("NYSE_HUN", 476);
    map.insert("NYSE_HXL", 477);
    map.insert("NYSE_IBM", 478);
    map.insert("NYSE_ICE", 479);
    map.insert("NYSE_IEX", 480);
    map.insert("NYSE_IFF", 481);
    map.insert("NYSE_IGT", 482);
    map.insert("NYSE_INGR", 483);
    map.insert("NYSE_IP", 484);
    map.insert("NYSE_IPG", 485);
    map.insert("NYSE_IT", 486);
    map.insert("NYSE_ITT", 487);
    map.insert("NYSE_ITW", 488);
    map.insert("NYSE_IVZ", 489);
    map.insert("NYSE_J", 490);
    map.insert("NYSE_JBL", 491);
    map.insert("NYSE_JCI", 492);
    map.insert("NYSE_JEF", 493);
    map.insert("NYSE_JLL", 494);
    map.insert("NYSE_JNJ", 495);
    map.insert("NYSE_JNPR", 496);
    map.insert("NYSE_JPM", 497);
    map.insert("NYSE_JWN", 498);
    map.insert("NYSE_K", 499);
    map.insert("NYSE_KBR", 500);
    map.insert("NYSE_KEY", 501);
    map.insert("NYSE_KKR", 502);
    map.insert("NYSE_KMB", 503);
    map.insert("NYSE_KMX", 504);
    map.insert("NYSE_KNX", 505);
    map.insert("NYSE_KO", 506);
    map.insert("NYSE_KR", 507);
    map.insert("NYSE_KSS", 508);
    map.insert("NYSE_L", 509);
    map.insert("NYSE_LAD", 510);
    map.insert("NYSE_LDOS", 511);
    map.insert("NYSE_LEA", 512);
    map.insert("NYSE_LEG", 513);
    map.insert("NYSE_LEN", 514);
    map.insert("NYSE_LH", 515);
    map.insert("NYSE_LHX", 516);
    map.insert("NYSE_LII", 517);
    map.insert("NYSE_LLY", 518);
    map.insert("NYSE_LMT", 519);
    map.insert("NYSE_LNC", 520);
    map.insert("NYSE_LOW", 521);
    map.insert("NYSE_LPX", 522);
    map.insert("NYSE_LUMN", 523);
    map.insert("NYSE_LUV", 524);
    map.insert("NYSE_LVS", 525);
    map.insert("NYSE_LYB", 526);
    map.insert("NYSE_LYV", 527);
    map.insert("NYSE_M", 528);
    map.insert("NYSE_MA", 529);
    map.insert("NYSE_MAN", 530);
    map.insert("NYSE_MAS", 531);
    map.insert("NYSE_MCD", 532);
    map.insert("NYSE_MCK", 533);
    map.insert("NYSE_MCO", 534);
    map.insert("NYSE_MDT", 535);
    map.insert("NYSE_MET", 536);
    map.insert("NYSE_MGM", 537);
    map.insert("NYSE_MHK", 538);
    map.insert("NYSE_MKL", 539);
    map.insert("NYSE_MLM", 540);
    map.insert("NYSE_MMC", 541);
    map.insert("NYSE_MMM", 542);
    map.insert("NYSE_MO", 543);
    map.insert("NYSE_MODG", 544);
    map.insert("NYSE_MOH", 545);
    map.insert("NYSE_MOS", 546);
    map.insert("NYSE_MRK", 547);
    map.insert("NYSE_MS", 548);
    map.insert("NYSE_MSCI", 549);
    map.insert("NYSE_MSI", 550);
    map.insert("NYSE_MSM", 551);
    map.insert("NYSE_MTB", 552);
    map.insert("NYSE_MTD", 553);
    map.insert("NYSE_MTN", 554);
    map.insert("NYSE_MTZ", 555);
    map.insert("NYSE_NEE", 556);
    map.insert("NYSE_NEM", 557);
    map.insert("NYSE_NI", 558);
    map.insert("NYSE_NKE", 559);
    map.insert("NYSE_NOC", 560);
    map.insert("NYSE_NOV", 561);
    map.insert("NYSE_NRG", 562);
    map.insert("NYSE_NSC", 563);
    map.insert("NYSE_NUE", 564);
    map.insert("NYSE_NYT", 565);
    map.insert("NYSE_OC", 566);
    map.insert("NYSE_OGE", 567);
    map.insert("NYSE_OKE", 568);
    map.insert("NYSE_OLN", 569);
    map.insert("NYSE_OMC", 570);
    map.insert("NYSE_ORCL", 571);
    map.insert("NYSE_ORI", 572);
    map.insert("NYSE_OSK", 573);
    map.insert("NYSE_OXY", 574);
    map.insert("NYSE_PEG", 575);
    map.insert("NYSE_PFE", 576);
    map.insert("NYSE_PG", 577);
    map.insert("NYSE_PGR", 578);
    map.insert("NYSE_PH", 579);
    map.insert("NYSE_PHM", 580);
    map.insert("NYSE_PII", 581);
    map.insert("NYSE_PKG", 582);
    map.insert("NYSE_PLD", 583);
    map.insert("NYSE_PM", 584);
    map.insert("NYSE_PNC", 585);
    map.insert("NYSE_PNW", 586);
    map.insert("NYSE_PPG", 587);
    map.insert("NYSE_PPL", 588);
    map.insert("NYSE_PRU", 589);
    map.insert("NYSE_PSA", 590);
    map.insert("NYSE_PVH", 591);
    map.insert("NYSE_PWR", 592);
    map.insert("NYSE_R", 593);
    map.insert("NYSE_RCL", 594);
    map.insert("NYSE_RF", 595);
    map.insert("NYSE_RGA", 596);
    map.insert("NYSE_RHI", 597);
    map.insert("NYSE_RJF", 598);
    map.insert("NYSE_RL", 599);
    map.insert("NYSE_RMD", 600);
    map.insert("NYSE_ROK", 601);
    map.insert("NYSE_ROL", 602);
    map.insert("NYSE_RPM", 603);
    map.insert("NYSE_RRX", 604);
    map.insert("NYSE_RS", 605);
    map.insert("NYSE_RSG", 606);
    map.insert("NYSE_RTX", 607);
    map.insert("NYSE_RVTY", 608);
    map.insert("NYSE_SAM", 609);
    map.insert("NYSE_SCHW", 610);
    map.insert("NYSE_SCI", 611);
    map.insert("NYSE_SEE", 612);
    map.insert("NYSE_SF", 613);
    map.insert("NYSE_SHW", 614);
    map.insert("NYSE_SJM", 615);
    map.insert("NYSE_SKX", 616);
    map.insert("NYSE_SLB", 617);
    map.insert("NYSE_SMG", 618);
    map.insert("NYSE_SNA", 619);
    map.insert("NYSE_SNV", 620);
    map.insert("NYSE_SNX", 621);
    map.insert("NYSE_SO", 622);
    map.insert("NYSE_SPG", 623);
    map.insert("NYSE_SPGI", 624);
    map.insert("NYSE_SPR", 625);
    map.insert("NYSE_SRE", 626);
    map.insert("NYSE_ST", 627);
    map.insert("NYSE_STE", 628);
    map.insert("NYSE_STT", 629);
    map.insert("NYSE_STZ", 630);
    map.insert("NYSE_SWK", 631);
    map.insert("NYSE_SYK", 632);
    map.insert("NYSE_SYY", 633);
    map.insert("NYSE_T", 634);
    map.insert("NYSE_TAP", 635);
    map.insert("NYSE_TDG", 636);
    map.insert("NYSE_TDY", 637);
    map.insert("NYSE_TFC", 638);
    map.insert("NYSE_TFX", 639);
    map.insert("NYSE_TGT", 640);
    map.insert("NYSE_THC", 641);
    map.insert("NYSE_THO", 642);
    map.insert("NYSE_TJX", 643);
    map.insert("NYSE_TKR", 644);
    map.insert("NYSE_TMO", 645);
    map.insert("NYSE_TNL", 646);
    map.insert("NYSE_TOL", 647);
    map.insert("NYSE_TPL", 648);
    map.insert("NYSE_TPR", 649);
    map.insert("NYSE_TPX", 650);
    map.insert("NYSE_TREX", 651);
    map.insert("NYSE_TRV", 652);
    map.insert("NYSE_TSN", 653);
    map.insert("NYSE_TTC", 654);
    map.insert("NYSE_TXT", 655);
    map.insert("NYSE_TYL", 656);
    map.insert("NYSE_UAA", 657);
    map.insert("NYSE_UGI", 658);
    map.insert("NYSE_UHS", 659);
    map.insert("NYSE_UNH", 660);
    map.insert("NYSE_UNM", 661);
    map.insert("NYSE_UNP", 662);
    map.insert("NYSE_UPS", 663);
    map.insert("NYSE_URI", 664);
    map.insert("NYSE_USB", 665);
    map.insert("NYSE_V", 666);
    map.insert("NYSE_VFC", 667);
    map.insert("NYSE_VLO", 668);
    map.insert("NYSE_VMC", 669);
    map.insert("NYSE_VYX", 670);
    map.insert("NYSE_VZ", 671);
    map.insert("NYSE_WAB", 672);
    map.insert("NYSE_WAL", 673);
    map.insert("NYSE_WAT", 674);
    map.insert("NYSE_WCC", 675);
    map.insert("NYSE_WEC", 676);
    map.insert("NYSE_WEX", 677);
    map.insert("NYSE_WFC", 678);
    map.insert("NYSE_WHR", 679);
    map.insert("NYSE_WM", 680);
    map.insert("NYSE_WMB", 681);
    map.insert("NYSE_WMT", 682);
    map.insert("NYSE_WOLF", 683);
    map.insert("NYSE_WRB", 684);
    map.insert("NYSE_WSM", 685);
    map.insert("NYSE_WSO", 686);
    map.insert("NYSE_WST", 687);
    map.insert("NYSE_WTI", 688);
    map.insert("NYSE_WTRG", 689);
    map.insert("NYSE_WU", 690);
    map.insert("NYSE_X", 691);
    map.insert("NYSE_XOM", 692);
    map.insert("NYSE_XPO", 693);
    map.insert("NYSE_YUM", 694);
    map.insert("NYSE_ZBH", 695);
    map.insert("TVC_CAC40", 696);
    map.insert("TVC_DJI", 697);
    map.insert("TVC_DXY", 698);
    map.insert("TVC_GOLD", 699);
    map.insert("TVC_IBEX35", 700);
    map.insert("TVC_NI225", 701);
    map.insert("TVC_SILVER", 702);
    map.insert("TVC_SPX", 703);
    map.insert("TVC_SX5E", 704);
    map.insert("TVC_UKOIL", 705);
    map.insert("TVC_USOIL", 706);
    map.insert("XETR_DAX", 707);
    map.insert("BNC_BLX", 708);
    map.insert("TVC_NDX", 709);
    map.insert("CBOT_KE1!", 710);
    map.insert("CBOT_UB1!", 711);
    map.insert("CBOT_ZC1!", 712);
    map.insert("CBOT_ZF1!", 713);
    map.insert("CBOT_ZL1!", 714);
    map.insert("CBOT_ZM1!", 715);
    map.insert("CBOT_ZN1!", 716);
    map.insert("CBOT_ZS1!", 717);
    map.insert("CBOT_ZT1!", 718);
    map.insert("CBOT_ZW1!", 719);
    map.insert("CME_HE1!", 720);
    map.insert("CME_LE1!", 721);
    map.insert("COMEX_HG1!", 722);
    map.insert("EUREX_FGBL1!", 723);
    map.insert("EUREX_FGBM1!", 724);
    map.insert("EUREX_FGBS1!", 725);
    map.insert("ICEEUR_I1!", 726);
    map.insert("ICEUS_CC1!", 727);
    map.insert("ICEUS_CT1!", 728);
    map.insert("ICEUS_KC1!", 729);
    map.insert("ICEUS_SB1!", 730);
    map.insert("NYMEX_HO1!", 731);
    map.insert("NYMEX_NG1!", 732);
    map.insert("NYMEX_PL1!", 733);
    map.insert("NYMEX_RB1!", 734);
    map.insert("NASDAQ_LPLA", 735);
    map.insert("NYSE_TRGP", 736);
    map.insert("NYSE_CPAY", 737);
    map.insert("NYSE_BAH", 738);
    map.insert("NYSE_GM", 739);
    map.insert("NASDAQ_TSLA", 740);
    map.insert("NYSE_BERY", 740);
    map.insert("NYSE_TWLO", 741);
    map.insert("NASDAQ_CDW", 742);
    map.insert("NYSE_CHGG", 743);
    map.insert("NASDAQ_PANW", 744);
    map.insert("NYSE_HWM", 745);
    map.insert("NYSE_AL", 746);
    map.insert("NYSE_POST", 747);
    map.insert("CBOT_TN1!", 748);
    map.insert("AMEX_GSLC", 749);
    map.insert("NYSE_VAC", 750);
    map.insert("NYSE_UBER", 751);
    map.insert("NYSE_EQH", 752);
    map.insert("NYSE_ESTC", 753);
    map.insert("NYSE_PINS", 754);
    map.insert("NYSE_ENOV", 755);
    map.insert("NYSE_ESI", 756);
    map.insert("NYSE_DT", 757);
    map.insert("NYSE_PAYC", 758);
    map.insert("NASDAQ_CHX", 759);
    map.insert("AMEX_SCHD", 760);
    map.insert("NASDAQ_SAIC", 761);
    map.insert("NYSE_XYL", 762);
    map.insert("NASDAQ_ENPH", 763);
    map.insert("CBOE_INDA", 764);
    map.insert("NYSE_KEYS", 765);
    map.insert("NASDAQ_DBX", 766);
    map.insert("NYSE_IR", 767);
    map.insert("NYSE_GDDY", 768);
    map.insert("NYSE_LEVI", 769);
    map.insert("NASDAQ_GH", 770);
    map.insert("NYSE_CHWY", 771);
    map.insert("NYSE_HUBS", 772);
    map.insert("NYSE_HII", 773);
    map.insert("NASDAQ_CG", 774);
    map.insert("NYSE_ALLY", 775);
    map.insert("NYSE_ANET", 776);
    map.insert("NASDAQ_PYPL", 777);
    map.insert("INDEX_ETHUSD", 778);
    map.insert("NYSE_GWRE", 779);
    map.insert("NYSE_SYF", 780);
    map.insert("NYSE_CABO", 781);
    map.insert("NYSE_NET", 782);
    map.insert("NASDAQ_ZI", 783);
    map.insert("NASDAQ_QRVO", 784);
    map.insert("NASDAQ_NTNX", 785);
    map.insert("NASDAQ_ESGU", 786);
    map.insert("NASDAQ_LYFT", 787);
    map.insert("NYSE_WH", 788);
    map.insert("CBOE_EFAV", 789);
    map.insert("AMEX_ARKG", 790);
    map.insert("NYSE_LW", 791);
    map.insert("NYSE_W", 792);
    map.insert("AMEX_ARKW", 793);
    map.insert("NASDAQ_ACHC", 794);
    map.insert("NASDAQ_DNLI", 795);
    map.insert("NYSE_FBIN", 796);
    map.insert("NASDAQ_RARE", 797);
    map.insert("NYSE_VOYA", 798);
    map.insert("NASDAQ_ZS", 799);
    map.insert("NASDAQ_ZM", 800);
    map.insert("NYSE_PSTG", 801);
    map.insert("NASDAQ_ZG", 802);
    map.insert("NYSE_ARES", 803);
    map.insert("NYSE_CARR", 804);
    map.insert("NASDAQ_SKYY", 805);
    map.insert("NYSE_ZTS", 806);
    map.insert("CBOE_MTUM", 807);
    map.insert("NYSE_SMAR", 808);
    map.insert("NASDAQ_FOXA", 809);
    map.insert("NASDAQ_VIR", 810);
    map.insert("NASDAQ_META", 811);
    map.insert("NYSE_CFG", 812);
    map.insert("NYSE_TRU", 813);
    map.insert("NYSE_SITE", 814);
    map.insert("NYSE_GMED", 815);
    map.insert("NASDAQ_MDB", 816);
    map.insert("NYSE_BURL", 817);
    map.insert("NYSE_COTY", 818);
    map.insert("NASDAQ_TNDM", 819);
    map.insert("NASDAQ_BPMC", 820);
    map.insert("NASDAQ_FIVN", 821);
    map.insert("NYSE_NVST", 822);
    map.insert("NASDAQ_RRR", 823);
    map.insert("NYSE_HCA", 824);
    map.insert("NYSE_AVTR", 825);
    map.insert("NYSE_CC", 826);
    map.insert("NASDAQ_FOXF", 827);
    map.insert("NASDAQ_APLS", 828);
    map.insert("NASDAQ_TTD", 829);
    map.insert("NYSE_ABBV", 830);
    map.insert("NYSE_PEN", 831);
    map.insert("NASDAQ_FANG", 832);
    map.insert("NYSE_BJ", 833);
    map.insert("NYSE_BILL", 834);
    map.insert("NYSE_WK", 835);
    map.insert("NASDAQ_PTON", 836);
    map.insert("NASDAQ_VXUS", 837);
    map.insert("NYSE_MPC", 838);
    map.insert("NASDAQ_COIN", 839);
    map.insert("NASDAQ_OKTA", 840);
    map.insert("NYSE_NCLH", 841);
    map.insert("NASDAQ_FRPT", 842);
    map.insert("NYSE_CTLT", 843);
    map.insert("NYSE_YETI", 844);
    map.insert("NYSE_OMF", 845);
    map.insert("NASDAQ_VIRT", 846);
    map.insert("NYSE_ELAN", 847);
    map.insert("NYSE_WMS", 848);
    map.insert("CBOE_VLUE", 849);
    map.insert("AMEX_XLC", 850);
    map.insert("NASDAQ_PCTY", 851);
    map.insert("NYSE_BFAM", 852);
    map.insert("NYSE_BLD", 853);
    map.insert("NYSE_EPAM", 854);
    map.insert("NYSE_IQV", 855);
    map.insert("NYSE_RNG", 856);
    map.insert("NYSE_OTIS", 857);
    map.insert("NYSE_DELL", 858);
    map.insert("NYSE_VVV", 859);
    map.insert("NYSE_KMI", 860);
    map.insert("NASDAQ_RUN", 861);
    map.insert("NASDAQ_CRWD", 862);
    map.insert("NASDAQ_VRNS", 863);
    map.insert("NASDAQ_NTLA", 864);
    map.insert("NASDAQ_DOCU", 865);
    map.insert("NYSE_ZWS", 866);
    map.insert("NASDAQ_MRNA", 867);
    map.insert("NASDAQ_LITE", 868);
    map.insert("NYSE_RH", 869);
    map.insert("AMEX_ARKK", 870);
    map.insert("NASDAQ_MEDP", 871);
    map.insert("NASDAQ_ROKU", 872);
    map.insert("CBOE_USMV", 873);
    map.insert("NYSE_AXTA", 874);
    map.insert("NYSE_CTVA", 875);
    map.insert("NASDAQ_KHC", 876);
    map.insert("NYSE_VST", 877);
    map.insert("NASDAQ_WDAY", 878);
    map.insert("NYSE_SQ", 879);
    map.insert("NYSE_DXC", 880);
    map.insert("AMEX_SPLV", 881);
    map.insert("NYSE_ESNT", 882);
    map.insert("NYSE_ARMK", 883);
    map.insert("NYSE_NOW", 884);
    map.insert("NYSE_HPE", 885);
    map.insert("NASDAQ_BL", 886);
    map.insert("NYSE_FND", 887);
    map.insert("AMEX_DGRO", 888);
    map.insert("NASDAQ_DDOG", 889);
    map.insert("NASDAQ_FIVE", 890);
    map.insert("NASDAQ_GOOG", 891);
    map.insert("NYSE_DOW", 892);
    map.insert("NYSE_FTV", 893);
    map.insert("NYSE_DAY", 894);
    map.insert("NASDAQ_MCHI", 895);
    map.insert("NYSE_SNAP", 896);
    map.insert("NASDAQ_PGNY", 897);
    map.insert("NYSE_TDOC", 898);
    map.insert("NASDAQ_HQY", 899);
    map.insert("NASDAQ_TXG", 900);
    map.insert("NASDAQ_TRIP", 901);
    map.insert("NASDAQ_FOX", 902);
    map.insert("NYSE_QTWO", 903);
    map.insert("NASDAQ_ETSY", 904);
    map.insert("NYSE_USFD", 905);
    map.insert("AMEX_HDV", 906);
    map.insert("NASDAQ_NWSA", 907);
    map.insert("NYSE_PLNT", 908);
    map.insert("NYSE_VEEV", 909);
    map.insert("CBOE_QUAL", 910);
    map.insert("AMEX_FTEC", 911);
    map.insert("NASDAQ_OLLI", 912);
    map.insert("NYSE_INSP", 913);
    map.insert("NYSE_CVNA", 914);
    map.insert("NYSE_HLT", 915);
    map.insert("NASDAQ_LAZR", 916);
    map.insert("NYSE_PFGC", 917);
    map.insert("NASDAQ_EXPI", 918);
    map
}

pub enum Fee {
    Price(f32),
    OrderValue(f32),
    // commission / (price * contract_size)
    Contract(f32),
}

pub const FEES: [Fee; N_MARKETS] = [
    Fee::Price(0.02),            // "AMEX_DIA",
    Fee::Price(0.02),            // "AMEX_EEM",
    Fee::Price(0.02),            // "AMEX_EFA",
    Fee::Price(0.02),            // "AMEX_EWJ",
    Fee::Price(0.02),            // "AMEX_EWT",
    Fee::Price(0.02),            // "AMEX_EWY",
    Fee::Price(0.02),            // "AMEX_EWZ",
    Fee::Price(0.02),            // "AMEX_FDN",
    Fee::Price(0.02),            // "AMEX_FVD",
    Fee::Price(0.02),            // "AMEX_GDX",
    Fee::Price(0.02),            // "AMEX_GDXJ",
    Fee::Price(0.02),            // "AMEX_GLD",
    Fee::Price(0.02),            // "AMEX_IHI",
    Fee::Price(0.02),            // "AMEX_IJH",
    Fee::Price(0.02),            // "AMEX_IJR",
    Fee::Price(0.02),            // "AMEX_IJS",
    Fee::Price(0.02),            // "AMEX_ITOT",
    Fee::Price(0.02),            // "AMEX_IVE",
    Fee::Price(0.02),            // "AMEX_IVW",
    Fee::Price(0.02),            // "AMEX_IWB",
    Fee::Price(0.02),            // "AMEX_IWD",
    Fee::Price(0.02),            // "AMEX_IWF",
    Fee::Price(0.02),            // "AMEX_IWM",
    Fee::Price(0.02),            // "AMEX_IWN",
    Fee::Price(0.02),            // "AMEX_IWO",
    Fee::Price(0.02),            // "AMEX_IWS",
    Fee::Price(0.02),            // "AMEX_IYR",
    Fee::Price(0.02),            // "AMEX_IYW",
    Fee::Price(0.02),            // "AMEX_KRE",
    Fee::Price(0.02),            // "AMEX_MDY",
    Fee::Price(0.02),            // "AMEX_OEF",
    Fee::Price(0.02),            // "AMEX_RSP",
    Fee::Price(0.02),            // "AMEX_SCHB",
    Fee::Price(0.02),            // "AMEX_SCHF",
    Fee::Price(0.02),            // "AMEX_SCHG",
    Fee::Price(0.02),            // "AMEX_SCHV",
    Fee::Price(0.02),            // "AMEX_SLV",
    Fee::Price(0.02),            // "AMEX_SPDW",
    Fee::Price(0.02),            // "AMEX_SPY",
    Fee::Price(0.02),            // "AMEX_SPYG",
    Fee::Price(0.02),            // "AMEX_SPYV",
    Fee::Price(0.02),            // "AMEX_TIP",
    Fee::Price(0.02),            // "AMEX_VDE",
    Fee::Price(0.02),            // "AMEX_VEA",
    Fee::Price(0.02),            // "AMEX_VGT",
    Fee::Price(0.02),            // "AMEX_VHT",
    Fee::Price(0.02),            // "AMEX_VNQ",
    Fee::Price(0.02),            // "AMEX_VOE",
    Fee::Price(0.02),            // "AMEX_VPL",
    Fee::Price(0.02),            // "AMEX_VT",
    Fee::Price(0.02),            // "AMEX_VTI",
    Fee::Price(0.02),            // "AMEX_VWO",
    Fee::Price(0.02),            // "AMEX_VXF",
    Fee::Price(0.02),            // "AMEX_XBI",
    Fee::Price(0.02),            // "AMEX_XLB",
    Fee::Price(0.02),            // "AMEX_XLE",
    Fee::Price(0.02),            // "AMEX_XLF",
    Fee::Price(0.02),            // "AMEX_XLI",
    Fee::Price(0.02),            // "AMEX_XLK",
    Fee::Price(0.02),            // "AMEX_XLP",
    Fee::Price(0.02),            // "AMEX_XLU",
    Fee::Price(0.02),            // "AMEX_XLV",
    Fee::Price(0.02),            // "AMEX_XLY",
    Fee::Contract(3. / 50.),     // "CAPITALCOM_RTY",
    Fee::Price(0.02),            // "CBOE_EFG",
    Fee::Price(0.02),            // "CBOE_EFV",
    Fee::Price(0.02),            // "CBOE_EZU",
    Fee::Price(0.02),            // "CBOE_IGV",
    Fee::OrderValue(0.000025),   // "FX_AUDCAD",
    Fee::OrderValue(0.000025),   // "FX_AUDCHF",
    Fee::OrderValue(0.000025),   // "FX_AUDJPY",
    Fee::OrderValue(0.000025),   // "FX_AUDNZD",
    Fee::OrderValue(0.000025),   // "FX_AUDUSD",
    Fee::OrderValue(0.000025),   // "FX_AUS200",
    Fee::OrderValue(0.000025),   // "FX_CADCHF",
    Fee::OrderValue(0.000025),   // "FX_CADJPY",
    Fee::OrderValue(0.000025),   // "FX_CHFJPY",
    Fee::OrderValue(0.000025),   // "FX_EURAUD",
    Fee::OrderValue(0.000025),   // "FX_EURCAD",
    Fee::OrderValue(0.000025),   // "FX_EURCHF",
    Fee::OrderValue(0.000025),   // "FX_EURGBP",
    Fee::OrderValue(0.000025),   // "FX_EURJPY",
    Fee::OrderValue(0.000025),   // "FX_EURNZD",
    Fee::OrderValue(0.000025),   // "FX_EURUSD",
    Fee::OrderValue(0.000025),   // "FX_GBPAUD",
    Fee::OrderValue(0.000025),   // "FX_GBPCAD",
    Fee::OrderValue(0.000025),   // "FX_GBPCHF",
    Fee::OrderValue(0.000025),   // "FX_GBPJPY",
    Fee::OrderValue(0.000025),   // "FX_GBPNZD",
    Fee::OrderValue(0.000025),   // "FX_GBPUSD",
    Fee::OrderValue(0.000025),   // "FX_IDC_CADUSD",
    Fee::OrderValue(0.000025),   // "FX_IDC_CHFUSD",
    Fee::OrderValue(0.000025),   // "FX_IDC_EURMXN",
    Fee::OrderValue(0.000025),   // "FX_IDC_EURNOK",
    Fee::OrderValue(0.000025),   // "FX_IDC_EURSEK",
    Fee::OrderValue(0.000025),   // "FX_IDC_GBPMXN",
    Fee::OrderValue(0.000025),   // "FX_IDC_GBPNOK",
    Fee::OrderValue(0.000025),   // "FX_IDC_GBPSEK",
    Fee::OrderValue(0.000025),   // "FX_IDC_JPYUSD",
    Fee::OrderValue(0.000025),   // "FX_IDC_USDNOK",
    Fee::OrderValue(0.000025),   // "FX_IDC_USDSGD",
    Fee::OrderValue(0.000025),   // "FX_NZDCAD",
    Fee::OrderValue(0.000025),   // "FX_NZDCHF",
    Fee::OrderValue(0.000025),   // "FX_NZDJPY",
    Fee::OrderValue(0.000025),   // "FX_NZDUSD",
    Fee::OrderValue(0.000025),   // "FX_UK100",
    Fee::OrderValue(0.000025),   // "FX_USDCAD",
    Fee::OrderValue(0.000025),   // "FX_USDCHF",
    Fee::OrderValue(0.000025),   // "FX_USDJPY",
    Fee::OrderValue(0.000025),   // "FX_USDMXN",
    Fee::OrderValue(0.000025),   // "FX_USDSEK",
    Fee::Price(0.02),            // "NASDAQ_AAL",
    Fee::Price(0.02),            // "NASDAQ_AAPL",
    Fee::Price(0.02),            // "NASDAQ_AAXJ",
    Fee::Price(0.02),            // "NASDAQ_ADBE",
    Fee::Price(0.02),            // "NASDAQ_ADI",
    Fee::Price(0.02),            // "NASDAQ_ADP",
    Fee::Price(0.02),            // "NASDAQ_ADSK",
    Fee::Price(0.02),            // "NASDAQ_AEP",
    Fee::Price(0.02),            // "NASDAQ_AKAM",
    Fee::Price(0.02),            // "NASDAQ_ALGN",
    Fee::Price(0.02),            // "NASDAQ_ALNY",
    Fee::Price(0.02),            // "NASDAQ_AMAT",
    Fee::Price(0.02),            // "NASDAQ_AMD",
    Fee::Price(0.02),            // "NASDAQ_AMED",
    Fee::Price(0.02),            // "NASDAQ_AMGN",
    Fee::Price(0.02),            // "NASDAQ_AMKR",
    Fee::Price(0.02),            // "NASDAQ_AMZN",
    Fee::Price(0.02),            // "NASDAQ_ANSS",
    Fee::Price(0.02),            // "NASDAQ_APA",
    Fee::Price(0.02),            // "NASDAQ_ARCC",
    Fee::Price(0.02),            // "NASDAQ_ARWR",
    Fee::Price(0.02),            // "NASDAQ_AVGO",
    Fee::Price(0.02),            // "NASDAQ_AXON",
    Fee::Price(0.02),            // "NASDAQ_AZPN",
    Fee::Price(0.02),            // "NASDAQ_AZTA",
    Fee::Price(0.02),            // "NASDAQ_BIIB",
    Fee::Price(0.02),            // "NASDAQ_BKNG",
    Fee::Price(0.02),            // "NASDAQ_BKR",
    Fee::Price(0.02),            // "NASDAQ_BMRN",
    Fee::Price(0.02),            // "NASDAQ_BRKR",
    Fee::Price(0.02),            // "NASDAQ_CACC",
    Fee::Price(0.02),            // "NASDAQ_CAR",
    Fee::Price(0.02),            // "NASDAQ_CASY",
    Fee::Price(0.02),            // "NASDAQ_CDNS",
    Fee::Price(0.02),            // "NASDAQ_CGNX",
    Fee::Price(0.02),            // "NASDAQ_CHDN",
    Fee::Price(0.02),            // "NASDAQ_CHRW",
    Fee::Price(0.02),            // "NASDAQ_CHTR",
    Fee::Price(0.02),            // "NASDAQ_CINF",
    Fee::Price(0.02),            // "NASDAQ_CMCSA",
    Fee::Price(0.02),            // "NASDAQ_CME",
    Fee::Price(0.02),            // "NASDAQ_COLM",
    Fee::Price(0.02),            // "NASDAQ_COO",
    Fee::Price(0.02),            // "NASDAQ_COST",
    Fee::Price(0.02),            // "NASDAQ_CPRT",
    Fee::Price(0.02),            // "NASDAQ_CROX",
    Fee::Price(0.02),            // "NASDAQ_CSCO",
    Fee::Price(0.02),            // "NASDAQ_CSGP",
    Fee::Price(0.02),            // "NASDAQ_CSX",
    Fee::Price(0.02),            // "NASDAQ_CTAS",
    Fee::Price(0.02),            // "NASDAQ_CTSH",
    Fee::Price(0.02),            // "NASDAQ_DLTR",
    Fee::Price(0.02),            // "NASDAQ_DOX",
    Fee::Price(0.02),            // "NASDAQ_DVY",
    Fee::Price(0.02),            // "NASDAQ_DXCM",
    Fee::Price(0.02),            // "NASDAQ_EA",
    Fee::Price(0.02),            // "NASDAQ_EBAY",
    Fee::Price(0.02),            // "NASDAQ_EEFT",
    Fee::Price(0.02),            // "NASDAQ_EMB",
    Fee::Price(0.02),            // "NASDAQ_ENTG",
    Fee::Price(0.02),            // "NASDAQ_EVRG",
    Fee::Price(0.02),            // "NASDAQ_EWBC",
    Fee::Price(0.02),            // "NASDAQ_EXC",
    Fee::Price(0.02),            // "NASDAQ_EXEL",
    Fee::Price(0.02),            // "NASDAQ_EXPE",
    Fee::Price(0.02),            // "NASDAQ_FAST",
    Fee::Price(0.02),            // "NASDAQ_FFIV",
    Fee::Price(0.02),            // "NASDAQ_FITB",
    Fee::Price(0.02),            // "NASDAQ_FLEX",
    Fee::Price(0.02),            // "NASDAQ_FSLR",
    Fee::Price(0.02),            // "NASDAQ_FTNT",
    Fee::Price(0.02),            // "NASDAQ_GEN",
    Fee::Price(0.02),            // "NASDAQ_GILD",
    Fee::Price(0.02),            // "NASDAQ_GNTX",
    Fee::Price(0.02),            // "NASDAQ_GOOGL",
    Fee::Price(0.02),            // "NASDAQ_GT",
    Fee::Price(0.02),            // "NASDAQ_HALO",
    Fee::Price(0.02),            // "NASDAQ_HAS",
    Fee::Price(0.02),            // "NASDAQ_HBAN",
    Fee::Price(0.02),            // "NASDAQ_HELE",
    Fee::Price(0.02),            // "NASDAQ_HOLX",
    Fee::Price(0.02),            // "NASDAQ_HON",
    Fee::Price(0.02),            // "NASDAQ_HSIC",
    Fee::Price(0.02),            // "NASDAQ_IBB",
    Fee::Price(0.02),            // "NASDAQ_IBKR",
    Fee::Price(0.02),            // "NASDAQ_ICLN",
    Fee::Price(0.02),            // "NASDAQ_IDXX",
    Fee::Price(0.02),            // "NASDAQ_ILMN",
    Fee::Price(0.02),            // "NASDAQ_INCY",
    Fee::Price(0.02),            // "NASDAQ_INTC",
    Fee::Price(0.02),            // "NASDAQ_INTU",
    Fee::Price(0.02),            // "NASDAQ_IONS",
    Fee::Price(0.02),            // "NASDAQ_IPGP",
    Fee::Price(0.02),            // "NASDAQ_IRDM",
    Fee::Price(0.02),            // "NASDAQ_ISRG",
    Fee::Price(0.02),            // "NASDAQ_IUSG",
    Fee::Price(0.02),            // "NASDAQ_IUSV",
    Fee::Price(0.02),            // "NASDAQ_JBHT",
    Fee::Price(0.02),            // "NASDAQ_JBLU",
    Fee::Price(0.02),            // "NASDAQ_JKHY",
    Fee::Price(0.02),            // "NASDAQ_KDP",
    Fee::Price(0.02),            // "NASDAQ_KLAC",
    Fee::Price(0.02),            // "NASDAQ_LKQ",
    Fee::Price(0.02),            // "NASDAQ_LNT",
    Fee::Price(0.02),            // "NASDAQ_LNW",
    Fee::Price(0.02),            // "NASDAQ_LRCX",
    Fee::Price(0.02),            // "NASDAQ_LSCC",
    Fee::Price(0.02),            // "NASDAQ_LSTR",
    Fee::Price(0.02),            // "NASDAQ_MANH",
    Fee::Price(0.02),            // "NASDAQ_MAR",
    Fee::Price(0.02),            // "NASDAQ_MASI",
    Fee::Price(0.02),            // "NASDAQ_MAT",
    Fee::Price(0.02),            // "NASDAQ_MCHP",
    Fee::Price(0.02),            // "NASDAQ_MDLZ",
    Fee::Price(0.02),            // "NASDAQ_MIDD",
    Fee::Price(0.02),            // "NASDAQ_MKSI",
    Fee::Price(0.02),            // "NASDAQ_MKTX",
    Fee::Price(0.02),            // "NASDAQ_MNST",
    Fee::Price(0.02),            // "NASDAQ_MPWR",
    Fee::Price(0.02),            // "NASDAQ_MRVL",
    Fee::Price(0.02),            // "NASDAQ_MSFT",
    Fee::Price(0.02),            // "NASDAQ_MSTR",
    Fee::Price(0.02),            // "NASDAQ_MU",
    Fee::Price(0.02),            // "NASDAQ_NBIX",
    Fee::Price(0.02),            // "NASDAQ_NDAQ",
    Fee::Price(0.02),            // "NASDAQ_NDSN",
    Fee::Price(0.02),            // "NASDAQ_NDX",
    Fee::Price(0.02),            // "NASDAQ_NFLX",
    Fee::Price(0.02),            // "NASDAQ_NTAP",
    Fee::Price(0.02),            // "NASDAQ_NTRS",
    Fee::Price(0.02),            // "NASDAQ_NVDA",
    Fee::Price(0.02),            // "NASDAQ_NWL",
    Fee::Price(0.02),            // "NASDAQ_NXST",
    Fee::Price(0.02),            // "NASDAQ_ODFL",
    Fee::Price(0.02),            // "NASDAQ_OLED",
    Fee::Price(0.02),            // "NASDAQ_OMCL",
    Fee::Price(0.02),            // "NASDAQ_ON",
    Fee::Price(0.02),            // "NASDAQ_ORLY",
    Fee::Price(0.02),            // "NASDAQ_OZK",
    Fee::Price(0.02),            // "NASDAQ_PARA",
    Fee::Price(0.02),            // "NASDAQ_PAYX",
    Fee::Price(0.02),            // "NASDAQ_PCAR",
    Fee::Price(0.02),            // "NASDAQ_PEGA",
    Fee::Price(0.02),            // "NASDAQ_PENN",
    Fee::Price(0.02),            // "NASDAQ_PEP",
    Fee::Price(0.02),            // "NASDAQ_PFF",
    Fee::Price(0.02),            // "NASDAQ_PFG",
    Fee::Price(0.02),            // "NASDAQ_PNFP",
    Fee::Price(0.02),            // "NASDAQ_PODD",
    Fee::Price(0.02),            // "NASDAQ_POOL",
    Fee::Price(0.02),            // "NASDAQ_PTC",
    Fee::Price(0.02),            // "NASDAQ_QCOM",
    Fee::Price(0.02),            // "NASDAQ_QQQ",
    Fee::Price(0.02),            // "NASDAQ_REGN",
    Fee::Price(0.02),            // "NASDAQ_RGEN",
    Fee::Price(0.02),            // "NASDAQ_RGLD",
    Fee::Price(0.02),            // "NASDAQ_ROP",
    Fee::Price(0.02),            // "NASDAQ_ROST",
    Fee::Price(0.02),            // "NASDAQ_SAIA",
    Fee::Price(0.02),            // "NASDAQ_SBUX",
    Fee::Price(0.02),            // "NASDAQ_SCZ",
    Fee::Price(0.02),            // "NASDAQ_SLAB",
    Fee::Price(0.02),            // "NASDAQ_SLM",
    Fee::Price(0.02),            // "NASDAQ_SMH",
    Fee::Price(0.02),            // "NASDAQ_SNPS",
    Fee::Price(0.02),            // "NASDAQ_SOXX",
    Fee::Price(0.02),            // "NASDAQ_SRPT",
    Fee::Price(0.02),            // "NASDAQ_SSNC",
    Fee::Price(0.02),            // "NASDAQ_STLD",
    Fee::Price(0.02),            // "NASDAQ_SWKS",
    Fee::Price(0.02),            // "NASDAQ_SYNA",
    Fee::Price(0.02),            // "NASDAQ_TECH",
    Fee::Price(0.02),            // "NASDAQ_TER",
    Fee::Price(0.02),            // "NASDAQ_TLT",
    Fee::Price(0.02),            // "NASDAQ_TMUS",
    Fee::Price(0.02),            // "NASDAQ_TRMB",
    Fee::Price(0.02),            // "NASDAQ_TROW",
    Fee::Price(0.02),            // "NASDAQ_TSCO",
    Fee::Price(0.02),            // "NASDAQ_TTEK",
    Fee::Price(0.02),            // "NASDAQ_TTWO",
    Fee::Price(0.02),            // "NASDAQ_TXN",
    Fee::Price(0.02),            // "NASDAQ_TXRH",
    Fee::Price(0.02),            // "NASDAQ_UAL",
    Fee::Price(0.02),            // "NASDAQ_ULTA",
    Fee::Price(0.02),            // "NASDAQ_UTHR",
    Fee::Price(0.02),            // "NASDAQ_VRSK",
    Fee::Price(0.02),            // "NASDAQ_VRSN",
    Fee::Price(0.02),            // "NASDAQ_VRTX",
    Fee::Price(0.02),            // "NASDAQ_VTRS",
    Fee::Price(0.02),            // "NASDAQ_WBA",
    Fee::Price(0.02),            // "NASDAQ_WDC",
    Fee::Price(0.02),            // "NASDAQ_WWD",
    Fee::Price(0.02),            // "NASDAQ_WYNN",
    Fee::Price(0.02),            // "NASDAQ_XEL",
    Fee::Price(0.02),            // "NASDAQ_XRAY",
    Fee::Price(0.02),            // "NASDAQ_ZBRA",
    Fee::Price(0.02),            // "NASDAQ_ZD",
    Fee::Price(0.02),            // "NASDAQ_ZION",
    Fee::Price(0.02),            // "NYSE_A",
    Fee::Price(0.02),            // "NYSE_AA",
    Fee::Price(0.02),            // "NYSE_AAP",
    Fee::Price(0.02),            // "NYSE_ABT",
    Fee::Price(0.02),            // "NYSE_ACM",
    Fee::Price(0.02),            // "NYSE_ACN",
    Fee::Price(0.02),            // "NYSE_ADM",
    Fee::Price(0.02),            // "NYSE_AES",
    Fee::Price(0.02),            // "NYSE_AFG",
    Fee::Price(0.02),            // "NYSE_AFL",
    Fee::Price(0.02),            // "NYSE_AGCO",
    Fee::Price(0.02),            // "NYSE_AIG",
    Fee::Price(0.02),            // "NYSE_AIZ",
    Fee::Price(0.02),            // "NYSE_AJG",
    Fee::Price(0.02),            // "NYSE_ALB",
    Fee::Price(0.02),            // "NYSE_ALK",
    Fee::Price(0.02),            // "NYSE_ALL",
    Fee::Price(0.02),            // "NYSE_AME",
    Fee::Price(0.02),            // "NYSE_AMG",
    Fee::Price(0.02),            // "NYSE_AMP",
    Fee::Price(0.02),            // "NYSE_AMT",
    Fee::Price(0.02),            // "NYSE_AN",
    Fee::Price(0.02),            // "NYSE_AON",
    Fee::Price(0.02),            // "NYSE_AOS",
    Fee::Price(0.02),            // "NYSE_APD",
    Fee::Price(0.02),            // "NYSE_APH",
    Fee::Price(0.02),            // "NYSE_ARW",
    Fee::Price(0.02),            // "NYSE_ASH",
    Fee::Price(0.02),            // "NYSE_AVY",
    Fee::Price(0.02),            // "NYSE_AWI",
    Fee::Price(0.02),            // "NYSE_AWK",
    Fee::Price(0.02),            // "NYSE_AXP",
    Fee::Price(0.02),            // "NYSE_AYI",
    Fee::Price(0.02),            // "NYSE_AZO",
    Fee::Price(0.02),            // "NYSE_BA",
    Fee::Price(0.02),            // "NYSE_BAC",
    Fee::Price(0.02),            // "NYSE_BALL",
    Fee::Price(0.02),            // "NYSE_BAX",
    Fee::Price(0.02),            // "NYSE_BBY",
    Fee::Price(0.02),            // "NYSE_BC",
    Fee::Price(0.02),            // "NYSE_BDX",
    Fee::Price(0.02),            // "NYSE_BEN",
    Fee::Price(0.02),            // "NYSE_BIO",
    Fee::Price(0.02),            // "NYSE_BK",
    Fee::Price(0.02),            // "NYSE_BLDR",
    Fee::Price(0.02),            // "NYSE_BLK",
    Fee::Price(0.02),            // "NYSE_BMY",
    Fee::Price(0.02),            // "NYSE_BR",
    Fee::Price(0.02),            // "NYSE_BRK.A",
    Fee::Price(0.02),            // "NYSE_BRO",
    Fee::Price(0.02),            // "NYSE_BSX",
    Fee::Price(0.02),            // "NYSE_BWA",
    Fee::Price(0.02),            // "NYSE_BX",
    Fee::Price(0.02),            // "NYSE_BYD",
    Fee::Price(0.02),            // "NYSE_C",
    Fee::Price(0.02),            // "NYSE_CACI",
    Fee::Price(0.02),            // "NYSE_CAG",
    Fee::Price(0.02),            // "NYSE_CAH",
    Fee::Price(0.02),            // "NYSE_CAT",
    Fee::Price(0.02),            // "NYSE_CB",
    Fee::Price(0.02),            // "NYSE_CBRE",
    Fee::Price(0.02),            // "NYSE_CCI",
    Fee::Price(0.02),            // "NYSE_CCK",
    Fee::Price(0.02),            // "NYSE_CCL",
    Fee::Price(0.02),            // "NYSE_CE",
    Fee::Price(0.02),            // "NYSE_CF",
    Fee::Price(0.02),            // "NYSE_CFR",
    Fee::Price(0.02),            // "NYSE_CHD",
    Fee::Price(0.02),            // "NYSE_CHE",
    Fee::Price(0.02),            // "NYSE_CI",
    Fee::Price(0.02),            // "NYSE_CIEN",
    Fee::Price(0.02),            // "NYSE_CL",
    Fee::Price(0.02),            // "NYSE_CLF",
    Fee::Price(0.02),            // "NYSE_CLX",
    Fee::Price(0.02),            // "NYSE_CMA",
    Fee::Price(0.02),            // "NYSE_CMG",
    Fee::Price(0.02),            // "NYSE_CMI",
    Fee::Price(0.02),            // "NYSE_CMS",
    Fee::Price(0.02),            // "NYSE_CNC",
    Fee::Price(0.02),            // "NYSE_CNP",
    Fee::Price(0.02),            // "NYSE_COF",
    Fee::Price(0.02),            // "NYSE_COHR",
    Fee::Price(0.02),            // "NYSE_COP",
    Fee::Price(0.02),            // "NYSE_COR",
    Fee::Price(0.02),            // "NYSE_CRL",
    Fee::Price(0.02),            // "NYSE_CRM",
    Fee::Price(0.02),            // "NYSE_CSL",
    Fee::Price(0.02),            // "NYSE_CVS",
    Fee::Price(0.02),            // "NYSE_CVX",
    Fee::Price(0.02),            // "NYSE_D",
    Fee::Price(0.02),            // "NYSE_DAL",
    Fee::Price(0.02),            // "NYSE_DAR",
    Fee::Price(0.02),            // "NYSE_DD",
    Fee::Price(0.02),            // "NYSE_DE",
    Fee::Price(0.02),            // "NYSE_DECK",
    Fee::Price(0.02),            // "NYSE_DFS",
    Fee::Price(0.02),            // "NYSE_DG",
    Fee::Price(0.02),            // "NYSE_DGX",
    Fee::Price(0.02),            // "NYSE_DHI",
    Fee::Price(0.02),            // "NYSE_DHR",
    Fee::Price(0.02),            // "NYSE_DIS",
    Fee::Price(0.02),            // "NYSE_DKS",
    Fee::Price(0.02),            // "NYSE_DLB",
    Fee::Price(0.02),            // "NYSE_DOV",
    Fee::Price(0.02),            // "NYSE_DPZ",
    Fee::Price(0.02),            // "NYSE_DRI",
    Fee::Price(0.02),            // "NYSE_DTE",
    Fee::Price(0.02),            // "NYSE_DUK",
    Fee::Price(0.02),            // "NYSE_DVA",
    Fee::Price(0.02),            // "NYSE_DVN",
    Fee::Price(0.02),            // "NYSE_ECL",
    Fee::Price(0.02),            // "NYSE_EFX",
    Fee::Price(0.02),            // "NYSE_EHC",
    Fee::Price(0.02),            // "NYSE_EIX",
    Fee::Price(0.02),            // "NYSE_EL",
    Fee::Price(0.02),            // "NYSE_ELV",
    Fee::Price(0.02),            // "NYSE_EME",
    Fee::Price(0.02),            // "NYSE_EMN",
    Fee::Price(0.02),            // "NYSE_EMR",
    Fee::Price(0.02),            // "NYSE_EOG",
    Fee::Price(0.02),            // "NYSE_EQT",
    Fee::Price(0.02),            // "NYSE_ES",
    Fee::Price(0.02),            // "NYSE_ETN",
    Fee::Price(0.02),            // "NYSE_ETR",
    Fee::Price(0.02),            // "NYSE_EVR",
    Fee::Price(0.02),            // "NYSE_EW",
    Fee::Price(0.02),            // "NYSE_EXP",
    Fee::Price(0.02),            // "NYSE_EXPD",
    Fee::Price(0.02),            // "NYSE_F",
    Fee::Price(0.02),            // "NYSE_FAF",
    Fee::Price(0.02),            // "NYSE_FCX",
    Fee::Price(0.02),            // "NYSE_FDS",
    Fee::Price(0.02),            // "NYSE_FDX",
    Fee::Price(0.02),            // "NYSE_FE",
    Fee::Price(0.02),            // "NYSE_FHN",
    Fee::Price(0.02),            // "NYSE_FI",
    Fee::Price(0.02),            // "NYSE_FICO",
    Fee::Price(0.02),            // "NYSE_FIS",
    Fee::Price(0.02),            // "NYSE_FLS",
    Fee::Price(0.02),            // "NYSE_FMC",
    Fee::Price(0.02),            // "NYSE_FNF",
    Fee::Price(0.02),            // "NYSE_G",
    Fee::Price(0.02),            // "NYSE_GAP",
    Fee::Price(0.02),            // "NYSE_GD",
    Fee::Price(0.02),            // "NYSE_GE",
    Fee::Price(0.02),            // "NYSE_GGG",
    Fee::Price(0.02),            // "NYSE_GIS",
    Fee::Price(0.02),            // "NYSE_GL",
    Fee::Price(0.02),            // "NYSE_GLW",
    Fee::Price(0.02),            // "NYSE_GNRC",
    Fee::Price(0.02),            // "NYSE_GPC",
    Fee::Price(0.02),            // "NYSE_GPK",
    Fee::Price(0.02),            // "NYSE_GPN",
    Fee::Price(0.02),            // "NYSE_GS",
    Fee::Price(0.02),            // "NYSE_GTLS",
    Fee::Price(0.02),            // "NYSE_GWW",
    Fee::Price(0.02),            // "NYSE_H",
    Fee::Price(0.02),            // "NYSE_HAL",
    Fee::Price(0.02),            // "NYSE_HBI",
    Fee::Price(0.02),            // "NYSE_HD",
    Fee::Price(0.02),            // "NYSE_HEI",
    Fee::Price(0.02),            // "NYSE_HES",
    Fee::Price(0.02),            // "NYSE_HIG",
    Fee::Price(0.02),            // "NYSE_HOG",
    Fee::Price(0.02),            // "NYSE_HPQ",
    Fee::Price(0.02),            // "NYSE_HUBB",
    Fee::Price(0.02),            // "NYSE_HUM",
    Fee::Price(0.02),            // "NYSE_HUN",
    Fee::Price(0.02),            // "NYSE_HXL",
    Fee::Price(0.02),            // "NYSE_IBM",
    Fee::Price(0.02),            // "NYSE_ICE",
    Fee::Price(0.02),            // "NYSE_IEX",
    Fee::Price(0.02),            // "NYSE_IFF",
    Fee::Price(0.02),            // "NYSE_IGT",
    Fee::Price(0.02),            // "NYSE_INGR",
    Fee::Price(0.02),            // "NYSE_IP",
    Fee::Price(0.02),            // "NYSE_IPG",
    Fee::Price(0.02),            // "NYSE_IT",
    Fee::Price(0.02),            // "NYSE_ITT",
    Fee::Price(0.02),            // "NYSE_ITW",
    Fee::Price(0.02),            // "NYSE_IVZ",
    Fee::Price(0.02),            // "NYSE_J",
    Fee::Price(0.02),            // "NYSE_JBL",
    Fee::Price(0.02),            // "NYSE_JCI",
    Fee::Price(0.02),            // "NYSE_JEF",
    Fee::Price(0.02),            // "NYSE_JLL",
    Fee::Price(0.02),            // "NYSE_JNJ",
    Fee::Price(0.02),            // "NYSE_JNPR",
    Fee::Price(0.02),            // "NYSE_JPM",
    Fee::Price(0.02),            // "NYSE_JWN",
    Fee::Price(0.02),            // "NYSE_K",
    Fee::Price(0.02),            // "NYSE_KBR",
    Fee::Price(0.02),            // "NYSE_KEY",
    Fee::Price(0.02),            // "NYSE_KKR",
    Fee::Price(0.02),            // "NYSE_KMB",
    Fee::Price(0.02),            // "NYSE_KMX",
    Fee::Price(0.02),            // "NYSE_KNX",
    Fee::Price(0.02),            // "NYSE_KO",
    Fee::Price(0.02),            // "NYSE_KR",
    Fee::Price(0.02),            // "NYSE_KSS",
    Fee::Price(0.02),            // "NYSE_L",
    Fee::Price(0.02),            // "NYSE_LAD",
    Fee::Price(0.02),            // "NYSE_LDOS",
    Fee::Price(0.02),            // "NYSE_LEA",
    Fee::Price(0.02),            // "NYSE_LEG",
    Fee::Price(0.02),            // "NYSE_LEN",
    Fee::Price(0.02),            // "NYSE_LH",
    Fee::Price(0.02),            // "NYSE_LHX",
    Fee::Price(0.02),            // "NYSE_LII",
    Fee::Price(0.02),            // "NYSE_LLY",
    Fee::Price(0.02),            // "NYSE_LMT",
    Fee::Price(0.02),            // "NYSE_LNC",
    Fee::Price(0.02),            // "NYSE_LOW",
    Fee::Price(0.02),            // "NYSE_LPX",
    Fee::Price(0.02),            // "NYSE_LUMN",
    Fee::Price(0.02),            // "NYSE_LUV",
    Fee::Price(0.02),            // "NYSE_LVS",
    Fee::Price(0.02),            // "NYSE_LYB",
    Fee::Price(0.02),            // "NYSE_LYV",
    Fee::Price(0.02),            // "NYSE_M",
    Fee::Price(0.02),            // "NYSE_MA",
    Fee::Price(0.02),            // "NYSE_MAN",
    Fee::Price(0.02),            // "NYSE_MAS",
    Fee::Price(0.02),            // "NYSE_MCD",
    Fee::Price(0.02),            // "NYSE_MCK",
    Fee::Price(0.02),            // "NYSE_MCO",
    Fee::Price(0.02),            // "NYSE_MDT",
    Fee::Price(0.02),            // "NYSE_MET",
    Fee::Price(0.02),            // "NYSE_MGM",
    Fee::Price(0.02),            // "NYSE_MHK",
    Fee::Price(0.02),            // "NYSE_MKL",
    Fee::Price(0.02),            // "NYSE_MLM",
    Fee::Price(0.02),            // "NYSE_MMC",
    Fee::Price(0.02),            // "NYSE_MMM",
    Fee::Price(0.02),            // "NYSE_MO",
    Fee::Price(0.02),            // "NYSE_MODG",
    Fee::Price(0.02),            // "NYSE_MOH",
    Fee::Price(0.02),            // "NYSE_MOS",
    Fee::Price(0.02),            // "NYSE_MRK",
    Fee::Price(0.02),            // "NYSE_MS",
    Fee::Price(0.02),            // "NYSE_MSCI",
    Fee::Price(0.02),            // "NYSE_MSI",
    Fee::Price(0.02),            // "NYSE_MSM",
    Fee::Price(0.02),            // "NYSE_MTB",
    Fee::Price(0.02),            // "NYSE_MTD",
    Fee::Price(0.02),            // "NYSE_MTN",
    Fee::Price(0.02),            // "NYSE_MTZ",
    Fee::Price(0.02),            // "NYSE_NEE",
    Fee::Price(0.02),            // "NYSE_NEM",
    Fee::Price(0.02),            // "NYSE_NI",
    Fee::Price(0.02),            // "NYSE_NKE",
    Fee::Price(0.02),            // "NYSE_NOC",
    Fee::Price(0.02),            // "NYSE_NOV",
    Fee::Price(0.02),            // "NYSE_NRG",
    Fee::Price(0.02),            // "NYSE_NSC",
    Fee::Price(0.02),            // "NYSE_NUE",
    Fee::Price(0.02),            // "NYSE_NYT",
    Fee::Price(0.02),            // "NYSE_OC",
    Fee::Price(0.02),            // "NYSE_OGE",
    Fee::Price(0.02),            // "NYSE_OKE",
    Fee::Price(0.02),            // "NYSE_OLN",
    Fee::Price(0.02),            // "NYSE_OMC",
    Fee::Price(0.02),            // "NYSE_ORCL",
    Fee::Price(0.02),            // "NYSE_ORI",
    Fee::Price(0.02),            // "NYSE_OSK",
    Fee::Price(0.02),            // "NYSE_OXY",
    Fee::Price(0.02),            // "NYSE_PEG",
    Fee::Price(0.02),            // "NYSE_PFE",
    Fee::Price(0.02),            // "NYSE_PG",
    Fee::Price(0.02),            // "NYSE_PGR",
    Fee::Price(0.02),            // "NYSE_PH",
    Fee::Price(0.02),            // "NYSE_PHM",
    Fee::Price(0.02),            // "NYSE_PII",
    Fee::Price(0.02),            // "NYSE_PKG",
    Fee::Price(0.02),            // "NYSE_PLD",
    Fee::Price(0.02),            // "NYSE_PM",
    Fee::Price(0.02),            // "NYSE_PNC",
    Fee::Price(0.02),            // "NYSE_PNW",
    Fee::Price(0.02),            // "NYSE_PPG",
    Fee::Price(0.02),            // "NYSE_PPL",
    Fee::Price(0.02),            // "NYSE_PRU",
    Fee::Price(0.02),            // "NYSE_PSA",
    Fee::Price(0.02),            // "NYSE_PVH",
    Fee::Price(0.02),            // "NYSE_PWR",
    Fee::Price(0.02),            // "NYSE_R",
    Fee::Price(0.02),            // "NYSE_RCL",
    Fee::Price(0.02),            // "NYSE_RF",
    Fee::Price(0.02),            // "NYSE_RGA",
    Fee::Price(0.02),            // "NYSE_RHI",
    Fee::Price(0.02),            // "NYSE_RJF",
    Fee::Price(0.02),            // "NYSE_RL",
    Fee::Price(0.02),            // "NYSE_RMD",
    Fee::Price(0.02),            // "NYSE_ROK",
    Fee::Price(0.02),            // "NYSE_ROL",
    Fee::Price(0.02),            // "NYSE_RPM",
    Fee::Price(0.02),            // "NYSE_RRX",
    Fee::Price(0.02),            // "NYSE_RS",
    Fee::Price(0.02),            // "NYSE_RSG",
    Fee::Price(0.02),            // "NYSE_RTX",
    Fee::Price(0.02),            // "NYSE_RVTY",
    Fee::Price(0.02),            // "NYSE_SAM",
    Fee::Price(0.02),            // "NYSE_SCHW",
    Fee::Price(0.02),            // "NYSE_SCI",
    Fee::Price(0.02),            // "NYSE_SEE",
    Fee::Price(0.02),            // "NYSE_SF",
    Fee::Price(0.02),            // "NYSE_SHW",
    Fee::Price(0.02),            // "NYSE_SJM",
    Fee::Price(0.02),            // "NYSE_SKX",
    Fee::Price(0.02),            // "NYSE_SLB",
    Fee::Price(0.02),            // "NYSE_SMG",
    Fee::Price(0.02),            // "NYSE_SNA",
    Fee::Price(0.02),            // "NYSE_SNV",
    Fee::Price(0.02),            // "NYSE_SNX",
    Fee::Price(0.02),            // "NYSE_SO",
    Fee::Price(0.02),            // "NYSE_SPG",
    Fee::Price(0.02),            // "NYSE_SPGI",
    Fee::Price(0.02),            // "NYSE_SPR",
    Fee::Price(0.02),            // "NYSE_SRE",
    Fee::Price(0.02),            // "NYSE_ST",
    Fee::Price(0.02),            // "NYSE_STE",
    Fee::Price(0.02),            // "NYSE_STT",
    Fee::Price(0.02),            // "NYSE_STZ",
    Fee::Price(0.02),            // "NYSE_SWK",
    Fee::Price(0.02),            // "NYSE_SYK",
    Fee::Price(0.02),            // "NYSE_SYY",
    Fee::Price(0.02),            // "NYSE_T",
    Fee::Price(0.02),            // "NYSE_TAP",
    Fee::Price(0.02),            // "NYSE_TDG",
    Fee::Price(0.02),            // "NYSE_TDY",
    Fee::Price(0.02),            // "NYSE_TFC",
    Fee::Price(0.02),            // "NYSE_TFX",
    Fee::Price(0.02),            // "NYSE_TGT",
    Fee::Price(0.02),            // "NYSE_THC",
    Fee::Price(0.02),            // "NYSE_THO",
    Fee::Price(0.02),            // "NYSE_TJX",
    Fee::Price(0.02),            // "NYSE_TKR",
    Fee::Price(0.02),            // "NYSE_TMO",
    Fee::Price(0.02),            // "NYSE_TNL",
    Fee::Price(0.02),            // "NYSE_TOL",
    Fee::Price(0.02),            // "NYSE_TPL",
    Fee::Price(0.02),            // "NYSE_TPR",
    Fee::Price(0.02),            // "NYSE_TPX",
    Fee::Price(0.02),            // "NYSE_TREX",
    Fee::Price(0.02),            // "NYSE_TRV",
    Fee::Price(0.02),            // "NYSE_TSN",
    Fee::Price(0.02),            // "NYSE_TTC",
    Fee::Price(0.02),            // "NYSE_TXT",
    Fee::Price(0.02),            // "NYSE_TYL",
    Fee::Price(0.02),            // "NYSE_UAA",
    Fee::Price(0.02),            // "NYSE_UGI",
    Fee::Price(0.02),            // "NYSE_UHS",
    Fee::Price(0.02),            // "NYSE_UNH",
    Fee::Price(0.02),            // "NYSE_UNM",
    Fee::Price(0.02),            // "NYSE_UNP",
    Fee::Price(0.02),            // "NYSE_UPS",
    Fee::Price(0.02),            // "NYSE_URI",
    Fee::Price(0.02),            // "NYSE_USB",
    Fee::Price(0.02),            // "NYSE_V",
    Fee::Price(0.02),            // "NYSE_VFC",
    Fee::Price(0.02),            // "NYSE_VLO",
    Fee::Price(0.02),            // "NYSE_VMC",
    Fee::Price(0.02),            // "NYSE_VYX",
    Fee::Price(0.02),            // "NYSE_VZ",
    Fee::Price(0.02),            // "NYSE_WAB",
    Fee::Price(0.02),            // "NYSE_WAL",
    Fee::Price(0.02),            // "NYSE_WAT",
    Fee::Price(0.02),            // "NYSE_WCC",
    Fee::Price(0.02),            // "NYSE_WEC",
    Fee::Price(0.02),            // "NYSE_WEX",
    Fee::Price(0.02),            // "NYSE_WFC",
    Fee::Price(0.02),            // "NYSE_WHR",
    Fee::Price(0.02),            // "NYSE_WM",
    Fee::Price(0.02),            // "NYSE_WMB",
    Fee::Price(0.02),            // "NYSE_WMT",
    Fee::Price(0.02),            // "NYSE_WOLF",
    Fee::Price(0.02),            // "NYSE_WRB",
    Fee::Price(0.02),            // "NYSE_WSM",
    Fee::Price(0.02),            // "NYSE_WSO",
    Fee::Price(0.02),            // "NYSE_WST",
    Fee::Price(0.02),            // "NYSE_WTI",
    Fee::Price(0.02),            // "NYSE_WTRG",
    Fee::Price(0.02),            // "NYSE_WU",
    Fee::Price(0.02),            // "NYSE_X",
    Fee::Price(0.02),            // "NYSE_XOM",
    Fee::Price(0.02),            // "NYSE_XPO",
    Fee::Price(0.02),            // "NYSE_YUM",
    Fee::Price(0.02),            // "NYSE_ZBH",
    Fee::Contract(2.75 / 10.),   // "TVC_CAC40",
    Fee::Contract(0.35 / 1.),    // "TVC_DJI",
    Fee::OrderValue(0.000025),   // "TVC_DXY",
    Fee::OrderValue(0.000025),   // "TVC_GOLD",
    Fee::Contract(2.75 / 10.),   // "TVC_IBEX35",
    Fee::Contract(0.24 / 100.),  // "TVC_NI225",
    Fee::OrderValue(0.000025),   // "TVC_SILVER",
    Fee::Contract(0.275 / 10.),  // "TVC_SPX",
    Fee::Contract(2.75 / 10.),   // "TVC_SX5E",
    Fee::Contract(3. / 1000.),   // "TVC_UKOIL",
    Fee::OrderValue(0.000025),   // "TVC_USOIL",
    Fee::Contract(2.75 / 10.),   // "XETR_DAX",
    Fee::Contract(5. / 0.1),     // "BNC_BLX",
    Fee::Contract(2.75 / 10.),   // "TVC_NDX",
    Fee::Contract(3. / 5000.),   // "CBOT_KE1!",
    Fee::Contract(3. / 100000.), // "CBOT_UB1!",
    Fee::Contract(3. / 5000.),   // "CBOT_ZC1!",
    Fee::Contract(3. / 100000.), // "CBOT_ZF1!",
    Fee::Contract(3. / 60000.),  // "CBOT_ZL1!",
    Fee::Contract(3. / 100.),    // "CBOT_ZM1!",
    Fee::Contract(3. / 100000.), // "CBOT_ZN1!",
    Fee::Contract(3. / 5000.),   // "CBOT_ZS1!",
    Fee::Contract(3. / 100000.), // "CBOT_ZT1!",
    Fee::Contract(3. / 5000.),   // "CBOT_ZW1!",
    Fee::Contract(3. / 40000.),  // "CME_HE1!",
    Fee::Contract(3. / 5000.),   // "CME_LE1!",
    Fee::Contract(3. / 40000.),  // "COMEX_HG1!",
    Fee::Contract(3. / 100000.), // "EUREX_FGBL1!",
    Fee::Contract(3. / 100000.), // "EUREX_FGBM1!",
    Fee::Contract(3. / 100000.), // "EUREX_FGBS1!",
    Fee::Contract(3. / 100000.), // "ICEEUR_I1!",
    Fee::Contract(3. / 5000.),   // "ICEUS_CC1!",
    Fee::Contract(3. / 5000.),   // "ICEUS_CT1!",
    Fee::Contract(3. / 5000.),   // "ICEUS_KC1!",
    Fee::Contract(3. / 5000.),   // "ICEUS_SB1!",
    Fee::Contract(3. / 42000.),  // "NYMEX_HO1!",
    Fee::OrderValue(0.000025),   // "NYMEX_NG1!",
    Fee::Contract(3. / 50.),     // "NYMEX_PL1!",
    Fee::Contract(3. / 42000.),  // "NYMEX_RB1!",
    Fee::Price(0.02),            // "NASDAQ_LPLA",
    Fee::Price(0.02),            // "NYSE_TRGP",
    Fee::Price(0.02),            // "NYSE_CPAY",
    Fee::Price(0.02),            // "NYSE_BAH",
    Fee::Price(0.02),            // "NYSE_GM",
    Fee::Price(0.02),            // "NASDAQ_TSLA",
    Fee::Price(0.02),            // "NYSE_BERY",
    Fee::Price(0.02),            // "NYSE_TWLO",
    Fee::Price(0.02),            // "NASDAQ_CDW",
    Fee::Price(0.02),            // "NYSE_CHGG",
    Fee::Price(0.02),            // "NASDAQ_PANW",
    Fee::Price(0.02),            // "NYSE_HWM",
    Fee::Price(0.02),            // "NYSE_AL",
    Fee::Price(0.02),            // "NYSE_POST",
    Fee::OrderValue(0.000025),   // "CBOT_TN1!",
    Fee::Price(0.02),            // "AMEX_GSLC",
    Fee::Price(0.02),            // "NYSE_VAC",
    Fee::Price(0.02),            // "NYSE_UBER",
    Fee::Price(0.02),            // "NYSE_EQH",
    Fee::Price(0.02),            // "NYSE_ESTC",
    Fee::Price(0.02),            // "NYSE_PINS",
    Fee::Price(0.02),            // "NYSE_ENOV",
    Fee::Price(0.02),            // "NYSE_ESI",
    Fee::Price(0.02),            // "NYSE_DT",
    Fee::Price(0.02),            // "NYSE_PAYC",
    Fee::Price(0.02),            // "NASDAQ_CHX",
    Fee::Price(0.02),            // "AMEX_SCHD",
    Fee::Price(0.02),            // "NASDAQ_SAIC",
    Fee::Price(0.02),            // "NYSE_XYL",
    Fee::Price(0.02),            // "NASDAQ_ENPH",
    Fee::Price(0.02),            // "CBOE_INDA",
    Fee::Price(0.02),            // "NYSE_KEYS",
    Fee::Price(0.02),            // "NASDAQ_DBX",
    Fee::Price(0.02),            // "NYSE_IR",
    Fee::Price(0.02),            // "NYSE_GDDY",
    Fee::Price(0.02),            // "NYSE_LEVI",
    Fee::Price(0.02),            // "NASDAQ_GH",
    Fee::Price(0.02),            // "NYSE_CHWY",
    Fee::Price(0.02),            // "NYSE_HUBS",
    Fee::Price(0.02),            // "NYSE_HII",
    Fee::Price(0.02),            // "NASDAQ_CG",
    Fee::Price(0.02),            // "NYSE_ALLY",
    Fee::Price(0.02),            // "NYSE_ANET",
    Fee::Price(0.02),            // "NASDAQ_PYPL",
    Fee::Price(0.02),            // "INDEX_ETHUSD",
    Fee::Price(0.02),            // "NYSE_GWRE",
    Fee::Price(0.02),            // "NYSE_SYF",
    Fee::Price(0.02),            // "NYSE_CABO",
    Fee::Price(0.02),            // "NYSE_NET",
    Fee::Price(0.02),            // "NASDAQ_ZI",
    Fee::Price(0.02),            // "NASDAQ_QRVO",
    Fee::Price(0.02),            // "NASDAQ_NTNX",
    Fee::Price(0.02),            // "NASDAQ_ESGU",
    Fee::Price(0.02),            // "NASDAQ_LYFT",
    Fee::Price(0.02),            // "NYSE_WH",
    Fee::Price(0.02),            // "CBOE_EFAV",
    Fee::Price(0.02),            // "AMEX_ARKG",
    Fee::Price(0.02),            // "NYSE_LW",
    Fee::Price(0.02),            // "NYSE_W",
    Fee::Price(0.02),            // "AMEX_ARKW",
    Fee::Price(0.02),            // "NASDAQ_ACHC",
    Fee::Price(0.02),            // "NASDAQ_DNLI",
    Fee::Price(0.02),            // "NYSE_FBIN",
    Fee::Price(0.02),            // "NASDAQ_RARE",
    Fee::Price(0.02),            // "NYSE_VOYA",
    Fee::Price(0.02),            // "NASDAQ_ZS",
    Fee::Price(0.02),            // "NASDAQ_ZM",
    Fee::Price(0.02),            // "NYSE_PSTG",
    Fee::Price(0.02),            // "NASDAQ_ZG",
    Fee::Price(0.02),            // "NYSE_ARES",
    Fee::Price(0.02),            // "NYSE_CARR",
    Fee::Price(0.02),            // "NASDAQ_SKYY",
    Fee::Price(0.02),            // "NYSE_ZTS",
    Fee::Price(0.02),            // "CBOE_MTUM",
    Fee::Price(0.02),            // "NYSE_SMAR",
    Fee::Price(0.02),            // "NASDAQ_FOXA",
    Fee::Price(0.02),            // "NASDAQ_VIR",
    Fee::Price(0.02),            // "NASDAQ_META",
    Fee::Price(0.02),            // "NYSE_CFG",
    Fee::Price(0.02),            // "NYSE_TRU",
    Fee::Price(0.02),            // "NYSE_SITE",
    Fee::Price(0.02),            // "NYSE_GMED",
    Fee::Price(0.02),            // "NASDAQ_MDB",
    Fee::Price(0.02),            // "NYSE_BURL",
    Fee::Price(0.02),            // "NYSE_COTY",
    Fee::Price(0.02),            // "NASDAQ_TNDM",
    Fee::Price(0.02),            // "NASDAQ_BPMC",
    Fee::Price(0.02),            // "NASDAQ_FIVN",
    Fee::Price(0.02),            // "NYSE_NVST",
    Fee::Price(0.02),            // "NASDAQ_RRR",
    Fee::Price(0.02),            // "NYSE_HCA",
    Fee::Price(0.02),            // "NYSE_AVTR",
    Fee::Price(0.02),            // "NYSE_CC",
    Fee::Price(0.02),            // "NASDAQ_FOXF",
    Fee::Price(0.02),            // "NASDAQ_APLS",
    Fee::Price(0.02),            // "NASDAQ_TTD",
    Fee::Price(0.02),            // "NYSE_ABBV",
    Fee::Price(0.02),            // "NYSE_PEN",
    Fee::Price(0.02),            // "NASDAQ_FANG",
    Fee::Price(0.02),            // "NYSE_BJ",
    Fee::Price(0.02),            // "NYSE_BILL",
    Fee::Price(0.02),            // "NYSE_WK",
    Fee::Price(0.02),            // "NASDAQ_PTON",
    Fee::Price(0.02),            // "NASDAQ_VXUS",
    Fee::Price(0.02),            // "NYSE_MPC",
    Fee::Price(0.02),            // "NASDAQ_COIN",
    Fee::Price(0.02),            // "NASDAQ_OKTA",
    Fee::Price(0.02),            // "NYSE_NCLH",
    Fee::Price(0.02),            // "NASDAQ_FRPT",
    Fee::Price(0.02),            // "NYSE_CTLT",
    Fee::Price(0.02),            // "NYSE_YETI",
    Fee::Price(0.02),            // "NYSE_OMF",
    Fee::Price(0.02),            // "NASDAQ_VIRT",
    Fee::Price(0.02),            // "NYSE_ELAN",
    Fee::Price(0.02),            // "NYSE_WMS",
    Fee::Price(0.02),            // "CBOE_VLUE",
    Fee::Price(0.02),            // "AMEX_XLC",
    Fee::Price(0.02),            // "NASDAQ_PCTY",
    Fee::Price(0.02),            // "NYSE_BFAM",
    Fee::Price(0.02),            // "NYSE_BLD",
    Fee::Price(0.02),            // "NYSE_EPAM",
    Fee::Price(0.02),            // "NYSE_IQV",
    Fee::Price(0.02),            // "NYSE_RNG",
    Fee::Price(0.02),            // "NYSE_OTIS",
    Fee::Price(0.02),            // "NYSE_DELL",
    Fee::Price(0.02),            // "NYSE_VVV",
    Fee::Price(0.02),            // "NYSE_KMI",
    Fee::Price(0.02),            // "NASDAQ_RUN",
    Fee::Price(0.02),            // "NASDAQ_CRWD",
    Fee::Price(0.02),            // "NASDAQ_VRNS",
    Fee::Price(0.02),            // "NASDAQ_NTLA",
    Fee::Price(0.02),            // "NASDAQ_DOCU",
    Fee::Price(0.02),            // "NYSE_ZWS",
    Fee::Price(0.02),            // "NASDAQ_MRNA",
    Fee::Price(0.02),            // "NASDAQ_LITE",
    Fee::Price(0.02),            // "NYSE_RH",
    Fee::Price(0.02),            // "AMEX_ARKK",
    Fee::Price(0.02),            // "NASDAQ_MEDP",
    Fee::Price(0.02),            // "NASDAQ_ROKU",
    Fee::Price(0.02),            // "CBOE_USMV",
    Fee::Price(0.02),            // "NYSE_AXTA",
    Fee::Price(0.02),            // "NYSE_CTVA",
    Fee::Price(0.02),            // "NASDAQ_KHC",
    Fee::Price(0.02),            // "NYSE_VST",
    Fee::Price(0.02),            // "NASDAQ_WDAY",
    Fee::Price(0.02),            // "NYSE_SQ",
    Fee::Price(0.02),            // "NYSE_DXC",
    Fee::Price(0.02),            // "AMEX_SPLV",
    Fee::Price(0.02),            // "NYSE_ESNT",
    Fee::Price(0.02),            // "NYSE_ARMK",
    Fee::Price(0.02),            // "NYSE_NOW",
    Fee::Price(0.02),            // "NYSE_HPE",
    Fee::Price(0.02),            // "NASDAQ_BL",
    Fee::Price(0.02),            // "NYSE_FND",
    Fee::Price(0.02),            // "AMEX_DGRO",
    Fee::Price(0.02),            // "NASDAQ_DDOG",
    Fee::Price(0.02),            // "NASDAQ_FIVE",
    Fee::Price(0.02),            // "NASDAQ_GOOG",
    Fee::Price(0.02),            // "NYSE_DOW",
    Fee::Price(0.02),            // "NYSE_FTV",
    Fee::Price(0.02),            // "NYSE_DAY",
    Fee::Price(0.02),            // "NASDAQ_MCHI",
    Fee::Price(0.02),            // "NYSE_SNAP",
    Fee::Price(0.02),            // "NASDAQ_PGNY",
    Fee::Price(0.02),            // "NYSE_TDOC",
    Fee::Price(0.02),            // "NASDAQ_HQY",
    Fee::Price(0.02),            // "NASDAQ_TXG",
    Fee::Price(0.02),            // "NASDAQ_TRIP",
    Fee::Price(0.02),            // "NASDAQ_FOX",
    Fee::Price(0.02),            // "NYSE_QTWO",
    Fee::Price(0.02),            // "NASDAQ_ETSY",
    Fee::Price(0.02),            // "NYSE_USFD",
    Fee::Price(0.02),            // "AMEX_HDV",
    Fee::Price(0.02),            // "NASDAQ_NWSA",
    Fee::Price(0.02),            // "NYSE_PLNT",
    Fee::Price(0.02),            // "NYSE_VEEV",
    Fee::Price(0.02),            // "CBOE_QUAL",
    Fee::Price(0.02),            // "AMEX_FTEC",
    Fee::Price(0.02),            // "NASDAQ_OLLI",
    Fee::Price(0.02),            // "NYSE_INSP",
    Fee::Price(0.02),            // "NYSE_CVNA",
    Fee::Price(0.02),            // "NYSE_HLT",
    Fee::Price(0.02),            // "NASDAQ_LAZR",
    Fee::Price(0.02),            // "NYSE_PFGC",
    Fee::Price(0.02),            // "NASDAQ_EXPI",
];

pub const N_MARKETS: usize = 920;
pub const MARKET_ID_TO_MARKET: [&'static str; N_MARKETS] = [
    "AMEX_DIA",
    "AMEX_EEM",
    "AMEX_EFA",
    "AMEX_EWJ",
    "AMEX_EWT",
    "AMEX_EWY",
    "AMEX_EWZ",
    "AMEX_FDN",
    "AMEX_FVD",
    "AMEX_GDX",
    "AMEX_GDXJ",
    "AMEX_GLD",
    "AMEX_IHI",
    "AMEX_IJH",
    "AMEX_IJR",
    "AMEX_IJS",
    "AMEX_ITOT",
    "AMEX_IVE",
    "AMEX_IVW",
    "AMEX_IWB",
    "AMEX_IWD",
    "AMEX_IWF",
    "AMEX_IWM",
    "AMEX_IWN",
    "AMEX_IWO",
    "AMEX_IWS",
    "AMEX_IYR",
    "AMEX_IYW",
    "AMEX_KRE",
    "AMEX_MDY",
    "AMEX_OEF",
    "AMEX_RSP",
    "AMEX_SCHB",
    "AMEX_SCHF",
    "AMEX_SCHG",
    "AMEX_SCHV",
    "AMEX_SLV",
    "AMEX_SPDW",
    "AMEX_SPY",
    "AMEX_SPYG",
    "AMEX_SPYV",
    "AMEX_TIP",
    "AMEX_VDE",
    "AMEX_VEA",
    "AMEX_VGT",
    "AMEX_VHT",
    "AMEX_VNQ",
    "AMEX_VOE",
    "AMEX_VPL",
    "AMEX_VT",
    "AMEX_VTI",
    "AMEX_VWO",
    "AMEX_VXF",
    "AMEX_XBI",
    "AMEX_XLB",
    "AMEX_XLE",
    "AMEX_XLF",
    "AMEX_XLI",
    "AMEX_XLK",
    "AMEX_XLP",
    "AMEX_XLU",
    "AMEX_XLV",
    "AMEX_XLY",
    "CAPITALCOM_RTY",
    "CBOE_EFG",
    "CBOE_EFV",
    "CBOE_EZU",
    "CBOE_IGV",
    "FX_AUDCAD",
    "FX_AUDCHF",
    "FX_AUDJPY",
    "FX_AUDNZD",
    "FX_AUDUSD",
    "FX_AUS200",
    "FX_CADCHF",
    "FX_CADJPY",
    "FX_CHFJPY",
    "FX_EURAUD",
    "FX_EURCAD",
    "FX_EURCHF",
    "FX_EURGBP",
    "FX_EURJPY",
    "FX_EURNZD",
    "FX_EURUSD",
    "FX_GBPAUD",
    "FX_GBPCAD",
    "FX_GBPCHF",
    "FX_GBPJPY",
    "FX_GBPNZD",
    "FX_GBPUSD",
    "FX_IDC_CADUSD",
    "FX_IDC_CHFUSD",
    "FX_IDC_EURMXN",
    "FX_IDC_EURNOK",
    "FX_IDC_EURSEK",
    "FX_IDC_GBPMXN",
    "FX_IDC_GBPNOK",
    "FX_IDC_GBPSEK",
    "FX_IDC_JPYUSD",
    "FX_IDC_USDNOK",
    "FX_IDC_USDSGD",
    "FX_NZDCAD",
    "FX_NZDCHF",
    "FX_NZDJPY",
    "FX_NZDUSD",
    "FX_UK100",
    "FX_USDCAD",
    "FX_USDCHF",
    "FX_USDJPY",
    "FX_USDMXN",
    "FX_USDSEK",
    "NASDAQ_AAL",
    "NASDAQ_AAPL",
    "NASDAQ_AAXJ",
    "NASDAQ_ADBE",
    "NASDAQ_ADI",
    "NASDAQ_ADP",
    "NASDAQ_ADSK",
    "NASDAQ_AEP",
    "NASDAQ_AKAM",
    "NASDAQ_ALGN",
    "NASDAQ_ALNY",
    "NASDAQ_AMAT",
    "NASDAQ_AMD",
    "NASDAQ_AMED",
    "NASDAQ_AMGN",
    "NASDAQ_AMKR",
    "NASDAQ_AMZN",
    "NASDAQ_ANSS",
    "NASDAQ_APA",
    "NASDAQ_ARCC",
    "NASDAQ_ARWR",
    "NASDAQ_AVGO",
    "NASDAQ_AXON",
    "NASDAQ_AZPN",
    "NASDAQ_AZTA",
    "NASDAQ_BIIB",
    "NASDAQ_BKNG",
    "NASDAQ_BKR",
    "NASDAQ_BMRN",
    "NASDAQ_BRKR",
    "NASDAQ_CACC",
    "NASDAQ_CAR",
    "NASDAQ_CASY",
    "NASDAQ_CDNS",
    "NASDAQ_CGNX",
    "NASDAQ_CHDN",
    "NASDAQ_CHRW",
    "NASDAQ_CHTR",
    "NASDAQ_CINF",
    "NASDAQ_CMCSA",
    "NASDAQ_CME",
    "NASDAQ_COLM",
    "NASDAQ_COO",
    "NASDAQ_COST",
    "NASDAQ_CPRT",
    "NASDAQ_CROX",
    "NASDAQ_CSCO",
    "NASDAQ_CSGP",
    "NASDAQ_CSX",
    "NASDAQ_CTAS",
    "NASDAQ_CTSH",
    "NASDAQ_DLTR",
    "NASDAQ_DOX",
    "NASDAQ_DVY",
    "NASDAQ_DXCM",
    "NASDAQ_EA",
    "NASDAQ_EBAY",
    "NASDAQ_EEFT",
    "NASDAQ_EMB",
    "NASDAQ_ENTG",
    "NASDAQ_EVRG",
    "NASDAQ_EWBC",
    "NASDAQ_EXC",
    "NASDAQ_EXEL",
    "NASDAQ_EXPE",
    "NASDAQ_FAST",
    "NASDAQ_FFIV",
    "NASDAQ_FITB",
    "NASDAQ_FLEX",
    "NASDAQ_FSLR",
    "NASDAQ_FTNT",
    "NASDAQ_GEN",
    "NASDAQ_GILD",
    "NASDAQ_GNTX",
    "NASDAQ_GOOGL",
    "NASDAQ_GT",
    "NASDAQ_HALO",
    "NASDAQ_HAS",
    "NASDAQ_HBAN",
    "NASDAQ_HELE",
    "NASDAQ_HOLX",
    "NASDAQ_HON",
    "NASDAQ_HSIC",
    "NASDAQ_IBB",
    "NASDAQ_IBKR",
    "NASDAQ_ICLN",
    "NASDAQ_IDXX",
    "NASDAQ_ILMN",
    "NASDAQ_INCY",
    "NASDAQ_INTC",
    "NASDAQ_INTU",
    "NASDAQ_IONS",
    "NASDAQ_IPGP",
    "NASDAQ_IRDM",
    "NASDAQ_ISRG",
    "NASDAQ_IUSG",
    "NASDAQ_IUSV",
    "NASDAQ_JBHT",
    "NASDAQ_JBLU",
    "NASDAQ_JKHY",
    "NASDAQ_KDP",
    "NASDAQ_KLAC",
    "NASDAQ_LKQ",
    "NASDAQ_LNT",
    "NASDAQ_LNW",
    "NASDAQ_LRCX",
    "NASDAQ_LSCC",
    "NASDAQ_LSTR",
    "NASDAQ_MANH",
    "NASDAQ_MAR",
    "NASDAQ_MASI",
    "NASDAQ_MAT",
    "NASDAQ_MCHP",
    "NASDAQ_MDLZ",
    "NASDAQ_MIDD",
    "NASDAQ_MKSI",
    "NASDAQ_MKTX",
    "NASDAQ_MNST",
    "NASDAQ_MPWR",
    "NASDAQ_MRVL",
    "NASDAQ_MSFT",
    "NASDAQ_MSTR",
    "NASDAQ_MU",
    "NASDAQ_NBIX",
    "NASDAQ_NDAQ",
    "NASDAQ_NDSN",
    "NASDAQ_NDX",
    "NASDAQ_NFLX",
    "NASDAQ_NTAP",
    "NASDAQ_NTRS",
    "NASDAQ_NVDA",
    "NASDAQ_NWL",
    "NASDAQ_NXST",
    "NASDAQ_ODFL",
    "NASDAQ_OLED",
    "NASDAQ_OMCL",
    "NASDAQ_ON",
    "NASDAQ_ORLY",
    "NASDAQ_OZK",
    "NASDAQ_PARA",
    "NASDAQ_PAYX",
    "NASDAQ_PCAR",
    "NASDAQ_PEGA",
    "NASDAQ_PENN",
    "NASDAQ_PEP",
    "NASDAQ_PFF",
    "NASDAQ_PFG",
    "NASDAQ_PNFP",
    "NASDAQ_PODD",
    "NASDAQ_POOL",
    "NASDAQ_PTC",
    "NASDAQ_QCOM",
    "NASDAQ_QQQ",
    "NASDAQ_REGN",
    "NASDAQ_RGEN",
    "NASDAQ_RGLD",
    "NASDAQ_ROP",
    "NASDAQ_ROST",
    "NASDAQ_SAIA",
    "NASDAQ_SBUX",
    "NASDAQ_SCZ",
    "NASDAQ_SLAB",
    "NASDAQ_SLM",
    "NASDAQ_SMH",
    "NASDAQ_SNPS",
    "NASDAQ_SOXX",
    "NASDAQ_SRPT",
    "NASDAQ_SSNC",
    "NASDAQ_STLD",
    "NASDAQ_SWKS",
    "NASDAQ_SYNA",
    "NASDAQ_TECH",
    "NASDAQ_TER",
    "NASDAQ_TLT",
    "NASDAQ_TMUS",
    "NASDAQ_TRMB",
    "NASDAQ_TROW",
    "NASDAQ_TSCO",
    "NASDAQ_TTEK",
    "NASDAQ_TTWO",
    "NASDAQ_TXN",
    "NASDAQ_TXRH",
    "NASDAQ_UAL",
    "NASDAQ_ULTA",
    "NASDAQ_UTHR",
    "NASDAQ_VRSK",
    "NASDAQ_VRSN",
    "NASDAQ_VRTX",
    "NASDAQ_VTRS",
    "NASDAQ_WBA",
    "NASDAQ_WDC",
    "NASDAQ_WWD",
    "NASDAQ_WYNN",
    "NASDAQ_XEL",
    "NASDAQ_XRAY",
    "NASDAQ_ZBRA",
    "NASDAQ_ZD",
    "NASDAQ_ZION",
    "NYSE_A",
    "NYSE_AA",
    "NYSE_AAP",
    "NYSE_ABT",
    "NYSE_ACM",
    "NYSE_ACN",
    "NYSE_ADM",
    "NYSE_AES",
    "NYSE_AFG",
    "NYSE_AFL",
    "NYSE_AGCO",
    "NYSE_AIG",
    "NYSE_AIZ",
    "NYSE_AJG",
    "NYSE_ALB",
    "NYSE_ALK",
    "NYSE_ALL",
    "NYSE_AME",
    "NYSE_AMG",
    "NYSE_AMP",
    "NYSE_AMT",
    "NYSE_AN",
    "NYSE_AON",
    "NYSE_AOS",
    "NYSE_APD",
    "NYSE_APH",
    "NYSE_ARW",
    "NYSE_ASH",
    "NYSE_AVY",
    "NYSE_AWI",
    "NYSE_AWK",
    "NYSE_AXP",
    "NYSE_AYI",
    "NYSE_AZO",
    "NYSE_BA",
    "NYSE_BAC",
    "NYSE_BALL",
    "NYSE_BAX",
    "NYSE_BBY",
    "NYSE_BC",
    "NYSE_BDX",
    "NYSE_BEN",
    "NYSE_BIO",
    "NYSE_BK",
    "NYSE_BLDR",
    "NYSE_BLK",
    "NYSE_BMY",
    "NYSE_BR",
    "NYSE_BRK.A",
    "NYSE_BRO",
    "NYSE_BSX",
    "NYSE_BWA",
    "NYSE_BX",
    "NYSE_BYD",
    "NYSE_C",
    "NYSE_CACI",
    "NYSE_CAG",
    "NYSE_CAH",
    "NYSE_CAT",
    "NYSE_CB",
    "NYSE_CBRE",
    "NYSE_CCI",
    "NYSE_CCK",
    "NYSE_CCL",
    "NYSE_CE",
    "NYSE_CF",
    "NYSE_CFR",
    "NYSE_CHD",
    "NYSE_CHE",
    "NYSE_CI",
    "NYSE_CIEN",
    "NYSE_CL",
    "NYSE_CLF",
    "NYSE_CLX",
    "NYSE_CMA",
    "NYSE_CMG",
    "NYSE_CMI",
    "NYSE_CMS",
    "NYSE_CNC",
    "NYSE_CNP",
    "NYSE_COF",
    "NYSE_COHR",
    "NYSE_COP",
    "NYSE_COR",
    "NYSE_CRL",
    "NYSE_CRM",
    "NYSE_CSL",
    "NYSE_CVS",
    "NYSE_CVX",
    "NYSE_D",
    "NYSE_DAL",
    "NYSE_DAR",
    "NYSE_DD",
    "NYSE_DE",
    "NYSE_DECK",
    "NYSE_DFS",
    "NYSE_DG",
    "NYSE_DGX",
    "NYSE_DHI",
    "NYSE_DHR",
    "NYSE_DIS",
    "NYSE_DKS",
    "NYSE_DLB",
    "NYSE_DOV",
    "NYSE_DPZ",
    "NYSE_DRI",
    "NYSE_DTE",
    "NYSE_DUK",
    "NYSE_DVA",
    "NYSE_DVN",
    "NYSE_ECL",
    "NYSE_EFX",
    "NYSE_EHC",
    "NYSE_EIX",
    "NYSE_EL",
    "NYSE_ELV",
    "NYSE_EME",
    "NYSE_EMN",
    "NYSE_EMR",
    "NYSE_EOG",
    "NYSE_EQT",
    "NYSE_ES",
    "NYSE_ETN",
    "NYSE_ETR",
    "NYSE_EVR",
    "NYSE_EW",
    "NYSE_EXP",
    "NYSE_EXPD",
    "NYSE_F",
    "NYSE_FAF",
    "NYSE_FCX",
    "NYSE_FDS",
    "NYSE_FDX",
    "NYSE_FE",
    "NYSE_FHN",
    "NYSE_FI",
    "NYSE_FICO",
    "NYSE_FIS",
    "NYSE_FLS",
    "NYSE_FMC",
    "NYSE_FNF",
    "NYSE_G",
    "NYSE_GAP",
    "NYSE_GD",
    "NYSE_GE",
    "NYSE_GGG",
    "NYSE_GIS",
    "NYSE_GL",
    "NYSE_GLW",
    "NYSE_GNRC",
    "NYSE_GPC",
    "NYSE_GPK",
    "NYSE_GPN",
    "NYSE_GS",
    "NYSE_GTLS",
    "NYSE_GWW",
    "NYSE_H",
    "NYSE_HAL",
    "NYSE_HBI",
    "NYSE_HD",
    "NYSE_HEI",
    "NYSE_HES",
    "NYSE_HIG",
    "NYSE_HOG",
    "NYSE_HPQ",
    "NYSE_HUBB",
    "NYSE_HUM",
    "NYSE_HUN",
    "NYSE_HXL",
    "NYSE_IBM",
    "NYSE_ICE",
    "NYSE_IEX",
    "NYSE_IFF",
    "NYSE_IGT",
    "NYSE_INGR",
    "NYSE_IP",
    "NYSE_IPG",
    "NYSE_IT",
    "NYSE_ITT",
    "NYSE_ITW",
    "NYSE_IVZ",
    "NYSE_J",
    "NYSE_JBL",
    "NYSE_JCI",
    "NYSE_JEF",
    "NYSE_JLL",
    "NYSE_JNJ",
    "NYSE_JNPR",
    "NYSE_JPM",
    "NYSE_JWN",
    "NYSE_K",
    "NYSE_KBR",
    "NYSE_KEY",
    "NYSE_KKR",
    "NYSE_KMB",
    "NYSE_KMX",
    "NYSE_KNX",
    "NYSE_KO",
    "NYSE_KR",
    "NYSE_KSS",
    "NYSE_L",
    "NYSE_LAD",
    "NYSE_LDOS",
    "NYSE_LEA",
    "NYSE_LEG",
    "NYSE_LEN",
    "NYSE_LH",
    "NYSE_LHX",
    "NYSE_LII",
    "NYSE_LLY",
    "NYSE_LMT",
    "NYSE_LNC",
    "NYSE_LOW",
    "NYSE_LPX",
    "NYSE_LUMN",
    "NYSE_LUV",
    "NYSE_LVS",
    "NYSE_LYB",
    "NYSE_LYV",
    "NYSE_M",
    "NYSE_MA",
    "NYSE_MAN",
    "NYSE_MAS",
    "NYSE_MCD",
    "NYSE_MCK",
    "NYSE_MCO",
    "NYSE_MDT",
    "NYSE_MET",
    "NYSE_MGM",
    "NYSE_MHK",
    "NYSE_MKL",
    "NYSE_MLM",
    "NYSE_MMC",
    "NYSE_MMM",
    "NYSE_MO",
    "NYSE_MODG",
    "NYSE_MOH",
    "NYSE_MOS",
    "NYSE_MRK",
    "NYSE_MS",
    "NYSE_MSCI",
    "NYSE_MSI",
    "NYSE_MSM",
    "NYSE_MTB",
    "NYSE_MTD",
    "NYSE_MTN",
    "NYSE_MTZ",
    "NYSE_NEE",
    "NYSE_NEM",
    "NYSE_NI",
    "NYSE_NKE",
    "NYSE_NOC",
    "NYSE_NOV",
    "NYSE_NRG",
    "NYSE_NSC",
    "NYSE_NUE",
    "NYSE_NYT",
    "NYSE_OC",
    "NYSE_OGE",
    "NYSE_OKE",
    "NYSE_OLN",
    "NYSE_OMC",
    "NYSE_ORCL",
    "NYSE_ORI",
    "NYSE_OSK",
    "NYSE_OXY",
    "NYSE_PEG",
    "NYSE_PFE",
    "NYSE_PG",
    "NYSE_PGR",
    "NYSE_PH",
    "NYSE_PHM",
    "NYSE_PII",
    "NYSE_PKG",
    "NYSE_PLD",
    "NYSE_PM",
    "NYSE_PNC",
    "NYSE_PNW",
    "NYSE_PPG",
    "NYSE_PPL",
    "NYSE_PRU",
    "NYSE_PSA",
    "NYSE_PVH",
    "NYSE_PWR",
    "NYSE_R",
    "NYSE_RCL",
    "NYSE_RF",
    "NYSE_RGA",
    "NYSE_RHI",
    "NYSE_RJF",
    "NYSE_RL",
    "NYSE_RMD",
    "NYSE_ROK",
    "NYSE_ROL",
    "NYSE_RPM",
    "NYSE_RRX",
    "NYSE_RS",
    "NYSE_RSG",
    "NYSE_RTX",
    "NYSE_RVTY",
    "NYSE_SAM",
    "NYSE_SCHW",
    "NYSE_SCI",
    "NYSE_SEE",
    "NYSE_SF",
    "NYSE_SHW",
    "NYSE_SJM",
    "NYSE_SKX",
    "NYSE_SLB",
    "NYSE_SMG",
    "NYSE_SNA",
    "NYSE_SNV",
    "NYSE_SNX",
    "NYSE_SO",
    "NYSE_SPG",
    "NYSE_SPGI",
    "NYSE_SPR",
    "NYSE_SRE",
    "NYSE_ST",
    "NYSE_STE",
    "NYSE_STT",
    "NYSE_STZ",
    "NYSE_SWK",
    "NYSE_SYK",
    "NYSE_SYY",
    "NYSE_T",
    "NYSE_TAP",
    "NYSE_TDG",
    "NYSE_TDY",
    "NYSE_TFC",
    "NYSE_TFX",
    "NYSE_TGT",
    "NYSE_THC",
    "NYSE_THO",
    "NYSE_TJX",
    "NYSE_TKR",
    "NYSE_TMO",
    "NYSE_TNL",
    "NYSE_TOL",
    "NYSE_TPL",
    "NYSE_TPR",
    "NYSE_TPX",
    "NYSE_TREX",
    "NYSE_TRV",
    "NYSE_TSN",
    "NYSE_TTC",
    "NYSE_TXT",
    "NYSE_TYL",
    "NYSE_UAA",
    "NYSE_UGI",
    "NYSE_UHS",
    "NYSE_UNH",
    "NYSE_UNM",
    "NYSE_UNP",
    "NYSE_UPS",
    "NYSE_URI",
    "NYSE_USB",
    "NYSE_V",
    "NYSE_VFC",
    "NYSE_VLO",
    "NYSE_VMC",
    "NYSE_VYX",
    "NYSE_VZ",
    "NYSE_WAB",
    "NYSE_WAL",
    "NYSE_WAT",
    "NYSE_WCC",
    "NYSE_WEC",
    "NYSE_WEX",
    "NYSE_WFC",
    "NYSE_WHR",
    "NYSE_WM",
    "NYSE_WMB",
    "NYSE_WMT",
    "NYSE_WOLF",
    "NYSE_WRB",
    "NYSE_WSM",
    "NYSE_WSO",
    "NYSE_WST",
    "NYSE_WTI",
    "NYSE_WTRG",
    "NYSE_WU",
    "NYSE_X",
    "NYSE_XOM",
    "NYSE_XPO",
    "NYSE_YUM",
    "NYSE_ZBH",
    "TVC_CAC40",
    "TVC_DJI",
    "TVC_DXY",
    "TVC_GOLD",
    "TVC_IBEX35",
    "TVC_NI225",
    "TVC_SILVER",
    "TVC_SPX",
    "TVC_SX5E",
    "TVC_UKOIL",
    "TVC_USOIL",
    "XETR_DAX",
    "BNC_BLX",
    "TVC_NDX",
    "CBOT_KE1!",
    "CBOT_UB1!",
    "CBOT_ZC1!",
    "CBOT_ZF1!",
    "CBOT_ZL1!",
    "CBOT_ZM1!",
    "CBOT_ZN1!",
    "CBOT_ZS1!",
    "CBOT_ZT1!",
    "CBOT_ZW1!",
    "CME_HE1!",
    "CME_LE1!",
    "COMEX_HG1!",
    "EUREX_FGBL1!",
    "EUREX_FGBM1!",
    "EUREX_FGBS1!",
    "ICEEUR_I1!",
    "ICEUS_CC1!",
    "ICEUS_CT1!",
    "ICEUS_KC1!",
    "ICEUS_SB1!",
    "NYMEX_HO1!",
    "NYMEX_NG1!",
    "NYMEX_PL1!",
    "NYMEX_RB1!",
    "NASDAQ_LPLA",
    "NYSE_TRGP",
    "NYSE_CPAY",
    "NYSE_BAH",
    "NYSE_GM",
    "NASDAQ_TSLA",
    "NYSE_BERY",
    "NYSE_TWLO",
    "NASDAQ_CDW",
    "NYSE_CHGG",
    "NASDAQ_PANW",
    "NYSE_HWM",
    "NYSE_AL",
    "NYSE_POST",
    "CBOT_TN1!",
    "AMEX_GSLC",
    "NYSE_VAC",
    "NYSE_UBER",
    "NYSE_EQH",
    "NYSE_ESTC",
    "NYSE_PINS",
    "NYSE_ENOV",
    "NYSE_ESI",
    "NYSE_DT",
    "NYSE_PAYC",
    "NASDAQ_CHX",
    "AMEX_SCHD",
    "NASDAQ_SAIC",
    "NYSE_XYL",
    "NASDAQ_ENPH",
    "CBOE_INDA",
    "NYSE_KEYS",
    "NASDAQ_DBX",
    "NYSE_IR",
    "NYSE_GDDY",
    "NYSE_LEVI",
    "NASDAQ_GH",
    "NYSE_CHWY",
    "NYSE_HUBS",
    "NYSE_HII",
    "NASDAQ_CG",
    "NYSE_ALLY",
    "NYSE_ANET",
    "NASDAQ_PYPL",
    "INDEX_ETHUSD",
    "NYSE_GWRE",
    "NYSE_SYF",
    "NYSE_CABO",
    "NYSE_NET",
    "NASDAQ_ZI",
    "NASDAQ_QRVO",
    "NASDAQ_NTNX",
    "NASDAQ_ESGU",
    "NASDAQ_LYFT",
    "NYSE_WH",
    "CBOE_EFAV",
    "AMEX_ARKG",
    "NYSE_LW",
    "NYSE_W",
    "AMEX_ARKW",
    "NASDAQ_ACHC",
    "NASDAQ_DNLI",
    "NYSE_FBIN",
    "NASDAQ_RARE",
    "NYSE_VOYA",
    "NASDAQ_ZS",
    "NASDAQ_ZM",
    "NYSE_PSTG",
    "NASDAQ_ZG",
    "NYSE_ARES",
    "NYSE_CARR",
    "NASDAQ_SKYY",
    "NYSE_ZTS",
    "CBOE_MTUM",
    "NYSE_SMAR",
    "NASDAQ_FOXA",
    "NASDAQ_VIR",
    "NASDAQ_META",
    "NYSE_CFG",
    "NYSE_TRU",
    "NYSE_SITE",
    "NYSE_GMED",
    "NASDAQ_MDB",
    "NYSE_BURL",
    "NYSE_COTY",
    "NASDAQ_TNDM",
    "NASDAQ_BPMC",
    "NASDAQ_FIVN",
    "NYSE_NVST",
    "NASDAQ_RRR",
    "NYSE_HCA",
    "NYSE_AVTR",
    "NYSE_CC",
    "NASDAQ_FOXF",
    "NASDAQ_APLS",
    "NASDAQ_TTD",
    "NYSE_ABBV",
    "NYSE_PEN",
    "NASDAQ_FANG",
    "NYSE_BJ",
    "NYSE_BILL",
    "NYSE_WK",
    "NASDAQ_PTON",
    "NASDAQ_VXUS",
    "NYSE_MPC",
    "NASDAQ_COIN",
    "NASDAQ_OKTA",
    "NYSE_NCLH",
    "NASDAQ_FRPT",
    "NYSE_CTLT",
    "NYSE_YETI",
    "NYSE_OMF",
    "NASDAQ_VIRT",
    "NYSE_ELAN",
    "NYSE_WMS",
    "CBOE_VLUE",
    "AMEX_XLC",
    "NASDAQ_PCTY",
    "NYSE_BFAM",
    "NYSE_BLD",
    "NYSE_EPAM",
    "NYSE_IQV",
    "NYSE_RNG",
    "NYSE_OTIS",
    "NYSE_DELL",
    "NYSE_VVV",
    "NYSE_KMI",
    "NASDAQ_RUN",
    "NASDAQ_CRWD",
    "NASDAQ_VRNS",
    "NASDAQ_NTLA",
    "NASDAQ_DOCU",
    "NYSE_ZWS",
    "NASDAQ_MRNA",
    "NASDAQ_LITE",
    "NYSE_RH",
    "AMEX_ARKK",
    "NASDAQ_MEDP",
    "NASDAQ_ROKU",
    "CBOE_USMV",
    "NYSE_AXTA",
    "NYSE_CTVA",
    "NASDAQ_KHC",
    "NYSE_VST",
    "NASDAQ_WDAY",
    "NYSE_SQ",
    "NYSE_DXC",
    "AMEX_SPLV",
    "NYSE_ESNT",
    "NYSE_ARMK",
    "NYSE_NOW",
    "NYSE_HPE",
    "NASDAQ_BL",
    "NYSE_FND",
    "AMEX_DGRO",
    "NASDAQ_DDOG",
    "NASDAQ_FIVE",
    "NASDAQ_GOOG",
    "NYSE_DOW",
    "NYSE_FTV",
    "NYSE_DAY",
    "NASDAQ_MCHI",
    "NYSE_SNAP",
    "NASDAQ_PGNY",
    "NYSE_TDOC",
    "NASDAQ_HQY",
    "NASDAQ_TXG",
    "NASDAQ_TRIP",
    "NASDAQ_FOX",
    "NYSE_QTWO",
    "NASDAQ_ETSY",
    "NYSE_USFD",
    "AMEX_HDV",
    "NASDAQ_NWSA",
    "NYSE_PLNT",
    "NYSE_VEEV",
    "CBOE_QUAL",
    "AMEX_FTEC",
    "NASDAQ_OLLI",
    "NYSE_INSP",
    "NYSE_CVNA",
    "NYSE_HLT",
    "NASDAQ_LAZR",
    "NYSE_PFGC",
    "NASDAQ_EXPI",
];
pub const SESSION_DURATIONS: [f32; N_MARKETS] = [
    7.5,   // "AMEX_DIA",
    7.5,   // "AMEX_EEM",
    7.5,   // "AMEX_EFA",
    7.5,   // "AMEX_EWJ",
    7.5,   // "AMEX_EWT",
    7.5,   // "AMEX_EWY",
    7.5,   // "AMEX_EWZ",
    7.5,   // "AMEX_FDN",
    7.5,   // "AMEX_FVD",
    7.5,   // "AMEX_GDX",
    7.5,   // "AMEX_GDXJ",
    7.5,   // "AMEX_GLD",
    7.5,   // "AMEX_IHI",
    7.5,   // "AMEX_IJH",
    7.5,   // "AMEX_IJR",
    7.5,   // "AMEX_IJS",
    7.5,   // "AMEX_ITOT",
    7.5,   // "AMEX_IVE",
    7.5,   // "AMEX_IVW",
    7.5,   // "AMEX_IWB",
    7.5,   // "AMEX_IWD",
    7.5,   // "AMEX_IWF",
    7.5,   // "AMEX_IWM",
    7.5,   // "AMEX_IWN",
    7.5,   // "AMEX_IWO",
    7.5,   // "AMEX_IWS",
    7.5,   // "AMEX_IYR",
    7.5,   // "AMEX_IYW",
    7.5,   // "AMEX_KRE",
    7.5,   // "AMEX_MDY",
    7.5,   // "AMEX_OEF",
    7.5,   // "AMEX_RSP",
    7.5,   // "AMEX_SCHB",
    7.5,   // "AMEX_SCHF",
    7.5,   // "AMEX_SCHG",
    7.5,   // "AMEX_SCHV",
    7.5,   // "AMEX_SLV",
    7.5,   // "AMEX_SPDW",
    7.5,   // "AMEX_SPY",
    7.5,   // "AMEX_SPYG",
    7.5,   // "AMEX_SPYV",
    7.5,   // "AMEX_TIP",
    7.5,   // "AMEX_VDE",
    7.5,   // "AMEX_VEA",
    7.5,   // "AMEX_VGT",
    7.5,   // "AMEX_VHT",
    7.5,   // "AMEX_VNQ",
    7.5,   // "AMEX_VOE",
    7.5,   // "AMEX_VPL",
    7.5,   // "AMEX_VT",
    7.5,   // "AMEX_VTI",
    7.5,   // "AMEX_VWO",
    7.5,   // "AMEX_VXF",
    7.5,   // "AMEX_XBI",
    7.5,   // "AMEX_XLB",
    7.5,   // "AMEX_XLE",
    7.5,   // "AMEX_XLF",
    7.5,   // "AMEX_XLI",
    7.5,   // "AMEX_XLK",
    7.5,   // "AMEX_XLP",
    7.5,   // "AMEX_XLU",
    7.5,   // "AMEX_XLV",
    7.5,   // "AMEX_XLY",
    23.,   // "CAPITALCOM_RTY",
    7.5,   // "CBOE_EFG",
    7.5,   // "CBOE_EFV",
    7.5,   // "CBOE_EZU",
    7.5,   // "CBOE_IGV",
    24.,   // "FX_AUDCAD",
    24.,   // "FX_AUDCHF",
    24.,   // "FX_AUDJPY",
    24.,   // "FX_AUDNZD",
    24.,   // "FX_AUDUSD",
    23.,   // "FX_AUS200",
    24.,   // "FX_CADCHF",
    24.,   // "FX_CADJPY",
    24.,   // "FX_CHFJPY",
    24.,   // "FX_EURAUD",
    24.,   // "FX_EURCAD",
    24.,   // "FX_EURCHF",
    24.,   // "FX_EURGBP",
    24.,   // "FX_EURJPY",
    24.,   // "FX_EURNZD",
    24.,   // "FX_EURUSD",
    24.,   // "FX_GBPAUD",
    24.,   // "FX_GBPCAD",
    24.,   // "FX_GBPCHF",
    24.,   // "FX_GBPJPY",
    24.,   // "FX_GBPNZD",
    24.,   // "FX_GBPUSD",
    24.,   // "FX_IDC_CADUSD",
    24.,   // "FX_IDC_CHFUSD",
    24.,   // "FX_IDC_EURMXN",
    24.,   // "FX_IDC_EURNOK",
    24.,   // "FX_IDC_EURSEK",
    24.,   // "FX_IDC_GBPMXN",
    24.,   // "FX_IDC_GBPNOK",
    24.,   // "FX_IDC_GBPSEK",
    24.,   // "FX_IDC_JPYUSD",
    24.,   // "FX_IDC_USDNOK",
    24.,   // "FX_IDC_USDSGD",
    24.,   // "FX_NZDCAD",
    24.,   // "FX_NZDCHF",
    24.,   // "FX_NZDJPY",
    24.,   // "FX_NZDUSD",
    23.,   // "FX_UK100",
    24.,   // "FX_USDCAD",
    24.,   // "FX_USDCHF",
    24.,   // "FX_USDJPY",
    24.,   // "FX_USDMXN",
    24.,   // "FX_USDSEK",
    7.5,   // "NASDAQ_AAL",
    7.5,   // "NASDAQ_AAPL",
    7.5,   // "NASDAQ_AAXJ",
    7.5,   // "NASDAQ_ADBE",
    7.5,   // "NASDAQ_ADI",
    7.5,   // "NASDAQ_ADP",
    7.5,   // "NASDAQ_ADSK",
    7.5,   // "NASDAQ_AEP",
    7.5,   // "NASDAQ_AKAM",
    7.5,   // "NASDAQ_ALGN",
    7.5,   // "NASDAQ_ALNY",
    7.5,   // "NASDAQ_AMAT",
    7.5,   // "NASDAQ_AMD",
    7.5,   // "NASDAQ_AMED",
    7.5,   // "NASDAQ_AMGN",
    7.5,   // "NASDAQ_AMKR",
    7.5,   // "NASDAQ_AMZN",
    7.5,   // "NASDAQ_ANSS",
    7.5,   // "NASDAQ_APA",
    7.5,   // "NASDAQ_ARCC",
    7.5,   // "NASDAQ_ARWR",
    7.5,   // "NASDAQ_AVGO",
    7.5,   // "NASDAQ_AXON",
    7.5,   // "NASDAQ_AZPN",
    7.5,   // "NASDAQ_AZTA",
    7.5,   // "NASDAQ_BIIB",
    7.5,   // "NASDAQ_BKNG",
    7.5,   // "NASDAQ_BKR",
    7.5,   // "NASDAQ_BMRN",
    7.5,   // "NASDAQ_BRKR",
    7.5,   // "NASDAQ_CACC",
    7.5,   // "NASDAQ_CAR",
    7.5,   // "NASDAQ_CASY",
    7.5,   // "NASDAQ_CDNS",
    7.5,   // "NASDAQ_CGNX",
    7.5,   // "NASDAQ_CHDN",
    7.5,   // "NASDAQ_CHRW",
    7.5,   // "NASDAQ_CHTR",
    7.5,   // "NASDAQ_CINF",
    7.5,   // "NASDAQ_CMCSA",
    7.5,   // "NASDAQ_CME",
    7.5,   // "NASDAQ_COLM",
    7.5,   // "NASDAQ_COO",
    7.5,   // "NASDAQ_COST",
    7.5,   // "NASDAQ_CPRT",
    7.5,   // "NASDAQ_CROX",
    7.5,   // "NASDAQ_CSCO",
    7.5,   // "NASDAQ_CSGP",
    7.5,   // "NASDAQ_CSX",
    7.5,   // "NASDAQ_CTAS",
    7.5,   // "NASDAQ_CTSH",
    7.5,   // "NASDAQ_DLTR",
    7.5,   // "NASDAQ_DOX",
    7.5,   // "NASDAQ_DVY",
    7.5,   // "NASDAQ_DXCM",
    7.5,   // "NASDAQ_EA",
    7.5,   // "NASDAQ_EBAY",
    7.5,   // "NASDAQ_EEFT",
    7.5,   // "NASDAQ_EMB",
    7.5,   // "NASDAQ_ENTG",
    7.5,   // "NASDAQ_EVRG",
    7.5,   // "NASDAQ_EWBC",
    7.5,   // "NASDAQ_EXC",
    7.5,   // "NASDAQ_EXEL",
    7.5,   // "NASDAQ_EXPE",
    7.5,   // "NASDAQ_FAST",
    7.5,   // "NASDAQ_FFIV",
    7.5,   // "NASDAQ_FITB",
    7.5,   // "NASDAQ_FLEX",
    7.5,   // "NASDAQ_FSLR",
    7.5,   // "NASDAQ_FTNT",
    7.5,   // "NASDAQ_GEN",
    7.5,   // "NASDAQ_GILD",
    7.5,   // "NASDAQ_GNTX",
    7.5,   // "NASDAQ_GOOGL",
    7.5,   // "NASDAQ_GT",
    7.5,   // "NASDAQ_HALO",
    7.5,   // "NASDAQ_HAS",
    7.5,   // "NASDAQ_HBAN",
    7.5,   // "NASDAQ_HELE",
    7.5,   // "NASDAQ_HOLX",
    7.5,   // "NASDAQ_HON",
    7.5,   // "NASDAQ_HSIC",
    7.5,   // "NASDAQ_IBB",
    7.5,   // "NASDAQ_IBKR",
    7.5,   // "NASDAQ_ICLN",
    7.5,   // "NASDAQ_IDXX",
    7.5,   // "NASDAQ_ILMN",
    7.5,   // "NASDAQ_INCY",
    7.5,   // "NASDAQ_INTC",
    7.5,   // "NASDAQ_INTU",
    7.5,   // "NASDAQ_IONS",
    7.5,   // "NASDAQ_IPGP",
    7.5,   // "NASDAQ_IRDM",
    7.5,   // "NASDAQ_ISRG",
    7.5,   // "NASDAQ_IUSG",
    7.5,   // "NASDAQ_IUSV",
    7.5,   // "NASDAQ_JBHT",
    7.5,   // "NASDAQ_JBLU",
    7.5,   // "NASDAQ_JKHY",
    7.5,   // "NASDAQ_KDP",
    7.5,   // "NASDAQ_KLAC",
    7.5,   // "NASDAQ_LKQ",
    7.5,   // "NASDAQ_LNT",
    7.5,   // "NASDAQ_LNW",
    7.5,   // "NASDAQ_LRCX",
    7.5,   // "NASDAQ_LSCC",
    7.5,   // "NASDAQ_LSTR",
    7.5,   // "NASDAQ_MANH",
    7.5,   // "NASDAQ_MAR",
    7.5,   // "NASDAQ_MASI",
    7.5,   // "NASDAQ_MAT",
    7.5,   // "NASDAQ_MCHP",
    7.5,   // "NASDAQ_MDLZ",
    7.5,   // "NASDAQ_MIDD",
    7.5,   // "NASDAQ_MKSI",
    7.5,   // "NASDAQ_MKTX",
    7.5,   // "NASDAQ_MNST",
    7.5,   // "NASDAQ_MPWR",
    7.5,   // "NASDAQ_MRVL",
    7.5,   // "NASDAQ_MSFT",
    7.5,   // "NASDAQ_MSTR",
    7.5,   // "NASDAQ_MU",
    7.5,   // "NASDAQ_NBIX",
    7.5,   // "NASDAQ_NDAQ",
    7.5,   // "NASDAQ_NDSN",
    7.5,   // "NASDAQ_NDX",
    7.5,   // "NASDAQ_NFLX",
    7.5,   // "NASDAQ_NTAP",
    7.5,   // "NASDAQ_NTRS",
    7.5,   // "NASDAQ_NVDA",
    7.5,   // "NASDAQ_NWL",
    7.5,   // "NASDAQ_NXST",
    7.5,   // "NASDAQ_ODFL",
    7.5,   // "NASDAQ_OLED",
    7.5,   // "NASDAQ_OMCL",
    7.5,   // "NASDAQ_ON",
    7.5,   // "NASDAQ_ORLY",
    7.5,   // "NASDAQ_OZK",
    7.5,   // "NASDAQ_PARA",
    7.5,   // "NASDAQ_PAYX",
    7.5,   // "NASDAQ_PCAR",
    7.5,   // "NASDAQ_PEGA",
    7.5,   // "NASDAQ_PENN",
    7.5,   // "NASDAQ_PEP",
    7.5,   // "NASDAQ_PFF",
    7.5,   // "NASDAQ_PFG",
    7.5,   // "NASDAQ_PNFP",
    7.5,   // "NASDAQ_PODD",
    7.5,   // "NASDAQ_POOL",
    7.5,   // "NASDAQ_PTC",
    7.5,   // "NASDAQ_QCOM",
    7.5,   // "NASDAQ_QQQ",
    7.5,   // "NASDAQ_REGN",
    7.5,   // "NASDAQ_RGEN",
    7.5,   // "NASDAQ_RGLD",
    7.5,   // "NASDAQ_ROP",
    7.5,   // "NASDAQ_ROST",
    7.5,   // "NASDAQ_SAIA",
    7.5,   // "NASDAQ_SBUX",
    7.5,   // "NASDAQ_SCZ",
    7.5,   // "NASDAQ_SLAB",
    7.5,   // "NASDAQ_SLM",
    7.5,   // "NASDAQ_SMH",
    7.5,   // "NASDAQ_SNPS",
    7.5,   // "NASDAQ_SOXX",
    7.5,   // "NASDAQ_SRPT",
    7.5,   // "NASDAQ_SSNC",
    7.5,   // "NASDAQ_STLD",
    7.5,   // "NASDAQ_SWKS",
    7.5,   // "NASDAQ_SYNA",
    7.5,   // "NASDAQ_TECH",
    7.5,   // "NASDAQ_TER",
    7.5,   // "NASDAQ_TLT",
    7.5,   // "NASDAQ_TMUS",
    7.5,   // "NASDAQ_TRMB",
    7.5,   // "NASDAQ_TROW",
    7.5,   // "NASDAQ_TSCO",
    7.5,   // "NASDAQ_TTEK",
    7.5,   // "NASDAQ_TTWO",
    7.5,   // "NASDAQ_TXN",
    7.5,   // "NASDAQ_TXRH",
    7.5,   // "NASDAQ_UAL",
    7.5,   // "NASDAQ_ULTA",
    7.5,   // "NASDAQ_UTHR",
    7.5,   // "NASDAQ_VRSK",
    7.5,   // "NASDAQ_VRSN",
    7.5,   // "NASDAQ_VRTX",
    7.5,   // "NASDAQ_VTRS",
    7.5,   // "NASDAQ_WBA",
    7.5,   // "NASDAQ_WDC",
    7.5,   // "NASDAQ_WWD",
    7.5,   // "NASDAQ_WYNN",
    7.5,   // "NASDAQ_XEL",
    7.5,   // "NASDAQ_XRAY",
    7.5,   // "NASDAQ_ZBRA",
    7.5,   // "NASDAQ_ZD",
    7.5,   // "NASDAQ_ZION",
    7.5,   // "NYSE_A",
    7.5,   // "NYSE_AA",
    7.5,   // "NYSE_AAP",
    7.5,   // "NYSE_ABT",
    7.5,   // "NYSE_ACM",
    7.5,   // "NYSE_ACN",
    7.5,   // "NYSE_ADM",
    7.5,   // "NYSE_AES",
    7.5,   // "NYSE_AFG",
    7.5,   // "NYSE_AFL",
    7.5,   // "NYSE_AGCO",
    7.5,   // "NYSE_AIG",
    7.5,   // "NYSE_AIZ",
    7.5,   // "NYSE_AJG",
    7.5,   // "NYSE_ALB",
    7.5,   // "NYSE_ALK",
    7.5,   // "NYSE_ALL",
    7.5,   // "NYSE_AME",
    7.5,   // "NYSE_AMG",
    7.5,   // "NYSE_AMP",
    7.5,   // "NYSE_AMT",
    7.5,   // "NYSE_AN",
    7.5,   // "NYSE_AON",
    7.5,   // "NYSE_AOS",
    7.5,   // "NYSE_APD",
    7.5,   // "NYSE_APH",
    7.5,   // "NYSE_ARW",
    7.5,   // "NYSE_ASH",
    7.5,   // "NYSE_AVY",
    7.5,   // "NYSE_AWI",
    7.5,   // "NYSE_AWK",
    7.5,   // "NYSE_AXP",
    7.5,   // "NYSE_AYI",
    7.5,   // "NYSE_AZO",
    7.5,   // "NYSE_BA",
    7.5,   // "NYSE_BAC",
    7.5,   // "NYSE_BALL",
    7.5,   // "NYSE_BAX",
    7.5,   // "NYSE_BBY",
    7.5,   // "NYSE_BC",
    7.5,   // "NYSE_BDX",
    7.5,   // "NYSE_BEN",
    7.5,   // "NYSE_BIO",
    7.5,   // "NYSE_BK",
    7.5,   // "NYSE_BLDR",
    7.5,   // "NYSE_BLK",
    7.5,   // "NYSE_BMY",
    7.5,   // "NYSE_BR",
    7.5,   // "NYSE_BRK.A",
    7.5,   // "NYSE_BRO",
    7.5,   // "NYSE_BSX",
    7.5,   // "NYSE_BWA",
    7.5,   // "NYSE_BX",
    7.5,   // "NYSE_BYD",
    7.5,   // "NYSE_C",
    7.5,   // "NYSE_CACI",
    7.5,   // "NYSE_CAG",
    7.5,   // "NYSE_CAH",
    7.5,   // "NYSE_CAT",
    7.5,   // "NYSE_CB",
    7.5,   // "NYSE_CBRE",
    7.5,   // "NYSE_CCI",
    7.5,   // "NYSE_CCK",
    7.5,   // "NYSE_CCL",
    7.5,   // "NYSE_CE",
    7.5,   // "NYSE_CF",
    7.5,   // "NYSE_CFR",
    7.5,   // "NYSE_CHD",
    7.5,   // "NYSE_CHE",
    7.5,   // "NYSE_CI",
    7.5,   // "NYSE_CIEN",
    7.5,   // "NYSE_CL",
    7.5,   // "NYSE_CLF",
    7.5,   // "NYSE_CLX",
    7.5,   // "NYSE_CMA",
    7.5,   // "NYSE_CMG",
    7.5,   // "NYSE_CMI",
    7.5,   // "NYSE_CMS",
    7.5,   // "NYSE_CNC",
    7.5,   // "NYSE_CNP",
    7.5,   // "NYSE_COF",
    7.5,   // "NYSE_COHR",
    7.5,   // "NYSE_COP",
    7.5,   // "NYSE_COR",
    7.5,   // "NYSE_CRL",
    7.5,   // "NYSE_CRM",
    7.5,   // "NYSE_CSL",
    7.5,   // "NYSE_CVS",
    7.5,   // "NYSE_CVX",
    7.5,   // "NYSE_D",
    7.5,   // "NYSE_DAL",
    7.5,   // "NYSE_DAR",
    7.5,   // "NYSE_DD",
    7.5,   // "NYSE_DE",
    7.5,   // "NYSE_DECK",
    7.5,   // "NYSE_DFS",
    7.5,   // "NYSE_DG",
    7.5,   // "NYSE_DGX",
    7.5,   // "NYSE_DHI",
    7.5,   // "NYSE_DHR",
    7.5,   // "NYSE_DIS",
    7.5,   // "NYSE_DKS",
    7.5,   // "NYSE_DLB",
    7.5,   // "NYSE_DOV",
    7.5,   // "NYSE_DPZ",
    7.5,   // "NYSE_DRI",
    7.5,   // "NYSE_DTE",
    7.5,   // "NYSE_DUK",
    7.5,   // "NYSE_DVA",
    7.5,   // "NYSE_DVN",
    7.5,   // "NYSE_ECL",
    7.5,   // "NYSE_EFX",
    7.5,   // "NYSE_EHC",
    7.5,   // "NYSE_EIX",
    7.5,   // "NYSE_EL",
    7.5,   // "NYSE_ELV",
    7.5,   // "NYSE_EME",
    7.5,   // "NYSE_EMN",
    7.5,   // "NYSE_EMR",
    7.5,   // "NYSE_EOG",
    7.5,   // "NYSE_EQT",
    7.5,   // "NYSE_ES",
    7.5,   // "NYSE_ETN",
    7.5,   // "NYSE_ETR",
    7.5,   // "NYSE_EVR",
    7.5,   // "NYSE_EW",
    7.5,   // "NYSE_EXP",
    7.5,   // "NYSE_EXPD",
    7.5,   // "NYSE_F",
    7.5,   // "NYSE_FAF",
    7.5,   // "NYSE_FCX",
    7.5,   // "NYSE_FDS",
    7.5,   // "NYSE_FDX",
    7.5,   // "NYSE_FE",
    7.5,   // "NYSE_FHN",
    7.5,   // "NYSE_FI",
    7.5,   // "NYSE_FICO",
    7.5,   // "NYSE_FIS",
    7.5,   // "NYSE_FLS",
    7.5,   // "NYSE_FMC",
    7.5,   // "NYSE_FNF",
    7.5,   // "NYSE_G",
    7.5,   // "NYSE_GAP",
    7.5,   // "NYSE_GD",
    7.5,   // "NYSE_GE",
    7.5,   // "NYSE_GGG",
    7.5,   // "NYSE_GIS",
    7.5,   // "NYSE_GL",
    7.5,   // "NYSE_GLW",
    7.5,   // "NYSE_GNRC",
    7.5,   // "NYSE_GPC",
    7.5,   // "NYSE_GPK",
    7.5,   // "NYSE_GPN",
    7.5,   // "NYSE_GS",
    7.5,   // "NYSE_GTLS",
    7.5,   // "NYSE_GWW",
    7.5,   // "NYSE_H",
    7.5,   // "NYSE_HAL",
    7.5,   // "NYSE_HBI",
    7.5,   // "NYSE_HD",
    7.5,   // "NYSE_HEI",
    7.5,   // "NYSE_HES",
    7.5,   // "NYSE_HIG",
    7.5,   // "NYSE_HOG",
    7.5,   // "NYSE_HPQ",
    7.5,   // "NYSE_HUBB",
    7.5,   // "NYSE_HUM",
    7.5,   // "NYSE_HUN",
    7.5,   // "NYSE_HXL",
    7.5,   // "NYSE_IBM",
    7.5,   // "NYSE_ICE",
    7.5,   // "NYSE_IEX",
    7.5,   // "NYSE_IFF",
    7.5,   // "NYSE_IGT",
    7.5,   // "NYSE_INGR",
    7.5,   // "NYSE_IP",
    7.5,   // "NYSE_IPG",
    7.5,   // "NYSE_IT",
    7.5,   // "NYSE_ITT",
    7.5,   // "NYSE_ITW",
    7.5,   // "NYSE_IVZ",
    7.5,   // "NYSE_J",
    7.5,   // "NYSE_JBL",
    7.5,   // "NYSE_JCI",
    7.5,   // "NYSE_JEF",
    7.5,   // "NYSE_JLL",
    7.5,   // "NYSE_JNJ",
    7.5,   // "NYSE_JNPR",
    7.5,   // "NYSE_JPM",
    7.5,   // "NYSE_JWN",
    7.5,   // "NYSE_K",
    7.5,   // "NYSE_KBR",
    7.5,   // "NYSE_KEY",
    7.5,   // "NYSE_KKR",
    7.5,   // "NYSE_KMB",
    7.5,   // "NYSE_KMX",
    7.5,   // "NYSE_KNX",
    7.5,   // "NYSE_KO",
    7.5,   // "NYSE_KR",
    7.5,   // "NYSE_KSS",
    7.5,   // "NYSE_L",
    7.5,   // "NYSE_LAD",
    7.5,   // "NYSE_LDOS",
    7.5,   // "NYSE_LEA",
    7.5,   // "NYSE_LEG",
    7.5,   // "NYSE_LEN",
    7.5,   // "NYSE_LH",
    7.5,   // "NYSE_LHX",
    7.5,   // "NYSE_LII",
    7.5,   // "NYSE_LLY",
    7.5,   // "NYSE_LMT",
    7.5,   // "NYSE_LNC",
    7.5,   // "NYSE_LOW",
    7.5,   // "NYSE_LPX",
    7.5,   // "NYSE_LUMN",
    7.5,   // "NYSE_LUV",
    7.5,   // "NYSE_LVS",
    7.5,   // "NYSE_LYB",
    7.5,   // "NYSE_LYV",
    7.5,   // "NYSE_M",
    7.5,   // "NYSE_MA",
    7.5,   // "NYSE_MAN",
    7.5,   // "NYSE_MAS",
    7.5,   // "NYSE_MCD",
    7.5,   // "NYSE_MCK",
    7.5,   // "NYSE_MCO",
    7.5,   // "NYSE_MDT",
    7.5,   // "NYSE_MET",
    7.5,   // "NYSE_MGM",
    7.5,   // "NYSE_MHK",
    7.5,   // "NYSE_MKL",
    7.5,   // "NYSE_MLM",
    7.5,   // "NYSE_MMC",
    7.5,   // "NYSE_MMM",
    7.5,   // "NYSE_MO",
    7.5,   // "NYSE_MODG",
    7.5,   // "NYSE_MOH",
    7.5,   // "NYSE_MOS",
    7.5,   // "NYSE_MRK",
    7.5,   // "NYSE_MS",
    7.5,   // "NYSE_MSCI",
    7.5,   // "NYSE_MSI",
    7.5,   // "NYSE_MSM",
    7.5,   // "NYSE_MTB",
    7.5,   // "NYSE_MTD",
    7.5,   // "NYSE_MTN",
    7.5,   // "NYSE_MTZ",
    7.5,   // "NYSE_NEE",
    7.5,   // "NYSE_NEM",
    7.5,   // "NYSE_NI",
    7.5,   // "NYSE_NKE",
    7.5,   // "NYSE_NOC",
    7.5,   // "NYSE_NOV",
    7.5,   // "NYSE_NRG",
    7.5,   // "NYSE_NSC",
    7.5,   // "NYSE_NUE",
    7.5,   // "NYSE_NYT",
    7.5,   // "NYSE_OC",
    7.5,   // "NYSE_OGE",
    7.5,   // "NYSE_OKE",
    7.5,   // "NYSE_OLN",
    7.5,   // "NYSE_OMC",
    7.5,   // "NYSE_ORCL",
    7.5,   // "NYSE_ORI",
    7.5,   // "NYSE_OSK",
    7.5,   // "NYSE_OXY",
    7.5,   // "NYSE_PEG",
    7.5,   // "NYSE_PFE",
    7.5,   // "NYSE_PG",
    7.5,   // "NYSE_PGR",
    7.5,   // "NYSE_PH",
    7.5,   // "NYSE_PHM",
    7.5,   // "NYSE_PII",
    7.5,   // "NYSE_PKG",
    7.5,   // "NYSE_PLD",
    7.5,   // "NYSE_PM",
    7.5,   // "NYSE_PNC",
    7.5,   // "NYSE_PNW",
    7.5,   // "NYSE_PPG",
    7.5,   // "NYSE_PPL",
    7.5,   // "NYSE_PRU",
    7.5,   // "NYSE_PSA",
    7.5,   // "NYSE_PVH",
    7.5,   // "NYSE_PWR",
    7.5,   // "NYSE_R",
    7.5,   // "NYSE_RCL",
    7.5,   // "NYSE_RF",
    7.5,   // "NYSE_RGA",
    7.5,   // "NYSE_RHI",
    7.5,   // "NYSE_RJF",
    7.5,   // "NYSE_RL",
    7.5,   // "NYSE_RMD",
    7.5,   // "NYSE_ROK",
    7.5,   // "NYSE_ROL",
    7.5,   // "NYSE_RPM",
    7.5,   // "NYSE_RRX",
    7.5,   // "NYSE_RS",
    7.5,   // "NYSE_RSG",
    7.5,   // "NYSE_RTX",
    7.5,   // "NYSE_RVTY",
    7.5,   // "NYSE_SAM",
    7.5,   // "NYSE_SCHW",
    7.5,   // "NYSE_SCI",
    7.5,   // "NYSE_SEE",
    7.5,   // "NYSE_SF",
    7.5,   // "NYSE_SHW",
    7.5,   // "NYSE_SJM",
    7.5,   // "NYSE_SKX",
    7.5,   // "NYSE_SLB",
    7.5,   // "NYSE_SMG",
    7.5,   // "NYSE_SNA",
    7.5,   // "NYSE_SNV",
    7.5,   // "NYSE_SNX",
    7.5,   // "NYSE_SO",
    7.5,   // "NYSE_SPG",
    7.5,   // "NYSE_SPGI",
    7.5,   // "NYSE_SPR",
    7.5,   // "NYSE_SRE",
    7.5,   // "NYSE_ST",
    7.5,   // "NYSE_STE",
    7.5,   // "NYSE_STT",
    7.5,   // "NYSE_STZ",
    7.5,   // "NYSE_SWK",
    7.5,   // "NYSE_SYK",
    7.5,   // "NYSE_SYY",
    7.5,   // "NYSE_T",
    7.5,   // "NYSE_TAP",
    7.5,   // "NYSE_TDG",
    7.5,   // "NYSE_TDY",
    7.5,   // "NYSE_TFC",
    7.5,   // "NYSE_TFX",
    7.5,   // "NYSE_TGT",
    7.5,   // "NYSE_THC",
    7.5,   // "NYSE_THO",
    7.5,   // "NYSE_TJX",
    7.5,   // "NYSE_TKR",
    7.5,   // "NYSE_TMO",
    7.5,   // "NYSE_TNL",
    7.5,   // "NYSE_TOL",
    7.5,   // "NYSE_TPL",
    7.5,   // "NYSE_TPR",
    7.5,   // "NYSE_TPX",
    7.5,   // "NYSE_TREX",
    7.5,   // "NYSE_TRV",
    7.5,   // "NYSE_TSN",
    7.5,   // "NYSE_TTC",
    7.5,   // "NYSE_TXT",
    7.5,   // "NYSE_TYL",
    7.5,   // "NYSE_UAA",
    7.5,   // "NYSE_UGI",
    7.5,   // "NYSE_UHS",
    7.5,   // "NYSE_UNH",
    7.5,   // "NYSE_UNM",
    7.5,   // "NYSE_UNP",
    7.5,   // "NYSE_UPS",
    7.5,   // "NYSE_URI",
    7.5,   // "NYSE_USB",
    7.5,   // "NYSE_V",
    7.5,   // "NYSE_VFC",
    7.5,   // "NYSE_VLO",
    7.5,   // "NYSE_VMC",
    7.5,   // "NYSE_VYX",
    7.5,   // "NYSE_VZ",
    7.5,   // "NYSE_WAB",
    7.5,   // "NYSE_WAL",
    7.5,   // "NYSE_WAT",
    7.5,   // "NYSE_WCC",
    7.5,   // "NYSE_WEC",
    7.5,   // "NYSE_WEX",
    7.5,   // "NYSE_WFC",
    7.5,   // "NYSE_WHR",
    7.5,   // "NYSE_WM",
    7.5,   // "NYSE_WMB",
    7.5,   // "NYSE_WMT",
    7.5,   // "NYSE_WOLF",
    7.5,   // "NYSE_WRB",
    7.5,   // "NYSE_WSM",
    7.5,   // "NYSE_WSO",
    7.5,   // "NYSE_WST",
    7.5,   // "NYSE_WTI",
    7.5,   // "NYSE_WTRG",
    7.5,   // "NYSE_WU",
    7.5,   // "NYSE_X",
    7.5,   // "NYSE_XOM",
    7.5,   // "NYSE_XPO",
    7.5,   // "NYSE_YUM",
    7.5,   // "NYSE_ZBH",
    7.5,   // "TVC_CAC40",
    23.,   // "TVC_DJI",
    7.5,   // "TVC_DXY",
    23.,   // "TVC_GOLD",
    12.,   // "TVC_IBEX35",
    6.5,   // "TVC_NI225",
    23.,   // "TVC_SILVER",
    23.,   // "TVC_SPX",
    23.,   // "TVC_SX5E",
    23.,   // "TVC_UKOIL",
    23.,   // "TVC_USOIL",
    19.5,  // "XETR_DAX",
    23.,   // "BNC_BLX",
    23.,   // "TVC_NDX",
    18.33, // "CBOT_KE1!",
    7.5,   // "CBOT_UB1!",
    18.33, // "CBOT_ZC1!",
    7.5,   // "CBOT_ZF1!",
    18.33, // "CBOT_ZL1!",
    18.33, // "CBOT_ZM1!",
    23.,   // "CBOT_ZN1!",
    18.33, // "CBOT_ZS1!",
    7.5,   // "CBOT_ZT1!",
    7.5,   // "CBOT_ZW1!",
    4.5,   // "CME_HE1!",
    4.5,   // "CME_LE1!",
    23.,   // "COMEX_HG1!",
    20.5,  // "EUREX_FGBL1!",
    7.5,   // "EUREX_FGBM1!",
    7.5,   // "EUREX_FGBS1!",
    7.5,   // "ICEEUR_I1!",
    7.5,   // "ICEUS_CC1!",
    7.5,   // "ICEUS_CT1!",
    7.5,   // "ICEUS_KC1!",
    7.5,   // "ICEUS_SB1!",
    23.,   // "NYMEX_HO1!",
    23.,   // "NYMEX_NG1!",
    23.,   // "NYMEX_PL1!",
    23.,   // "NYMEX_RB1!",
    7.5,   // "NASDAQ_LPLA",
    7.5,   // "NYSE_TRGP",
    7.5,   // "NYSE_CPAY",
    7.5,   // "NYSE_BAH",
    7.5,   // "NYSE_GM",
    7.5,   // "NASDAQ_TSLA",
    7.5,   // "NYSE_BERY",
    7.5,   // "NYSE_TWLO",
    7.5,   // "NASDAQ_CDW",
    7.5,   // "NYSE_CHGG",
    7.5,   // "NASDAQ_PANW",
    7.5,   // "NYSE_HWM",
    7.5,   // "NYSE_AL",
    7.5,   // "NYSE_POST",
    23.,   // "CBOT_TN1!",
    7.5,   // "AMEX_GSLC",
    7.5,   // "NYSE_VAC",
    7.5,   // "NYSE_UBER",
    7.5,   // "NYSE_EQH",
    7.5,   // "NYSE_ESTC",
    7.5,   // "NYSE_PINS",
    7.5,   // "NYSE_ENOV",
    7.5,   // "NYSE_ESI",
    7.5,   // "NYSE_DT",
    7.5,   // "NYSE_PAYC",
    7.5,   // "NASDAQ_CHX",
    7.5,   // "AMEX_SCHD",
    7.5,   // "NASDAQ_SAIC",
    7.5,   // "NYSE_XYL",
    7.5,   // "NASDAQ_ENPH",
    7.5,   // "CBOE_INDA",
    7.5,   // "NYSE_KEYS",
    7.5,   // "NASDAQ_DBX",
    7.5,   // "NYSE_IR",
    7.5,   // "NYSE_GDDY",
    7.5,   // "NYSE_LEVI",
    7.5,   // "NASDAQ_GH",
    7.5,   // "NYSE_CHWY",
    7.5,   // "NYSE_HUBS",
    7.5,   // "NYSE_HII",
    7.5,   // "NASDAQ_CG",
    7.5,   // "NYSE_ALLY",
    7.5,   // "NYSE_ANET",
    7.5,   // "NASDAQ_PYPL",
    7.5,   // "INDEX_ETHUSD",
    7.5,   // "NYSE_GWRE",
    7.5,   // "NYSE_SYF",
    7.5,   // "NYSE_CABO",
    7.5,   // "NYSE_NET",
    7.5,   // "NASDAQ_ZI",
    7.5,   // "NASDAQ_QRVO",
    7.5,   // "NASDAQ_NTNX",
    7.5,   // "NASDAQ_ESGU",
    7.5,   // "NASDAQ_LYFT",
    7.5,   // "NYSE_WH",
    7.5,   // "CBOE_EFAV",
    7.5,   // "AMEX_ARKG",
    7.5,   // "NYSE_LW",
    7.5,   // "NYSE_W",
    7.5,   // "AMEX_ARKW",
    7.5,   // "NASDAQ_ACHC",
    7.5,   // "NASDAQ_DNLI",
    7.5,   // "NYSE_FBIN",
    7.5,   // "NASDAQ_RARE",
    7.5,   // "NYSE_VOYA",
    7.5,   // "NASDAQ_ZS",
    7.5,   // "NASDAQ_ZM",
    7.5,   // "NYSE_PSTG",
    7.5,   // "NASDAQ_ZG",
    7.5,   // "NYSE_ARES",
    7.5,   // "NYSE_CARR",
    7.5,   // "NASDAQ_SKYY",
    7.5,   // "NYSE_ZTS",
    7.5,   // "CBOE_MTUM",
    7.5,   // "NYSE_SMAR",
    7.5,   // "NASDAQ_FOXA",
    7.5,   // "NASDAQ_VIR",
    7.5,   // "NASDAQ_META",
    7.5,   // "NYSE_CFG",
    7.5,   // "NYSE_TRU",
    7.5,   // "NYSE_SITE",
    7.5,   // "NYSE_GMED",
    7.5,   // "NASDAQ_MDB",
    7.5,   // "NYSE_BURL",
    7.5,   // "NYSE_COTY",
    7.5,   // "NASDAQ_TNDM",
    7.5,   // "NASDAQ_BPMC",
    7.5,   // "NASDAQ_FIVN",
    7.5,   // "NYSE_NVST",
    7.5,   // "NASDAQ_RRR",
    7.5,   // "NYSE_HCA",
    7.5,   // "NYSE_AVTR",
    7.5,   // "NYSE_CC",
    7.5,   // "NASDAQ_FOXF",
    7.5,   // "NASDAQ_APLS",
    7.5,   // "NASDAQ_TTD",
    7.5,   // "NYSE_ABBV",
    7.5,   // "NYSE_PEN",
    7.5,   // "NASDAQ_FANG",
    7.5,   // "NYSE_BJ",
    7.5,   // "NYSE_BILL",
    7.5,   // "NYSE_WK",
    7.5,   // "NASDAQ_PTON",
    7.5,   // "NASDAQ_VXUS",
    7.5,   // "NYSE_MPC",
    7.5,   // "NASDAQ_COIN",
    7.5,   // "NASDAQ_OKTA",
    7.5,   // "NYSE_NCLH",
    7.5,   // "NASDAQ_FRPT",
    7.5,   // "NYSE_CTLT",
    7.5,   // "NYSE_YETI",
    7.5,   // "NYSE_OMF",
    7.5,   // "NASDAQ_VIRT",
    7.5,   // "NYSE_ELAN",
    7.5,   // "NYSE_WMS",
    7.5,   // "CBOE_VLUE",
    7.5,   // "AMEX_XLC",
    7.5,   // "NASDAQ_PCTY",
    7.5,   // "NYSE_BFAM",
    7.5,   // "NYSE_BLD",
    7.5,   // "NYSE_EPAM",
    7.5,   // "NYSE_IQV",
    7.5,   // "NYSE_RNG",
    7.5,   // "NYSE_OTIS",
    7.5,   // "NYSE_DELL",
    7.5,   // "NYSE_VVV",
    7.5,   // "NYSE_KMI",
    7.5,   // "NASDAQ_RUN",
    7.5,   // "NASDAQ_CRWD",
    7.5,   // "NASDAQ_VRNS",
    7.5,   // "NASDAQ_NTLA",
    7.5,   // "NASDAQ_DOCU",
    7.5,   // "NYSE_ZWS",
    7.5,   // "NASDAQ_MRNA",
    7.5,   // "NASDAQ_LITE",
    7.5,   // "NYSE_RH",
    7.5,   // "AMEX_ARKK",
    7.5,   // "NASDAQ_MEDP",
    7.5,   // "NASDAQ_ROKU",
    7.5,   // "CBOE_USMV",
    7.5,   // "NYSE_AXTA",
    7.5,   // "NYSE_CTVA",
    7.5,   // "NASDAQ_KHC",
    7.5,   // "NYSE_VST",
    7.5,   // "NASDAQ_WDAY",
    7.5,   // "NYSE_SQ",
    7.5,   // "NYSE_DXC",
    7.5,   // "AMEX_SPLV",
    7.5,   // "NYSE_ESNT",
    7.5,   // "NYSE_ARMK",
    7.5,   // "NYSE_NOW",
    7.5,   // "NYSE_HPE",
    7.5,   // "NASDAQ_BL",
    7.5,   // "NYSE_FND",
    7.5,   // "AMEX_DGRO",
    7.5,   // "NASDAQ_DDOG",
    7.5,   // "NASDAQ_FIVE",
    7.5,   // "NASDAQ_GOOG",
    7.5,   // "NYSE_DOW",
    7.5,   // "NYSE_FTV",
    7.5,   // "NYSE_DAY",
    7.5,   // "NASDAQ_MCHI",
    7.5,   // "NYSE_SNAP",
    7.5,   // "NASDAQ_PGNY",
    7.5,   // "NYSE_TDOC",
    7.5,   // "NASDAQ_HQY",
    7.5,   // "NASDAQ_TXG",
    7.5,   // "NASDAQ_TRIP",
    7.5,   // "NASDAQ_FOX",
    7.5,   // "NYSE_QTWO",
    7.5,   // "NASDAQ_ETSY",
    7.5,   // "NYSE_USFD",
    7.5,   // "AMEX_HDV",
    7.5,   // "NASDAQ_NWSA",
    7.5,   // "NYSE_PLNT",
    7.5,   // "NYSE_VEEV",
    7.5,   // "CBOE_QUAL",
    7.5,   // "AMEX_FTEC",
    7.5,   // "NASDAQ_OLLI",
    7.5,   // "NYSE_INSP",
    7.5,   // "NYSE_CVNA",
    7.5,   // "NYSE_HLT",
    7.5,   // "NASDAQ_LAZR",
    7.5,   // "NYSE_PFGC",
    7.5,   // "NASDAQ_EXPI",
];

pub fn process_ts_3m() -> Result<()> {
    let assets = BIN_PATH;
    let assets = std::fs::read_dir(assets)?.collect::<Result<Vec<_>, _>>()?;
    for asset_path in assets {
        let asset_path = asset_path.path();
        with_paths! {
            ts_3m_path = asset_path / "ts_3M.bin",
            ts_d_path = asset_path / "ts_d.bin",
        }
        let ts_d = Reader::<u16>::open(ts_d_path)?;
        let mut encoder = BinaryEncoder::new(ts_3m_path)?;
        let ts_d = ts_d.slice();
        for ts_d in ts_d {
            let date = START_DATE + Duration::days(*ts_d as i64);
            let mut month = date.month() as i32;
            let mut dur_y = date.year() - START_YEAR as i32;
            let mut ts_3m = (dur_y * 4 + (month - 1) / 3) as u16;
            encoder.push(ts_3m)?;
        }
    }
    Ok(())
}
