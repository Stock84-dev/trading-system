use std::ffi::CString;
use std::ops::Deref;
use std::pin::Pin;
use std::sync::Arc;

use async_bincode::tokio::{AsyncBincodeReader, AsyncBincodeWriter};
use async_bincode::{AsyncDestination, BincodeWriterFor};
use deepsize::DeepSizeOf;
use ergnomics::OptionExt;
use eyre::{Result, bail};
// use futures_util::{SinkExt as _, StreamExt as _};
// use tokio::io::{AsyncReadExt, AsyncWriteExt};
use futures::prelude::*;
use libc::{
    MAP_SHARED, O_CREAT, O_RDWR, PROT_READ, PROT_WRITE, c_void, ftruncate, mmap, munmap, off_t,
    shm_open, shm_unlink, size_t,
};
use serde::ser::SerializeStruct;
// use futures_io::{AsyncReadExt, AsyncWriteExt};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use tokio::net::unix::{OwnedReadHalf, OwnedWriteHalf};
use tokio::sync::Mutex;

pub const SOCKET_PATH: &str = "/tmp/server.sock";

pub trait Task: Serialize + for<'a> Deserialize<'a> {
    const ID: TaskId;
    type Output;
}
pub type TaskId = u8;

pub struct Client {
    reader: AsyncBincodeReader<OwnedReadHalf, Response>,
    writer: Arc<Mutex<AsyncBincodeWriter<OwnedWriteHalf, Request, AsyncDestination>>>,
}

impl Client {
    pub async fn connect() -> Result<Self> {
        let stream = tokio::net::UnixStream::connect(SOCKET_PATH).await?;
        let (reader, writer) = stream.into_split();
        Ok(Self {
            reader: AsyncBincodeReader::from(reader),
            writer: Arc::new(Mutex::new(AsyncBincodeWriter::from(writer).for_async())),
        })
    }

    pub async fn get<T: Task>(&mut self, task: T) -> Result<GetGuard<T::Output>> {
        let get = Request::Get(Key {
            id: T::ID,
            payload: bincode::serialize(&task)?.into(),
        });
        let mut guard = self.writer.lock().await;
        guard.send(get).await?;
        let response = self.reader.next().await.some()??;
        Ok(match response {
            Response::Get { shm_id, size } => unsafe {
                let shm_name = CString::new(format!("/server_{}", shm_id)).unwrap();
                let fd = shm_open(shm_name.as_ptr(), 0, 0o600);
                if fd < 0 {
                    bail!("failed to open shared memory");
                }
                let addr = mmap(std::ptr::null_mut(), size, PROT_READ, MAP_SHARED, fd, 0);
                if addr == libc::MAP_FAILED {
                    bail!("failed to mmap shared memory");
                }
                GetGuard {
                    writer: self.writer.clone(),
                    ptr: addr as usize,
                    shm_id,
                    len: size / std::mem::size_of::<T::Output>(),
                    _t: std::marker::PhantomData,
                }
            },
            Response::Err(err) => return Err(eyre::eyre!(err)),
        })
    }
}

pub struct GetGuard<T> {
    writer: Arc<Mutex<AsyncBincodeWriter<OwnedWriteHalf, Request, AsyncDestination>>>,
    ptr: usize,
    shm_id: u64,
    len: usize,
    _t: std::marker::PhantomData<T>,
}

impl<T> Deref for GetGuard<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr as *const T, self.len) }
    }
}

impl<T> Drop for GetGuard<T> {
    fn drop(&mut self) {
        unsafe {
            munmap(self.ptr as *mut c_void, self.len * std::mem::size_of::<T>());
            let shm_name = CString::new(format!("/server_{}", self.shm_id)).unwrap();
            shm_unlink(shm_name.as_ptr());
        }
        let handle = tokio::runtime::Handle::current();
        handle.block_on(async {
            let mut guard = self.writer.lock().await;
            let _ = guard
                .send(Request::Free {
                    shm_id: self.shm_id,
                })
                .await;
        });
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Response {
    Get { shm_id: u64, size: usize },
    Err(String),
}

#[derive(Serialize, Deserialize, Debug)]
pub enum Request {
    Get(Key),
    Invalidate(Key),
    Free { shm_id: u64 },
}

#[derive(Debug, Serialize, Deserialize, Eq, Hash, PartialEq, Clone)]
pub struct Key {
    pub id: TaskId,
    pub payload: Box<[u8]>,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Serialize, Deserialize, Debug)]
pub enum NodeKey {
    WfaChunk(WfaChunkKey),
    CcvPath(CcvPathKey),
    Labels(LabelsKey),
    YearlyOrdinal(YearlyOrdinalKey),
}

#[derive(Clone, DeepSizeOf, Serialize, Deserialize, Debug)]
pub enum NodeKind {
    WfaChunk,
    CcvPath,
    Labels,
    YearlyOrdinal,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy, Serialize, Deserialize, Debug)]
pub struct WfaChunkKey {
    pub market_id: u16,
    pub end_ts_d: u16,
    pub duration_ts_d: u16,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy, Serialize, Deserialize, Debug)]
pub struct CcvPathKey {
    pub chunk: WfaChunkKey,
    pub path: u16,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy, Serialize, Deserialize, Debug)]
pub struct LabelsKey {
    pub path: CcvPathKey,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy, Serialize, Deserialize, Debug)]
pub struct YearlyOrdinalKey {
    pub path: CcvPathKey,
}

#[derive(Clone, DeepSizeOf, Serialize, Deserialize, Debug)]
pub struct NodeValue {
    pub kind: NodeKind,
    pub descriptor: CacheDescriptor,
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct CacheDescriptor {
    pub size: u32,
    pub file_id: u32,
}

impl DeepSizeOf for CacheDescriptor {
    fn deep_size_of_children(&self, _: &mut deepsize::Context) -> usize {
        self.size as usize
    }
}

// Pin item
// Small and large values
// High throughput
// Low latency
// High hit rate
// Segmented shared memory space

// struct MyCache {
//     ghost_old: BlockedBloomFilter,
//     ghost_new: BlockedBloomFilter,
//     small: Fifo,
//     large: Fifo,
// }
//
// struct Fifo {
//     index: HashTable<K, u32>,
//     fifo: Vec<V>,
// }
//
// pub struct HashTable<K, V> {
//     ptr: usize,
//     len: usize,
//     phantom: PhantomData<(K, V)>,
// }
//
// impl HashTable {
//     pub fn get(&self, k: &K) -> Option<&V> {
//         let hash = k.hash();
//         let index = hash % self.ptr;
//         let mut index = index;
//         loop {
//             let entry = self.ptr[index];
//             if entry.key == k {
//                 return Some(&entry.value);
//             }
//             index = (index + 1) % self.ptr;
//             if index == hash % self.ptr {
//                 return None;
//             }
//         }
//     }
// }
//
// pub trait L1CacheTag {
//     const TAG: u8;
// }
//
// use foyer::CacheEntry;
//
// pub struct L1BucketU8U8 {
//     // Contains 1 byte hash
//     ghost: [u8; 17],
//     sk: [u8; 2],
//     sv: [u8; 2],
//     main_count: [u8; 7],
//     mk: [u8; 18],
//     mv: [u8; 18],
// }
//
// pub struct L1BucketU8U32 {
//     // Contains 1 byte hash
//     ghost: [u8; 9],
//     sk: [u8; 1],
//     sv: [u32; 1],
//     main_count: [u8; 2],
//     mk: [u8; 9],
//     mv: [u32; 9],
// }
//
// #[repr(packed)]
// pub struct L1BucketU32U32 {
//     // Contains 1 byte hash
//     ghost: [u8; 6],
//     sk: [u32; 1],
//     sv: [u32; 1],
//     main_count: [u8; 2],
//     mk: [u32; 6],
//     mv: [u32; 6],
// }
