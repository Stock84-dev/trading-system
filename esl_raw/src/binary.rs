use std::fs::File;
use std::io::{BufWriter, Write as _};
use std::marker::PhantomData;
use std::path::Path;

use eyre::Result;

pub struct BinaryEncoder<T> {
    writer: BufWriter<File>,
    _t: PhantomData<T>,
}

impl<T: Copy + 'static> BinaryEncoder<T> {
    #[inline(always)]
    pub fn from_file(file: File) -> Self {
        let writer = BufWriter::with_capacity(16 * 1024 * 1024, file);
        // let writer = BufWriter::new(File::create(path)?);
        Self {
            writer,
            _t: PhantomData,
        }
    }

    #[inline(always)]
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let writer = BufWriter::with_capacity(16 * 1024 * 1024, File::create(path)?);
        // let writer = BufWriter::new(File::create(path)?);
        Ok(Self {
            writer,
            _t: PhantomData,
        })
    }

    #[inline(always)]
    pub fn write(&mut self, data: &[T]) -> Result<()> {
        let len = std::mem::size_of_val(data);
        let data = unsafe { core::slice::from_raw_parts(data.as_ptr() as *const u8, len) };
        self.writer.write_all(data)?;
        Ok(())
    }

    #[inline(always)]
    pub fn push(&mut self, data: T) -> Result<()> {
        let data = unsafe {
            core::slice::from_raw_parts(&data as *const T as *const u8, core::mem::size_of::<T>())
        };
        self.writer.write_all(data)?;
        Ok(())
    }

    #[inline(always)]
    pub fn push_any<U: Copy>(&mut self, data: U) -> Result<()> {
        let data = unsafe {
            core::slice::from_raw_parts(&data as *const U as *const u8, core::mem::size_of::<U>())
        };
        self.writer.write_all(data)?;
        Ok(())
    }

    #[inline(always)]
    pub fn flush(&mut self) -> Result<()> {
        self.writer.flush()?;
        Ok(())
    }
}

impl<T> Drop for BinaryEncoder<T> {
    #[inline(always)]
    fn drop(&mut self) {
        self.writer.flush().unwrap();
    }
}
