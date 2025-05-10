use std::fs::File;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::path::Path;

use eyre::Result;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;
use parquet::file::reader::FileReader as _;
use parquet::file::serialized_reader::SerializedFileReader;
use parquet::file::writer::SerializedFileWriter;
use parquet::record::{RecordReader as _, RecordWriter as _};

pub struct ParquetEncoder<T>
where
    for<'a> &'a [T]: ::parquet::record::RecordWriter<T>,
{
    writer: ManuallyDrop<SerializedFileWriter<File>>,
    row_group: Vec<T>,
}

impl<T> ParquetEncoder<T>
where
    for<'a> &'a [T]: ::parquet::record::RecordWriter<T>,
{
    #[inline(always)]
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let row_group = Vec::<T>::with_capacity(ROW_GROUP_SIZE_BYTES / core::mem::size_of::<T>());
        let schema = row_group.as_slice().schema()?;
        let file = File::create(path)?;
        let properties = std::sync::Arc::new(
            WriterProperties::builder()
                .set_compression(Compression::ZSTD(ZstdLevel::try_new(COMPRESSION_LEVEL)?))
                .build(),
        );
        let writer = ManuallyDrop::new(SerializedFileWriter::new(file, schema, properties)?);
        Ok(Self { writer, row_group })
    }

    #[inline(always)]
    pub fn write(&mut self, entry: impl Into<T>) -> Result<()> {
        if self.row_group.len() == self.row_group.capacity() {
            let mut row_group = self.writer.next_row_group()?;
            self.row_group.as_slice().write_to_row_group(&mut row_group)?;
            row_group.close()?;
            self.row_group.clear();
        }
        self.row_group.push(entry.into());
        Ok(())
    }
}

impl<T> Drop for ParquetEncoder<T>
where
    for<'a> &'a [T]: ::parquet::record::RecordWriter<T>,
{
    fn drop(&mut self) {
        if !self.row_group.is_empty() {
            let mut row_group = self.writer.next_row_group().unwrap();
            self.row_group.as_slice().write_to_row_group(&mut row_group).unwrap();
            row_group.close().unwrap();
        }
        unsafe {
            let writer = core::ptr::read(&self.writer);
            ManuallyDrop::into_inner(writer).close().unwrap();
        }
    }
}

pub struct ParquetDecoder<T> {
    reader: SerializedFileReader<File>,
    i: usize,
    _t: PhantomData<T>,
}

impl<T> ParquetDecoder<T>
where
    Vec<T>: ::parquet::record::RecordReader<T>,
{
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let file = File::open(path)?;
        let reader = SerializedFileReader::new(file)?;
        Ok(Self {
            reader,
            _t: PhantomData,
            i: 0,
        })
    }

    pub fn read_group_into(&mut self, data: &mut Vec<T>) -> Result<Option<()>> {
        if self.i == self.reader.num_row_groups() {
            return Ok(None);
        }
        let mut row_group = self.reader.get_row_group(self.i)?;
        let n_rows = row_group.metadata().num_rows() as usize;
        data.read_from_row_group(&mut *row_group, n_rows)?;
        self.i += 1;
        Ok(Some(()))
    }

    pub fn read_group(&mut self) -> Result<Option<Vec<T>>> {
        if self.i == self.reader.num_row_groups() {
            return Ok(None);
        }
        let mut data = Vec::<T>::with_capacity(ROW_GROUP_SIZE_BYTES / core::mem::size_of::<T>());
        self.read_group_into(&mut data)?;
        Ok(Some(data))
    }
}

impl<T: Copy> IntoIterator for ParquetDecoder<T>
where
    Vec<T>: ::parquet::record::RecordReader<T>,
{
    type IntoIter = ParquetDecoderIterator<T>;
    type Item = T;

    fn into_iter(self) -> Self::IntoIter {
        ParquetDecoderIterator {
            decoder: self,
            data: Vec::new(),
            i: 0,
        }
    }
}

pub struct ParquetDecoderIterator<T> {
    decoder: ParquetDecoder<T>,
    data: Vec<T>,
    i: usize,
}

impl<T: Copy> Iterator for ParquetDecoderIterator<T>
where
    Vec<T>: ::parquet::record::RecordReader<T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.i == self.data.len() {
            self.data.clear();
            self.i = 0;
            self.decoder.read_group_into(&mut self.data).ok()?;
            if self.data.is_empty() {
                return None;
            }
        }

        let item = unsafe { *self.data.get_unchecked(self.i) };
        self.i += 1;
        Some(item)
    }
}

pub const COMPRESSION_LEVEL: i32 = 2;
pub const ROW_GROUP_SIZE_BYTES: usize = 1024 * 1024;
