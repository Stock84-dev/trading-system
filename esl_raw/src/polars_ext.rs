use eyre::Result;

pub fn extract_bitset<'a>(
    df: &'a polars::prelude::DataFrame,
    column: &str,
) -> Result<impl Iterator<Item = &'a [u8]>> {
    let series = df.column(column)?;
    let chunked_array = series.bool()?;
    Ok(chunked_array.chunks().iter().map(|x| {
        // SAFETY: Column got checked that is f32. Checked downcasting doesn't work when
        // interoperating betwwen python polars and and this crate because they are built with
        // different versions of a compiler so type ids are different.
        unsafe {
            x.as_any()
                .downcast_ref_unchecked::<arrow2::array::BooleanArray>()
                .values()
                .as_slice()
                .0
        }
    }))
}

macro_rules! define_extract_fn {
    ($($fn:ident),*) => {
        define_extract_fn!($($fn; $fn);*);
    };
    ($($fn:ident; $series:ident);*) => {
        $(
            paste! {
                pub fn [<extract_ $fn>]<'a>(
                    df: &'a polars::prelude::DataFrame,
                    column: &str
                ) -> Result<impl Iterator<Item = &'a [$series]>> {
                    let series = df.column(column)?;
                    let chunked_array = series.$fn()?;
                    Ok(chunked_array.chunks().iter().map(|x| {
                        // SAFETY: Column got checked that is f32. Checked downcasting doesn't work when
                        // interoperating betwwen python polars and and this crate because they are built with
                        // different versions of a compiler so type ids are different.
                        unsafe {
                            x.as_any()
                                .downcast_ref_unchecked::<arrow2::array::PrimitiveArray<$series>>()
                                .values()
                                .as_slice()
                        }
                    }))
                }
            }
        )*
    };
}

define_extract_fn!(u8, i8, i16, u16, i32, u32, i64, u64, f32, f64);
define_extract_fn!(datetime; i64);
