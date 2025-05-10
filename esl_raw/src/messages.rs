use deepsize::DeepSizeOf;

pub enum Request {
    Get(Fn),
    GetLevel(Fn, u8),
    Invalidate(Fn),
    InvalidateLevel(Fn, u8),
}

pub enum Fn {
    Market(Market),
    WfaChunk(WfaChunk),
    CcvPath(CcvPath),
    Labels(Labels),
    YearlyOrdinal(YearlyOrdinal),
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy)]
pub struct Market {
    pub id: u16,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy)]
pub struct WfaChunk {
    pub end_ts_d: u16,
    pub duration_ts_d: u16,
    pub market: Market,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy)]
pub struct CcvPath {
    pub chunk: WfaChunk,
    pub path: u16,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy)]
pub struct Labels {
    pub path: CcvPath,
}

#[derive(Hash, PartialEq, Eq, DeepSizeOf, Clone, Copy)]
pub struct YearlyOrdinal {
    pub path: CcvPath,
}
