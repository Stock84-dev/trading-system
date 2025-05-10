use ergnomics::*;

#[derive(Debug)]
pub struct Hlc {
    pub h: f32,
    pub l: f32,
    pub c: f32,
}

pub struct HlcBuilderFromL1 {
    timeframe_ms: i64,
    pub close_ts_ms: i64,
    h: f32,
    l: f32,
    c: f32,
}

impl HlcBuilderFromL1 {
    #[inline]
    pub fn new(timeframe_ms: u32, ts_ms: i64, bid: f32, ask: f32) -> Self {
        let timeframe_ms = timeframe_ms as i64;
        let close_ts_ms = ts_ms - ts_ms % timeframe_ms + timeframe_ms;
        Self {
            timeframe_ms,
            close_ts_ms,
            h: ask,
            l: bid,
            c: (bid + ask) * 0.5,
        }
    }

    #[inline]
    pub fn update(&mut self, ts_ms: i64, bid: f32, ask: f32) -> HlcBuilderFromL1Iterator {
        if !(ts_ms > self.close_ts_ms) {
            self.h.max_mut(ask);
            self.l.min_mut(bid);
            self.c = (bid + ask) * 0.5;
            HlcBuilderFromL1Iterator { inner: None }
        } else {
            HlcBuilderFromL1Iterator {
                inner: Some(HlcBuilderFromL1IteratorInner {
                    builder: self,
                    ts_ms,
                    ask,
                    bid,
                }),
            }
        }
    }
}

pub struct HlcBuilderFromL1Iterator<'a> {
    inner: Option<HlcBuilderFromL1IteratorInner<'a>>,
}

impl<'a> Iterator for HlcBuilderFromL1Iterator<'a> {
    type Item = Hlc;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.as_mut().map(|x| x.next()).flatten()
    }
}

struct HlcBuilderFromL1IteratorInner<'a> {
    builder: &'a mut HlcBuilderFromL1,
    ts_ms: i64,
    ask: f32,
    bid: f32,
}

impl<'a> Iterator for HlcBuilderFromL1IteratorInner<'a> {
    type Item = Hlc;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ts_ms > self.builder.close_ts_ms {
            dbg!(self.builder.close_ts_ms);
            self.builder.close_ts_ms += self.builder.timeframe_ms;
            Some(Hlc {
                h: self.builder.h,
                l: self.builder.l,
                c: self.builder.c,
            })
        } else {
            None
        }
    }
}

impl<'a> Drop for HlcBuilderFromL1IteratorInner<'a> {
    fn drop(&mut self) {
        self.builder.h = self.ask;
        self.builder.l = self.bid;
        self.builder.c = (self.bid + self.ask) * 0.5;
    }
}
