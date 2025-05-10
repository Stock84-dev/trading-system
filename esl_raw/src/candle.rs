pub struct Hlc {
    pub h: f32,
    pub l: f32,
    pub c: f32,
}

struct HlcBuilderFromL1 {
    timeframe_ms: i64,
    close_ts_ms: i64,
    h: f32,
    l: f32,
    c: f32,
}

impl HlcBuilderFromL1 {
    #[inline]
    pub fn new(timeframe_ms: u32, ts_ms: i64, bid: f32, ask: f32, last_ts_ms: i64) -> Self {
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
        // CTX.debug(0, 1, ts_ms as u64 * 1_000_000).add((
        //     lines!(bid, 0, 0),
        //     lines!(ask, 0, 0),
        // ));
        unsafe {
            if !(ts_ms > self.close_ts_ms) {
                self.h.max_mut(ask);
                self.l.min_mut(bid);
                self.c = (bid + ask) * 0.5;
            } else {
                while ts_ms > self.close_ts_ms {
                    let h = self.features.h.last().copied().unwrap_unchecked();
                    let l = self.features.l.last().copied().unwrap_unchecked();
                    let c = self.features.c.last().copied().unwrap_unchecked();
                    self.features.h.push(c);
                    self.features.l.push(c);
                    self.features.c.push(c);
                    *self.features.atr_05_14.last_mut().unwrap_unchecked() = atr;
                    self.features.atr_05_14.push(0.);
                    self.close_ts_ms += self.timeframe_ms;
                }
                *self.features.h.last_mut().unwrap_unchecked() = ask;
                *self.features.l.last_mut().unwrap_unchecked() = bid;
                *self.features.c.last_mut().unwrap_unchecked() = (bid + ask) * 0.5;
            }
        }
    }
}

struct HlcBuilderFromL1Iterator<'a> {
    builder: &'a mut HlcBuilderFromL1,
    ts_ms: i64,
    ask: f32,
    bid: f32,
}

impl<'a> Iterator for HlcBuilderFromL1Iterator<'a> {
    type Item = Hlc;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.ts_ms > self.close_ts_ms {
            let h = self.features.h.last().copied().unwrap_unchecked();
            let l = self.features.l.last().copied().unwrap_unchecked();
            let c = self.features.c.last().copied().unwrap_unchecked();
            self.features.h.push(c);
            self.features.l.push(c);
            self.features.c.push(c);
            *self.features.atr_05_14.last_mut().unwrap_unchecked() = atr;
            self.features.atr_05_14.push(0.);
            self.close_ts_ms += self.timeframe_ms;
        } else {
            None
        }
    }
}

impl<'a> Drop for HlcBuilderFromL1Iterator<'a> {
    fn drop(&mut self) {
        self.builder.h = self.ask;
        self.builder.l = self.bid;
        self.builder.c = (self.bid + self.ask) * 0.5;
    }
}

