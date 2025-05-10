use crate::hlc::Hlc;

pub struct ATR {
    true_range: [f32; 14],
    sum_true_range: f32,
}

impl ATR {
    pub fn new() -> Self {
        Self {
            true_range: [0.0; 14],
            sum_true_range: 0.0,
        }
    }

    pub fn update(&mut self, prev_c: f32, hlc: &Hlc, i: u32) -> f32 {
        let true_range = (hlc.h - hlc.l).max(hlc.h - prev_c).abs().max(prev_c - hlc.l).abs();
        let i = (i % 14) as usize;
        self.sum_true_range -= self.true_range[i];
        self.true_range[i] = true_range;
        self.sum_true_range += true_range;
        let atr = self.sum_true_range / 14. * 0.5;
        atr
    }
}
