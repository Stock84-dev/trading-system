use ergnomics::*;
use rayon::prelude::*;

pub fn clamp_extreme_wicks(candles: &mut [Candle]) {
    let mut sum_deviations = 0.0;
    let mut n_deviations = 0;
    for c in &candles[..] {
        let max = c.open.max(c.close);
        let min = c.open.min(c.close);
        if c.high != max {
            sum_deviations += c.high / max;
            n_deviations += 1;
        }
        if c.low != min {
            sum_deviations += min / c.low;
            n_deviations += 1;
        }
    }
    let avg_deviation = sum_deviations / n_deviations as f32;
    let mut sum_sq = 0.;
    for c in &candles[..] {
        let max = c.open.max(c.close);
        let min = c.open.min(c.close);
        if c.high != max {
            sum_sq += (c.high / max - avg_deviation).abs();
            // sum_sq += (c.high / max - avg_deviation) * (c.high / max - avg_deviation);
        }
        if c.low != min {
            sum_sq += (min / c.low - avg_deviation).abs();
            // sum_sq += (min / c.low - avg_deviation) * (min / c.low - avg_deviation);
        }
    }
    // let std = (sum_sq / n_deviations as f32).sqrt();
    let std = (sum_sq / n_deviations as f32);
    for c in candles {
        let max = c.open.max(c.close);
        let min = c.open.min(c.close);
        c.high = c.high.min(max * (avg_deviation + std * 13.));
        c.low = c.low.max(min / (avg_deviation + std * 13.));
    }
}

pub fn replace_phantom_candles(candles: &mut [Candle], source: &[Candle]) {
    (1..candles.len()).into_par_iter().for_each(|i| {
        let c = candles[i];
        let slice = unsafe { candles.cast_mut() };
        if c.open == c.close && c.high == c.close && c.low == c.close {
            replace_phantom_candle(slice.as_slice(), i, source);
        }
    });
}

pub fn impute_sequence_of_identity_candles(in_candles: &mut [Candle], source: &[Candle]) {
    (1..in_candles.len()).into_par_iter().for_each(|i| {
        let mut true_source_i = 0;
        let mut in_candles = unsafe { in_candles.cast_mut::<Candle>() };
        let close = in_candles[i].close;
        if in_candles[i].close == in_candles[i].open
            && in_candles[i].close == in_candles[i].high
            && in_candles[i].close == in_candles[i].low
        {
            let mut j = i - 1;
            while j != 0 {
                if !(in_candles[j].close == in_candles[j].open
                    && in_candles[j].close == in_candles[j].high
                    && in_candles[j].close == in_candles[j].low)
                    || close != in_candles[j].close
                {
                    break;
                }
                j -= 1;
            }
            if i - j < 3 {
                return;
            }
            if true_source_i == 0 {
                // if asset == "FX_CADCHF" {
                //     let date = START_DATE + Duration::days(in_candles[i].ts_d as i64);
                //     dbg!(date);
                // }
                true_source_i = i;
            }
            let start_close = in_candles[j].close;
            let end_i = i + 1;
            let end_close = in_candles[end_i].open;
            let n_candles = end_i - j + 1;
            let change = end_close / start_close;
            let mut best_candle_i = end_i;
            let mut best_score = f32::MAX;
            let true_source_i =
                (true_source_i as f32 / in_candles.len() as f32 * source.len() as f32) as usize;
            let mut range = true_source_i..source.len().saturating_sub(n_candles);
            range.end = range.end.min(true_source_i + 65536);
            range.start = range.start.saturating_sub(65536 - range.len());
            for k in range {
                let start_close = source[k].close;
                let end_close = source[k + n_candles].open;
                let change_candidate = end_close / start_close;
                let std = std_dev(&source[k..k + n_candles]) / start_close;
                // let std = 0.;
                let score = (change_candidate - change).abs() + std / 10.;
                if score < best_score {
                    best_score = score;
                    best_candle_i = k;
                }
            }
            let ref_close = source[best_candle_i].close;
            let c = in_candles[j].close;
            for offset in 1..n_candles - 1 {
                let change = source[best_candle_i + offset].open / ref_close;
                in_candles[j + offset].open = c * change;
                let change = source[best_candle_i + offset].high / ref_close;
                in_candles[j + offset].high = c * change;
                let change = source[best_candle_i + offset].low / ref_close;
                in_candles[j + offset].low = c * change;
                let change = source[best_candle_i + offset].close / ref_close;
                in_candles[j + offset].close = c * change;
                in_candles[j + offset].modified = true;
            }
            let mut end_gap =
                in_candles[j + n_candles - 1].open - in_candles[j + n_candles - 2].close;
            let start_gap = in_candles[j].close - in_candles[j + 1].open;
            let total_distance = (n_candles - 2) as f32;
            for offset in 1..n_candles - 1 {
                let distance = (offset - 1) as f32 / total_distance;
                let gap = end_gap * distance + start_gap * (1. - distance);
                in_candles[j + offset].open += gap;
                in_candles[j + offset].high += gap;
                in_candles[j + offset].low += gap;
                in_candles[j + offset].close += gap;
            }
            for offset in 2..n_candles {
                let close = in_candles[j + offset].open;
                in_candles[j + offset - 1].close = close;
                in_candles[j + offset - 1].high.max_mut(close);
                in_candles[j + offset - 1].low.min_mut(close);
            }
        }
    });
}

pub fn std_dev(x: &[Candle]) -> f32 {
    let n_samples = x.len() as f32;
    let mut sum = 0.0;
    for &v in x {
        sum += v.close;
    }
    let mean = sum / n_samples;
    let mut sum_squared = 0.0;
    for &v in x {
        let delta = v.close - mean;
        sum_squared += delta * delta;
    }
    (sum_squared / n_samples).sqrt()
}

/// Generates a random candle that closely matches to the previous path between prev and next
/// candle.
pub fn replace_phantom_candle(candles: &mut [Candle], i: usize, source: &[Candle]) {
    let prev_c = candles[i.saturating_sub(1)];
    let next_c = candles[(i + 1).min(candles.len() - 1)];
    let target_change = next_c.open / prev_c.close;
    let mut best_candle_i = i;
    let mut best_change = f32::MAX;
    let start_i = (i as f32 / candles.len() as f32 * source.len() as f32) as usize;
    let mut range = start_i + 1..source.len() - 1;
    range.end = range.end.min(start_i + 65536);
    range.start = range.start.saturating_sub(65536 - range.len()).max(1);

    for k in range {
        if source[k].close == source[k].open
            && source[k].close == source[k].high
            && source[k].close == source[k].low
            || source[k].modified
        {
            continue;
        }
        let change = source[k + 1].open / source[k - 1].close;
        if (target_change - change).abs() < best_change {
            best_change = (target_change - change).abs();
            best_candle_i = k;
        }
    }
    let ref_close = source[best_candle_i].open;
    let ref_c = prev_c.close;
    candles[i].open = prev_c.close;
    // candles[i].open = ref_c * source[best_candle_i].open / ref_close;
    candles[i].high = ref_c * source[best_candle_i].high / ref_close;
    candles[i].low = ref_c * source[best_candle_i].low / ref_close;
    // candles[i].close = ref_c * source[best_candle_i].close / ref_close;
    candles[i].close = next_c.open;
    let max = candles[i].open.max(candles[i].close);
    let min = candles[i].open.min(candles[i].close);
    candles[i].high.max_mut(max);
    candles[i].low.min_mut(min);
    candles[i].modified = true;
}

pub fn replace_gapping_candle(candles: &mut [Candle], i: usize) {
    let target_change = candles[i].open / candles[i - 1].close;
    let mut best_candle_i = i + 1;
    let mut best_change = f32::MAX;
    for k in i + 1..candles.len() {
        if candles[k].close == candles[k].open
            && candles[k].close == candles[k].high
            && candles[k].close == candles[k].low
        {
            continue;
        }
        // avoiding gaps
        let change = candles[k].close / candles[k].open;
        if (target_change - change).abs() < best_change {
            best_change = (target_change - change).abs();
            best_candle_i = k;
        }
    }
    let ref_close = candles[best_candle_i].open;
    let ref_c = candles[i - 1].close;
    candles[i].open = ref_c * candles[best_candle_i].open / ref_close;
    candles[i].high = ref_c * candles[best_candle_i].high / ref_close;
    candles[i].low = ref_c * candles[best_candle_i].low / ref_close;
    candles[i].close = ref_c * candles[best_candle_i].close / ref_close;
    candles[i].modified = true;
}

#[derive(Clone, Copy)]
#[repr(C)]
pub struct Candle {
    pub ts_d: u16,
    pub modified: bool,
    pub ts_s: i32,
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
    pub volume: f32,
}
