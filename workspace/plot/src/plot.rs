#![feature(lazy_cell)]
use std::sync::LazyLock;

use chrono::{Datelike, Duration, NaiveDate};
use ergnomics::*;
use esl_gpu::*;
use esl_raw::*;
use eyre::Result;
use path_no_alloc::with_paths;
static CTX: LazyLock<DebugContext<1>> = LazyLock::new(|| DebugContext::default());
const START_DATE: NaiveDate = NaiveDate::from_ymd_opt(1896, 1, 1).unwrap();

fn main() -> Result<()> {
    // Used for storign plotting data
    let debug_streams = CTX.debug_streams().clone();
    with_paths! {
        // BIN_PATH points to a directory containing data, modify the file paths as needed
        path = BIN_PATH / "NYSE_AVY",
        open = path / "open.bin",
        high = path / "high.bin",
        low = path / "low.bin",
        close = path / "close.bin",
        volume = path / "volume.bin",
        ts_s = path / "ts_s.bin",
    }
    let open_reader = Reader::<f32>::open(open)?;
    let high_reader = Reader::<f32>::open(high)?;
    let low_reader = Reader::<f32>::open(low)?;
    let close_reader = Reader::<f32>::open(close)?;
    let ts_s_reader = Reader::<i32>::open(ts_s)?;
    let volume_reader = Reader::<f32>::open(volume)?;
    let open = &open_reader.slice();
    let high = &high_reader.slice();
    let low = &low_reader.slice();
    let close = &close_reader.slice();
    let ts_s = &ts_s_reader.slice();
    let volume = &volume_reader.slice();

    for i in 0..open.len() {
        let timestamp_ns = ts_s[i] as i64 * 1_000_000_000;
        let price = (open[i], high[i], low[i], close[i]);

        CTX.debug(0, 1, timestamp_ns)
            .add((candles!(price, 0, 0), bars!(volume[i], 0, 1)));
    }

    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default(),
        persist_window: true,
        ..Default::default()
    };
    let plot = niobe::Plot::from_debug_streams(debug_streams, 0);
    eframe::run_native("Plot", options, Box::new(|_cc| Box::new(MyApp { plot }))).unwrap();
    Ok(())
}

struct MyApp {
    plot: niobe::Plot,
}

impl eframe::App for MyApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
            self.plot.show(ui);
        });
    }
}
