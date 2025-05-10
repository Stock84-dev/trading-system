#![feature(lazy_cell)]

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ops::{Range, RangeInclusive};
use std::rc::Rc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;

use candles::CandlesWidget;
use chrono::{Duration, DurationRound, TimeDelta};
use egui::{Context, Event, Id, LayerId, Order, Painter, PointerButton, TextStyle, Ui};
use egui_plot::{
    AxisHints, CoordinatesFormatter, Corner, GridInput, GridMark, Legend, PlotBounds, PlotPoint,
    PlotTransform, PlotUi,
};
use emath::{Align2, Pos2, Rect, Vec2, Vec2b};
use enum_dispatch::enum_dispatch;
use ergnomics::*;
use esl_gpu::{
    Bounds, CandlesOptions, DebugStreamData, DebugStreams, LinesOptions, PlotItem, PlotItemData,
    PlotItemOptionsRef, PointsOptions, TextsOptions, VerticalBarsOptions,
};
use generic_series::GenericSeries;
use lines::LinesWidget;
use points::PointsWidget;
use sync::{Arc, Mutex, lock, read};
use texts::TextsWidget;
use vertical_bars::VerticalBarsWidget;

mod candles;
mod generic_series;
mod lines;
mod points;
mod texts;
mod vertical_bars;
// pub mod debug;

pub struct Plot {
    subplots: Vec<SubPlot>,
    inner: PlotInner,
    stream_state: Arc<Mutex<PlotStreamState>>,
    click_events: VecDeque<i64>,
    streams: DebugStreams,
    account_id: usize,
    bounds_frames: u8,
    modified: bool,
    resetted: bool,
}

struct PlotStreamState {
    appended: bool,
    ctx: Option<Context>,
    max_compute_duration_ms: u32,
    modified: bool,
}

struct PlotInner {
    x_grid_step: Rc<RefCell<i64>>,
}

const BOUNDS_FRAMES: u8 = 255;

/// API usage:
///
/// UI usage:
/// mouse scroll to zoom in/out
/// mouse drag to pan
/// double click to reset (might not work on remote desktop)
/// press `r` to reset if mouse over plot
/// right click drag zoom to area of interest
///
/// Bugs: `egui_plot` doesn't display all values when the name of multiple series is the same
impl Plot {
    pub fn from_debug_streams(debug_streams: DebugStreams, account_id: usize) -> Self {
        let state = Arc::new(Mutex::new(
            PlotStreamState {
                appended: false,
                ctx: None,
                max_compute_duration_ms: 1,
                modified: false,
            },
            || "plot",
        ));
        let state2 = state.clone();
        let debug_streams2 = debug_streams.clone();
        std::thread::spawn(move || {
            // return;
            let mut lens: Vec<Vec<usize>> = vec![];
            let mut sleep_ms = 1;
            loop {
                if Arc::count(&state2) == 1 {
                    return;
                }
                let streams = read!(debug_streams2);
                let mut reset = false;
                if streams.streams.len() != lens.len() {
                    lens.clear();
                    lens.extend(
                        streams.streams.iter().map(|x| x.iter().map(|x| x.ts_ns.len()).collect()),
                    );
                    reset = true;
                }
                for (stream, lens) in streams.streams.iter().zip(&mut lens) {
                    if stream.len() != lens.len() {
                        reset = true;
                        lens.clear();
                        lens.extend(stream.iter().map(|x| x.ts_ns.len()));
                    }
                    for (item, len) in stream.iter().zip(lens.iter_mut()) {
                        if item.ts_ns.len() != *len {
                            reset = true;
                            *len = item.ts_ns.len();
                        }
                    }
                }
                drop(streams);
                if reset {
                    let mut state = lock!(state2);
                    state.appended = true;
                    state.ctx.on_ref(|x| x.request_repaint());
                    // sleep_ms = (state.max_compute_duration_ms as f32 * 1.) as u64;
                    sleep_ms = (state.max_compute_duration_ms as f32 * 10.) as u64;
                }
                std::thread::sleep(std::time::Duration::from_millis(sleep_ms));
            }
        });
        Self {
            subplots: vec![],
            inner: PlotInner {
                x_grid_step: Rc::new(RefCell::new(1)),
            },
            click_events: VecDeque::new(),
            streams: debug_streams,
            account_id,
            stream_state: state,
            bounds_frames: BOUNDS_FRAMES,
            modified: false,
            resetted: false,
        }
    }

    pub fn show(&mut self, ui: &mut Ui) {
        {
            let account_streams = read!(self.streams);
            let streams = &account_streams.streams[self.account_id];
            for (plot_id, plot) in self.subplots.iter_mut().enumerate() {
                // remove all streams that are not in the debug_streams
                plot.fields.retain(|field| {
                    streams
                        .iter()
                        .any(|stream| stream.items.iter().any(|x| field.equals_item(x)))
                })
            }
            for (stream_id, stream) in streams.iter().enumerate() {
                for (field_id, field) in stream.items.iter().enumerate() {
                    let subplot = match self.subplots.iter_mut().find(|x| x.id == field.plot) {
                        Some(x) => x,
                        None => {
                            self.subplots.push(SubPlot::new(field.plot, vec![]));
                            self.subplots.last_mut().unwrap()
                        },
                    };
                    if subplot.fields.iter().any(|x| x.equals_item(field)) {
                        continue;
                    }
                    macro_rules! build_fields {
                    ($($variant:ident),*) => {
                        match &field.data {
                            $(
                                PlotItemData::$variant { options, .. } => Field {
                                    series: Series::$variant(GenericSeries::new(
                                        self.streams.clone(),
                                        paste::paste!([<$variant Widget>]),
                                        self.account_id,
                                        stream_id,
                                        field_id,
                                        options.clone(),
                                    )),
                                    name: field.name.clone(),
                                },
                            )*
                        }
                    };
                }
                    subplot
                        .fields
                        .push(build_fields!(Lines, Points, VerticalBars, Candles, Texts));
                }
            }
            drop(account_streams);
        }
        self.subplots.sort_by_key(|x| x.id);

        let mut scroll = Vec2::ZERO;
        ui.input(|i| {
            for event in &i.events {
                if let Event::MouseWheel {
                    unit: _,
                    delta,
                    modifiers: _,
                } = event
                {
                    scroll += *delta;
                }
            }
        });

        let x_axis_height = 12.;
        let sub_plot_height = (ui.available_height() - x_axis_height) / self.subplots.len() as f32;
        let sub_plot_height = sub_plot_height - ui.spacing().item_spacing.y;
        ui.vertical(|ui| {
            let bounds;
            let appended = {
                let mut state = lock!(self.stream_state);
                state.modified = self.modified;
                let reset = state.appended;
                state.ctx = Some(ui.ctx().clone());
                state.appended = false;
                drop(state);
                if !self.modified {
                    let streams = read!(self.streams);
                    bounds = streams.bounds_ns.clone();
                    // bounds = match &streams.bounds_ns {
                    //     Bounds::Range(range) => {
                    //         let range = range.clone();
                    //         if self.bounds_frames == 0 {
                    //             RwLockUpgradableReadGuard::upgrade(streams).bounds_ns =
                    //                 Bounds::WasRange(range.clone());
                    //             self.bounds_frames = BOUNDS_FRAMES;
                    //             Bounds::WasRange(range)
                    //         } else {
                    //             self.bounds_frames -= 1;
                    //             Bounds::Range(range)
                    //         }
                    //     },
                    //     x => x.clone(),
                    // };
                } else {
                    bounds = Bounds::Auto;
                }
                if reset {
                    ui.ctx().request_repaint();
                }
                reset
            };
            let mut clicked = false;
            if !self.modified && appended {
                for sub_plot in &mut self.subplots {
                    for field in &mut sub_plot.fields {
                        field.series.append(ui.ctx(), &self.stream_state);
                    }
                }
            }
            let mut args = SubPlotArgs {
                inner: &self.inner,
                id: ui.id().with("plot"),
                height: sub_plot_height,
                last: false,
                n_plots: self.subplots.len(),
                scroll,
                bounds,
                clicked: &mut clicked,
                ui,
                modified: &mut self.modified,
                reset: false,
                resetted: self.resetted,
            };
            for i in 0..self.subplots.len().saturating_sub(1) {
                self.subplots[i].show(&mut args, &self.stream_state);
            }
            if let Some(sub_plot) = self.subplots.last_mut() {
                args.last = true;
                sub_plot.show(&mut args, &self.stream_state);
            }
            // Remote desktop isn't sending double click events correctly, checking for them
            // manually
            let now = chrono::Utc::now().timestamp_nanos_opt().unwrap();
            while let Some(ts) = self.click_events.front() {
                if now - ts > 200_000_000 {
                    self.click_events.pop_front();
                } else {
                    break;
                }
            }
            // borrow checker
            let mut reset = args.reset;
            self.modified |= clicked || scroll != Vec2::ZERO;
            if clicked {
                self.click_events.push_back(now);
                if self.click_events.len() >= 2 {
                    reset = true;
                }
            }
            if reset {
                self.modified = false;
                self.resetted = true;
                self.subplots.iter_mut().for_each(|x| {
                    x.fields
                        .iter_mut()
                        .for_each(|x| x.series.reset(ui.ctx(), &self.stream_state));
                });
            } else {
                self.resetted = false;
            }
        });
    }
}

struct SubPlotArgs<'a> {
    ui: &'a mut Ui,
    inner: &'a PlotInner,
    id: Id,
    height: f32,
    last: bool,
    n_plots: usize,
    scroll: Vec2,
    bounds: Bounds,
    clicked: &'a mut bool,
    modified: &'a mut bool,
    reset: bool,
    resetted: bool,
}

pub struct SubPlot {
    id: usize,
    fields: Vec<Field>,
}

impl SubPlot {
    pub fn new(id: usize, fields: Vec<Field>) -> Self {
        Self { id, fields }
    }

    pub fn show(&mut self, args: &mut SubPlotArgs, stream_state: &Arc<Mutex<PlotStreamState>>) {
        let x_grid_step = args.inner.x_grid_step.clone();
        let x_fmt = move |x: GridMark, _digits, _range: &RangeInclusive<f64>| {
            x_fmt(x.value as i64, *x_grid_step.borrow())
        };

        let x_grid_step = args.inner.x_grid_step.clone();
        let x_grid = move |input: GridInput| -> Vec<GridMark> {
            let mut marks = vec![];

            let (min, max) = input.bounds;
            let min = min.floor() as i64;
            let max = max.ceil() as i64;
            let range = max.saturating_sub(min);
            let n_grids = 10;
            let step = range / n_grids;
            let i = match STEPS.binary_search_by_key(&step, |x| *x) {
                Ok(i) => i,
                Err(i) => i,
            }
            .min(STEPS.len() - 1);
            let step = if range / STEPS[i] > n_grids {
                STEPS[(i + 1).min(STEPS.len() - 1)]
            } else {
                STEPS[i]
            };
            *x_grid_step.borrow_mut() = step;
            let dt = chrono::NaiveDateTime::from_timestamp_nanos(min as i64).unwrap();
            let Some(start) = dt
                .duration_trunc(Duration::nanoseconds(STEPS[i] as i64))
                .unwrap()
                .timestamp_nanos_opt()
            else {
                return marks;
            };

            for x in (start..=max).step_by(step as usize) {
                let dt = chrono::NaiveDateTime::from_timestamp_nanos(x as i64).unwrap();
                marks.push(GridMark {
                    value: x as f64,
                    step_size: step as f64,
                });
            }

            marks
        };

        let mut plot = egui_plot::Plot::new(self.id)
            .legend(Legend::default())
            .height(args.height)
            .x_axis_position(egui_plot::VPlacement::Bottom)
            .link_axis(args.id, true, false)
            .link_cursor(args.id, true, false)
            .y_axis_position(egui_plot::HPlacement::Right)
            .custom_x_axes(vec![
                AxisHints::new_x().formatter(x_fmt).placement(egui_plot::VPlacement::Bottom),
            ])
            .x_grid_spacer(x_grid)
            .label_formatter(|_, _| "".to_string())
            // .auto_bounds(Vec2b::new(false, true))
            .set_margin_fraction(Vec2::ZERO)
            // .allow_drag(Vec2b::new(true, false))
            .allow_drag(Vec2b::new(false, false))
            .allow_scroll(false)
            .allow_zoom(false);
        if args.last {
            plot = plot.show_axes(Vec2b::new(true, true));
        } else {
            plot = plot.show_axes(Vec2b::new(false, true));
        }
        let pos = args.ui.next_widget_position();
        let painter = args.ui.painter().with_clip_rect(Rect {
            min: pos,
            max: pos + Vec2::new(args.ui.available_width(), args.height),
        });
        let mut transform = PlotTransform::new(Rect::ZERO, PlotBounds::NOTHING, false, false);
        plot.show(args.ui, |plot_ui| {
            if args.scroll != Vec2::ZERO {
                let value = (args.scroll.y + args.scroll.x) / args.n_plots as f32;
                let bounds = plot_ui.plot_bounds();
                plot_ui.zoom_bounds(
                    Vec2::new((value * 0.15).exp(), 1.),
                    PlotPoint::new(bounds.max()[0], 0.),
                );
                plot_ui.set_auto_bounds(Vec2b::new(false, true));
                // plot_ui.set_auto_bounds(Vec2b::new(false, true));
            }
            let mut y_bounds = f64::MAX..f64::MIN;
            for series in &mut self.fields {
                series.series.show(plot_ui, stream_state, &args.bounds, &mut y_bounds);
            }
            if args.resetted {
                plot_ui.set_auto_bounds(Vec2b::new(true, true));
            }
            match &args.bounds {
                Bounds::Auto => {
                    // plot_ui.set_auto_bounds(Vec2b::new(true, true));
                },
                Bounds::Range(bounds) => {
                    plot_ui.set_plot_bounds(PlotBounds::from_min_max(
                        [bounds.start as f64, y_bounds.start],
                        [bounds.end as f64, y_bounds.end],
                    ));
                },
                // Bounds::WasRange(_) => {
                //     plot_ui.set_auto_bounds(Vec2b::new(false, true));
                // },
            }
            transform = *plot_ui.transform();
            let response = plot_ui.response();
            if response.contains_pointer()
                && plot_ui
                    .ctx()
                    .input_mut(|x| x.consume_key(egui::Modifiers::NONE, egui::Key::R))
            {
                args.reset = true;
            }
            if response.dragged_by(PointerButton::Primary) {
                let mut delta = -response.drag_delta();
                let dvalue_dpos = plot_ui.transform().dvalue_dpos();
                delta.x *= dvalue_dpos[0] as f32;
                delta.y *= dvalue_dpos[1] as f32;
                // if !allow_drag.x {
                //     delta.x = 0.0;
                // }
                // if !allow_drag.y {
                delta.y = 0.0;
                // }
                plot_ui.translate_bounds(delta);
                plot_ui.set_auto_bounds(Vec2b::new(false, true));
                *args.modified = true;
            }
            *args.clicked |= plot_ui.response().clicked();
        });
        let coordinates_corner = egui_plot::Corner::LeftTop;
        let padded_frame = transform.frame().shrink(4.0);
        let (_, mut position) = match coordinates_corner {
            Corner::LeftTop => (Align2::LEFT_TOP, padded_frame.left_top()),
            Corner::RightTop => (Align2::RIGHT_TOP, padded_frame.right_top()),
            Corner::LeftBottom => (Align2::LEFT_BOTTOM, padded_frame.left_bottom()),
            Corner::RightBottom => (Align2::RIGHT_BOTTOM, padded_frame.right_bottom()),
        };
        if let Some(pointer) = args.ui.ctx().pointer_latest_pos() {
            let coordinate = transform.value_from_position(pointer);
            let dt = chrono::NaiveDateTime::from_timestamp_nanos(coordinate.x as i64).unwrap();
            let text = format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%9f"));
            let font_id = TextStyle::Monospace.resolve(args.ui.style());
            position.x += painter
                .text(
                    position,
                    Align2::LEFT_TOP,
                    text,
                    font_id,
                    args.ui.visuals().text_color(),
                )
                .width();
        }
        for series in &mut self.fields {
            series.series.show_coordinates(args.ui, &painter, &mut position, &transform);
        }
    }
}

struct Field {
    series: Series,
    name: String,
}

impl Field {
    fn equals_item(&self, x: &PlotItem) -> bool {
        macro_rules! match_options {
            ($($variant:ident),*) => {
                match &x.data {
                    $(
                        PlotItemData::$variant { options, .. } => {
                            if let Series::$variant(x) = &self.series {
                                x.options == *options
                            } else {
                                false
                            }
                        },
                    )*
                }
            };
        }
        x.name == self.name && match_options!(Lines, Points, VerticalBars, Candles, Texts)
    }
}

#[enum_dispatch]
pub enum Series {
    Lines(GenericSeries<LinesWidget>),
    Points(GenericSeries<PointsWidget>),
    VerticalBars(GenericSeries<VerticalBarsWidget>),
    Candles(GenericSeries<CandlesWidget>),
    Texts(GenericSeries<TextsWidget>),
}

#[enum_dispatch(Series)]
trait SeriesTrait {
    fn reset(&mut self, ctx: &egui::Context, stream_state: &Arc<Mutex<PlotStreamState>>);
    fn append(&mut self, ctx: &Context, plot_state: &Arc<Mutex<PlotStreamState>>);
    fn show(
        &mut self,
        plot_ui: &mut PlotUi,
        plot_state: &Arc<Mutex<PlotStreamState>>,
        x_bounds: &Bounds,
        y_bounds: &mut Range<f64>,
    );
    fn show_coordinates(
        &mut self,
        ui: &mut Ui,
        painter: &Painter,
        position: &mut Pos2,
        transform: &PlotTransform,
    );
}

const MINS_PER_DAY: f64 = 24.0 * 60.0;
const MINS_PER_H: f64 = 60.0;

const N_STEPS: usize = 19;
const STEPS: [i64; N_STEPS] = [
    1i64, // ns
    10,
    100,
    1_000, // us
    10_000,
    100_000,
    1_000_000, // ms
    10_000_000,
    100_000_000,
    1_000_000_000,            // s
    10_000_000_000,           // 10s
    60_000_000_000,           // min
    600_000_000_000,          // 10 min
    3_600_000_000_000,        // h
    36_000_000_000_000,       // 10 h
    86_400_000_000_000,       // day
    7 * 86_400_000_000_000,   // week
    30 * 86_400_000_000_000,  // month
    365 * 86_400_000_000_000, // year
];
const STEP_DURATIONS: [Duration; N_STEPS] = [
    Duration::nanoseconds(1),
    Duration::nanoseconds(10),
    Duration::nanoseconds(100),
    Duration::microseconds(1),
    Duration::microseconds(10),
    Duration::microseconds(100),
    Duration::milliseconds(1),
    Duration::milliseconds(10),
    Duration::milliseconds(100),
    Duration::seconds(1),
    Duration::seconds(10),
    Duration::minutes(1),
    Duration::minutes(10),
    Duration::hours(1),
    Duration::hours(10),
    Duration::days(1),
    Duration::days(7),
    Duration::days(30),
    Duration::days(365),
];

// Using locks with rayon leads to deadlocks
static THREAD_POOL: LazyLock<threadpool::ThreadPool> =
    LazyLock::new(|| threadpool::Builder::new().thread_name("plot-thread".to_string()).build());

fn x_fmt(ts_ns: i64, step: i64) -> String {
    let duration = chrono::Duration::nanoseconds(step);
    let dt = chrono::NaiveDateTime::from_timestamp_nanos(ts_ns).unwrap();
    let dt = dt.duration_round(duration).unwrap();
    match step.abs() {
        1i64 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%9f")),
        10 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%9f")),
        100 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%9f")),
        1_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%6f")),
        10_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%6f")),
        100_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%6f")),
        1_000_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%3f")),
        10_000_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%3f")),
        100_000_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S.%3f")),
        1_000_000_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S")),
        10_000_000_000 => format!("{}", dt.format("%Y-%m-%d %H:%M:%S")),
        60_000_000_000 => format!("{}", dt.format("%Y-%m-%d %H:%M")),
        600_000_000_000 => format!("{}", dt.format("%Y-%m-%d %H:%M")),
        3_600_000_000_000 => format!("{}", dt.format("%Y-%m-%d %H")),
        36_000_000_000_000 => format!("{}", dt.format("%Y-%m-%d %H")),
        86_400_000_000_000 => format!("{}", dt.format("%Y-%m-%d")),
        604_800_000_000_000 => format!("{}", dt.format("%Y-%m-%d")),
        2_592_000_000_000_000 => format!("{}", dt.format("%Y-%m")),
        31_536_000_000_000_000 => format!("{}", dt.format("%Y")),
        _ => unreachable!(),
    }
}

fn convert_color(color: esl_gpu::Color) -> egui::Color32 {
    egui::Color32::from_rgba_premultiplied(color.0, color.1, color.2, color.3)
}
