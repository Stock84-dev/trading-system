use std::collections::VecDeque;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, Ordering};

use atomic_float::AtomicF32;
use chrono::Utc;
use eframe::egui::Context;
use egui::{Painter, TextStyle, Ui};
use egui_plot::{Corner, Line, PlotBounds, PlotPoint, PlotPoints, PlotTransform, PlotUi};
use emath::{Align2, Pos2, Rect};
use ergnomics::ConstLen;
use esl_gpu::{
    Bounds, Color, DebugStream, DebugStreamData, DebugStreams, PlotItem, PointsOptions, Primitive,
};
use float_pretty_print::PrettyPrintFloat;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use sync::{Arc, Mutex, MutexGuard, RwLock, lock, read};

use crate::{PlotStreamState, SeriesTrait, THREAD_POOL, convert_color};

pub trait SeriesWidget: Send + Sync + 'static {
    type Options: DebugStream;
    type Point: Point;
    type Fold: IntoIterator<Item = Self::Point> + ConstLen;
    fn fold(
        &self,
        x: &[i64],
        y: &[<Self::Options as DebugStream>::Primitive],
        item: &PlotItem,
        bounds: &Range<f64>,
        n_points: usize,
    ) -> Self::Fold;
    fn primitives<'a>(&self, item: &'a PlotItem)
    -> &'a [<Self::Options as DebugStream>::Primitive];
    fn convert(
        &self,
        item: &PlotItem,
        bounds: &Range<f64>,
        n_points: usize,
        x: i64,
        next_x: i64,
        primitive: &<Self::Options as DebugStream>::Primitive,
    ) -> Self::Point;
    fn appended(&self, points: &mut [Self::Point]) {}
    fn show(&self, plot_ui: &mut PlotUi, item: &PlotItem, data: Vec<Self::Point>);
    fn show_coordinates(
        &self,
        ui: &mut Ui,
        painter: &Painter,
        coordinates_position: &mut Pos2,
        transform: &PlotTransform,
        item: &PlotItem,
        point: &Self::Point,
    );
}

pub trait Point: Clone + Sync + Send {
    fn x(&self) -> f64;
    fn y(&self) -> f64;
}

struct PointsState<F: SeriesWidget> {
    lod_data: VecDeque<F::Point>,
    view_range: Range<usize>,
    lod_tmp: Vec<F::Point>,
    bounds: Option<Range<f64>>,
    prev_bounds: PlotBounds,
    lod_level: usize,
    _f: std::marker::PhantomData<F>,
}

struct PointsInner<F: SeriesWidget> {
    state: Mutex<PointsState<F>>,
    lod_data: Mutex<Vec<F::Point>>,
    job_running: AtomicBool,
    streams: DebugStreams,
    account_id: usize,
    stream_id: usize,
    field_id: usize,
    command: Mutex<Option<Command>>,
    widget: F,
}

enum Command {
    UpdateBounds(Range<f64>),
    Append,
    Reset,
}

pub struct GenericSeries<F: SeriesWidget> {
    inner: Arc<PointsInner<F>>,
    prev_bounds: Option<PlotBounds>,
    reset: u8,
    pub options: F::Options,
}

fn bounds_eq(bounds: PlotBounds, self_bounds: PlotBounds) -> bool {
    let width = self_bounds.width();
    let threshold = 0.1;
    // (bounds.min()[0] - self_bounds.min()[0]).abs() / width < threshold
    //     && (bounds.max()[0] - self_bounds.max()[0]).abs() / width < threshold
    false
}

impl<F: SeriesWidget> GenericSeries<F> {
    pub fn new(
        streams: DebugStreams,
        widget: F,
        account_id: usize,
        stream_id: usize,
        field_id: usize,
        options: F::Options,
    ) -> Self {
        Self {
            inner: Arc::new(PointsInner {
                lod_data: Mutex::new(Vec::new(), || format!("lod data {}", field_id)),
                state: Mutex::new(
                    PointsState {
                        view_range: Range::default(),
                        lod_tmp: Vec::new(),
                        bounds: None,
                        lod_level: 1,
                        lod_data: VecDeque::new(),
                        prev_bounds: PlotBounds::new_symmetrical(1.),
                        _f: std::marker::PhantomData,
                    },
                    || format!("series state {}", field_id),
                ),
                job_running: AtomicBool::default(),
                command: Mutex::new(None, || "command"),
                widget,
                field_id,
                streams,
                account_id,
                stream_id,
            }),
            prev_bounds: None,
            reset: 0,
            options,
        }
    }
}

impl<T: SeriesWidget> SeriesTrait for GenericSeries<T> {
    fn append(&mut self, ctx: &Context, plot_state: &Arc<Mutex<PlotStreamState>>) {
        self.reset = 2;
        lock!(self.inner.command).replace(Command::Append);
        if !self.inner.job_running.load(Ordering::Acquire) {
            self.inner.job_running.store(true, Ordering::Release);
            let ctx = ctx.clone();
            let inner = self.inner.clone();
            let plot_state = plot_state.clone();
            THREAD_POOL.execute(|| update(ctx, inner, plot_state));
        }
    }

    fn reset(&mut self, ctx: &Context, plot_state: &Arc<Mutex<PlotStreamState>>) {
        self.reset = 2;
        lock!(self.inner.command).replace(Command::Reset);
        if !self.inner.job_running.load(Ordering::Acquire) {
            self.inner.job_running.store(true, Ordering::Release);
            let ctx = ctx.clone();
            let inner = self.inner.clone();
            let plot_state = plot_state.clone();
            THREAD_POOL.execute(|| update(ctx, inner, plot_state));
        }
    }

    fn show(
        &mut self,
        plot_ui: &mut PlotUi,
        plot_state: &Arc<Mutex<PlotStreamState>>,
        x_bounds: &Bounds,
        y_bounds: &mut Range<f64>,
    ) {
        if !self.inner.job_running.load(Ordering::Acquire) {
            // Waiting for 2 frames to pass after resetting. Otherwise it will enter into an
            // infinite loop where it cycle between previous bounds and resseted bounds.
            self.reset = self.reset.saturating_sub(1);
            if self.reset == 0 {
                let bounds = plot_ui.plot_bounds();
                let now = std::time::Instant::now();
                let mut state = lock!(self.inner.state);
                loop {
                    match &mut self.prev_bounds {
                        Some(prev_bounds) => {
                            if *prev_bounds == bounds {
                                break;
                            } else {
                                *prev_bounds = bounds;
                            }
                        },
                        None => self.prev_bounds = Some(bounds),
                    }
                    match &state.bounds {
                        Some(self_bounds)
                            // if !bounds_eq(bounds, *self_bounds) 
                            => {},
                        None => {},
                        _ => break,
                    }
                    state.prev_bounds = bounds;
                    self.inner.job_running.store(true, Ordering::Release);
                    // println!("{:?}: spawning", Utc::now());
                    let ctx = plot_ui.ctx().clone();
                    let inner = self.inner.clone();
                    let plot_state = plot_state.clone();
                    lock!(self.inner.command)
                        .replace(Command::UpdateBounds(bounds.min()[0]..bounds.max()[0]));
                    THREAD_POOL.execute(move || update(ctx, inner, plot_state));
                    break;
                }
            }
            // println!("State Lock: {:?}", now.elapsed());
        }
        let now = std::time::Instant::now();
        let data = {
            let guard = lock!(self.inner.lod_data);
            guard.clone()
        };
        if let Bounds::Range(x_bounds) = x_bounds {
            data.iter().for_each(|x| {
                y_bounds.start = y_bounds.start.min(x.y());
                y_bounds.end = y_bounds.end.max(x.y());
            });
        }
        // println!("Lock: {:?}", now.elapsed());
        let guard = sync::read!(self.inner.streams);
        self.inner.widget.show(
            plot_ui,
            &guard.streams[self.inner.account_id][self.inner.stream_id].items[self.inner.field_id],
            data,
        );
    }

    fn show_coordinates(
        &mut self,
        ui: &mut Ui,
        painter: &Painter,
        coordinates_position: &mut Pos2,
        transform: &PlotTransform,
    ) {
        if let Some(pointer) = ui.ctx().pointer_latest_pos() {
            let data = lock!(self.inner.lod_data);
            if !data.is_empty() {
                let coordinate = transform.value_from_position(pointer);
                let i = match data
                    .binary_search_by_key(&OrderedFloat(coordinate.x), |x| OrderedFloat(x.x()))
                {
                    Ok(i) => i,
                    Err(i) => i,
                };
                let left = i.saturating_sub(1);
                let right = i.min(data.len() - 1);
                let value = if coordinate.x - data[left].x() < data[right].x() - coordinate.x {
                    &data[left]
                } else {
                    &data[right]
                };
                self.inner.widget.show_coordinates(
                    ui,
                    painter,
                    coordinates_position,
                    transform,
                    &sync::read!(self.inner.streams).streams[self.inner.account_id]
                        [self.inner.stream_id]
                        .items[self.inner.field_id],
                    value,
                );
            }
        }
    }
}

fn update<F: SeriesWidget>(
    ctx: Context,
    inner: Arc<PointsInner<F>>,
    plot_state: Arc<Mutex<PlotStreamState>>,
) {
    loop {
        let command = lock!(inner.command).take();
        match command {
            Some(Command::UpdateBounds(bounds)) => {
                let mut state = lock!(inner.state);
                update_bounds(&ctx, &inner, bounds, state, &plot_state, false)
            },
            Some(Command::Reset) => {
                lock!(inner.command).take();
                let mut state = lock!(inner.state);
                state.bounds = None;
                update_bounds(&ctx, &inner, 0.0..0.0, state, &plot_state, false);
            },
            Some(Command::Append) => {
                let modified = lock!(plot_state).modified;
                let mut state = lock!(inner.state);
                if !modified {
                    let guard = read!(inner.streams);
                    let bounds = match &guard.bounds_ns {
                        // Bounds::Range(bounds) | Bounds::WasRange(bounds) => bounds.start as
                        // f64..bounds.end as f64,
                        Bounds::Range(bounds) => bounds.start as f64..bounds.end as f64,
                        _ => {
                            let x = &guard.streams[inner.account_id][inner.stream_id].ts_ns;
                            let min_x = state.bounds.as_ref().map(|x| x.start).unwrap_or(0.);
                            let max_x = x.last().copied().unwrap_or(1) as f64;
                            min_x..max_x
                        },
                    };
                    drop(guard);
                    update_bounds(&ctx, &inner, bounds, state, &plot_state, true);
                } else {
                    update_bounds(
                        &ctx,
                        &inner,
                        state.bounds.clone().unwrap_or_default(),
                        state,
                        &plot_state,
                        true,
                    );
                }
            },
            None => break,
        };
    }
    inner.job_running.store(false, Ordering::Release);
    ctx.request_repaint();
}

fn update_bounds<T: SeriesWidget>(
    ctx: &Context,
    inner: &PointsInner<T>,
    bounds: Range<f64>,
    mut state: MutexGuard<PointsState<T>>,
    plot_state: &Mutex<PlotStreamState>,
    appended: bool,
) {
    macro_rules! lod_iter {
        (
            $guard:expr,
            $x:expr,
            $y:expr,
            $range:expr,
            $bounds:expr,
            $n_visible_points:expr,
            $lod_level:expr
        ) => {{
            let item = &$guard.streams[inner.account_id][inner.stream_id].items[inner.field_id];
            $x[$range.clone()]
                .par_chunks($lod_level)
                .zip($y[$range].par_chunks($lod_level))
                .flat_map_iter(|(x, y)| inner.widget.fold(x, y, item, $bounds, $n_visible_points))
        }};
    }
    macro_rules! iter {
        ($guard:expr, $x:expr, $y:expr, $range:expr, $bounds:expr, $n_visible_points:expr) => {{
            let item = &$guard.streams[inner.account_id][inner.stream_id].items[inner.field_id];
            let x_slice = &$x[$range.clone()];
            let start = x_slice.get(0).copied().unwrap_or(0);
            let end = x_slice.get(x_slice.len().saturating_sub(1)).copied().unwrap_or(start);
            let x_diff = ((end - start) as f64 / x_slice.len() as f64) as i64;
            (x_slice, &$y[$range]).into_par_iter().enumerate().map(move |(i, (x, y))| {
                let next_x = $x.get(i + 1).copied().unwrap_or(x + x_diff);
                inner.widget.convert(item, $bounds, $n_visible_points, *x, next_x, y)
            })
        }};
    }
    macro_rules! read {
        ($guard:ident, $x:ident, $y:ident) => {
            let $guard = sync::read!(inner.streams);
            let $x = &$guard.streams[inner.account_id][inner.stream_id].ts_ns;
            let $y = inner.widget.primitives(
                &$guard.streams[inner.account_id][inner.stream_id].items[inner.field_id],
            );
        };
        ($guard:ident, $x:ident) => {
            let $guard = sync::read!(inner.streams);
            let $x = &$guard.streams[inner.account_id][inner.stream_id].ts_ns;
        };
    }
    // puffin::profile_scope!("compute");
    let now = std::time::Instant::now();
    match &mut state.bounds {
        Some(self_bounds)
            // if !bounds_eq(bounds, *self_bounds) 
            => {
            *self_bounds = bounds.clone();
            let min_x = bounds.start as i64;
            let max_x = bounds.end as i64;
            read!(guard, x);
            let start = match x.binary_search_by_key(&(min_x), |x| *x) {
                Ok(x) => x,
                Err(x) => x,
            }
            .min(x.len().saturating_sub(1));
            let end = match x.binary_search_by_key(&(max_x), |x| *x) {
                Ok(x) => x,
                Err(x) => x,
            }
            .min(x.len());
            drop(guard);
            let view_len = end - start;
            let rect = ctx.available_rect();
            let pixels_per_point = ctx.pixels_per_point();
            let pixel_rect = rect * pixels_per_point;
            let mut update = false;
            let lod_level = if view_len > pixel_rect.width() as usize {
                view_len / pixel_rect.width() as usize
            } else {
                // recomputing candle width when data is appended
                update = true;
                1
            };
            let mut ratio = (state.lod_level as f32 / lod_level as f32);
            if ratio < 1. {
                ratio = 1. / ratio;
            }
            if ratio > 1.5 {
                update = true;
                state.lod_level = lod_level;
            }

            // Recompute all lod
            if update {
                state.lod_data.clear();
                read!(guard, x, y);
                if state.lod_level == 1 {
                    state.lod_data.par_extend(iter!(guard, x, y, start..end, &bounds, view_len));
                } else {
                    let lod_level = state.lod_level;
                    state.lod_data.par_extend(lod_iter!(
                        guard,
                        x,
                        y,
                        start..end,
                        &bounds,
                        view_len,
                        lod_level
                    ));
                }
            } else {
                // Partially recompute lod
                if start > state.view_range.start {
                    let len = start - state.view_range.start;
                    let lod_len = if state.lod_level == 1 {
                        len.saturating_sub(1)
                    } else {
                        len / state.lod_level * <T::Fold as ConstLen>::LEN
                    };
                    // optimization
                    for _ in 0..lod_len {
                        state.lod_data.pop_front();
                    }

                    let start_x =
                        sync::read!(inner.streams).streams[inner.account_id][inner.stream_id].ts_ns[start] as f64;
                    while let Some(point) = state.lod_data.front() {
                        if point.x() < start_x as f64 {
                            state.lod_data.pop_front();
                        } else {
                            break;
                        }
                    }
                }
                if end < state.view_range.end {
                    let len = state.view_range.end - end;
                    let lod_len = if state.lod_level == 1 {
                        len.saturating_sub(1)
                    } else {
                        len / state.lod_level * <T::Fold as ConstLen>::LEN
                    };
                    // optimization
                    for _ in 0..lod_len {
                        state.lod_data.pop_back();
                    }
                    let end_x =
                        sync::read!(inner.streams).streams[inner.account_id][inner.stream_id].ts_ns[end] as f64;
                    while let Some(point) = state.lod_data.back() {
                        if point.x() > end_x as f64 {
                            state.lod_data.pop_back();
                        } else {
                            break;
                        }
                    }
                }
                if start < state.view_range.start {
                    read!(guard, x, y);
                    let range = start..state.view_range.start;
                    if state.lod_level == 1 {
                        state.lod_tmp.par_extend(iter!(guard, x, y, range, &(bounds.start..bounds.end), view_len));
                        while let Some(point) = state.lod_tmp.pop() {
                            state.lod_data.push_front(point);
                        }
                    } else {
                    let lod_level = state.lod_level;
                        state.lod_tmp.par_extend(lod_iter!(
                            guard,
                            x,
                            y,
                            range,
                            &(bounds.start..bounds.end),
                            view_len,
                            lod_level
                        ));
                        while let Some(point) = state.lod_tmp.pop() {
                            state.lod_data.push_front(point);
                        }
                    }
                }
                if end > state.view_range.end {
                    read!(guard, x, y);
                    let range = state.view_range.end..end;
                    if state.lod_level == 1 {
                        state.lod_data.par_extend(iter!(guard, x, y, range, &bounds, view_len));
                    } else {
                        let lod_level = state.lod_level;
                        state.lod_data.par_extend(lod_iter!(
                            guard,
                            x,
                            y,
                            range,
                            &bounds,
                            view_len,
                            lod_level
                        ));
                    }
                }
            }
            state.view_range = start..end;
        },
        None => {
            state.lod_data.clear();
            let rect = ctx.screen_rect();
            let pixels_per_point = ctx.pixels_per_point();
            let pixel_rect = rect * pixels_per_point;
            read!(guard, x, y);
            state.view_range = 0..x.len();
            let view_len = x.len();
            let min_x = x.first().copied().unwrap_or(0) as f64;
            let max_x = x.last().copied().unwrap_or(1) as f64;
            // let (min_y, max_y) = y
            //     .par_iter()
            //     .fold(
            //         || (f32::MAX, f32::MIN),
            //         |acc, p| (acc.0.min(p.y()), acc.1.max(p.y())),
            //     )
            //     .reduce(
            //         || (f32::MAX, f32::MIN),
            //         |acc, p| (acc.0.min(p.0), acc.1.max(p.1)),
            //     );
            state.bounds = Some(min_x as f64..max_x as f64);
            // state.bounds = Some(PlotBounds::from_min_max(
            //     [min_x as f64, min_y as f64],
            //     [max_x as f64, max_y as f64],
            // ));
            let lod_level = if view_len > pixel_rect.width() as usize {
                view_len / pixel_rect.width() as usize
            } else {
                1
            };
            state.lod_level = lod_level;
            if state.lod_level == 1 {
                state.lod_data.par_extend(iter!(guard, x, y, .., &bounds, view_len));
            } else {
                    let lod_level = state.lod_level;
                state.lod_data.par_extend(lod_iter!(
                    guard,
                    x,
                    y,
                    ..,
                    &bounds,
                    view_len,
                    lod_level
                ));
            }
        },
        _ => return,
    }
    // println!("Compute pre_finish: {:?}", now.elapsed());
    let data = state.lod_data.make_contiguous();
    if appended {
        inner.widget.appended(data);
    }
    let mut lod_data = lock!(inner.lod_data);
    let mut min_y = f64::MAX;
    let mut max_y = f64::MIN;
    lod_data.clear();
    lod_data.extend(data.iter().map(|x| {
        if x.y() < min_y {
            min_y = x.y();
        }
        if x.y() > max_y {
            max_y = x.y();
        }
        x.clone()
    }));
    drop(lod_data);
    let elapsed_ms = now.elapsed().as_millis() as u32;
    let mut plot_state = lock!(plot_state);
    plot_state.max_compute_duration_ms = plot_state.max_compute_duration_ms.max(elapsed_ms);
    // if plot_state.auto_bounds {
    //     match plot_state.set_ui_bounds {
    //         Some(ui_bounds) => {
    //             plot_state.set_ui_bounds = Some(PlotBounds::from_min_max(
    //                 [bounds.start as f64, min_y.min(ui_bounds.min()[1])],
    //                 [bounds.end as f64, max_y.max(ui_bounds.max()[1])],
    //             ));
    //         }
    //         None => {
    //             plot_state.set_ui_bounds = Some(PlotBounds::from_min_max(
    //                 [bounds.start as f64, min_y],
    //                 [bounds.end as f64, max_y],
    //             ));
    //         }
    //     }
    // }
    drop(plot_state);
    // dbg!(state.time.elapsed());
    // println!("Compute: {:?}", now.elapsed());
    // println!("{:?}: done", Utc::now());
}

pub fn fold<W: SeriesWidget<Fold = [<W as SeriesWidget>::Point; 2]>>(
    widget: &W,
    x: &[i64],
    y: &[<W::Options as DebugStream>::Primitive],
    item: &PlotItem,
    bounds: &Range<f64>,
    n_points: usize,
) -> W::Fold {
    let mut min_i = 0;
    let mut max_i = 0;
    let mut min_y = f32::MAX;
    let mut max_y = f32::MIN;
    for (i, p) in y.iter().enumerate() {
        if p.y() < min_y {
            min_y = p.y();
            min_i = i;
        }
        if p.y() > max_y {
            max_y = p.y();
            max_i = i;
        }
    }
    let x_coord = x[x.len() / 2];
    let x_diff = x[x.len() - 1] - x[0];
    let next_x = x_coord + x_diff;
    [
        widget.convert(item, bounds, n_points, x_coord, next_x, &y[min_i]),
        widget.convert(item, bounds, n_points, x_coord, next_x, &y[max_i]),
    ]
}

pub fn show_coordinates(
    ui: &mut Ui,
    painter: &Painter,
    coordinates_position: &mut Pos2,
    transform: &PlotTransform,
    item: &PlotItem,
    value: f64,
    color: Color,
) {
    let font_id = TextStyle::Monospace.resolve(ui.style());
    let text = format!(" {}", PrettyPrintFloat(value));
    coordinates_position.x += painter
        .text(
            *coordinates_position,
            Align2::LEFT_TOP,
            text,
            font_id,
            convert_color(color),
        )
        .width();
}
