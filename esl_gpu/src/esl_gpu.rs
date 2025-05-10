#![cfg_attr(target_os = "cuda", no_std)]
#![feature(array_from_fn)]
#![feature(generic_const_exprs)]
#![feature(generic_associated_types)]
#![feature(slice_swap_unchecked)]

extern crate alloc;

#[macro_use]
extern crate bytemuck;

use core::ops::Range;

pub use debug::*;
use ergnomics_gpu::*;

pub use num::*;
mod debug;
mod num;

#[cfg(not(target_os = "cuda"))]
pub struct DebugStreamsInner {
    pub streams: Vec<Vec<DebugStreamData>>,
    pub bounds_ns: Bounds,
}

#[cfg(not(target_os = "cuda"))]
pub type DebugStreams = sync::Arc<sync::RwLock<DebugStreamsInner>>;

pub trait Debugger: for<'a> DebuggerImpl<'a> {}

pub trait DebuggerImpl<'a> {
    type SelectedDebugStream: SelectedDebugStream;
    /// Store plotting data. Data visualization automatically shows for first account id.
    /// `stream_id` identifies a dataframe. Each call to this function must enter a complete row.
    /// If used in a branch then use another `stream_id`. `stream_id` must be unique to a branch in
    /// code.
    /// # Example:
    /// ```
    /// ctx.debug(0, 0, ts_ns).add((
    ///     lines!(c, 0),
    ///     points!(c2, 0, 0, |x| x
    ///         .shape(PointShape::Up)
    ///         .color(Color::GREEN)
    ///         .radius(5.)),
    ///     bars!(bar, 0, 1),
    ///     texts!((c2, "Hello, world!")),
    ///     candles!((open, high, low, close), 0, 2),
    /// ));
    /// ```
    fn debug(
        &'a self,
        account_id: usize,
        stream_id: usize,
        ts_ns: i64,
    ) -> Self::SelectedDebugStream;
    /// Go to area of interest in plot window
    fn set_debug_bounds(&self, bounds_ns: Range<u64>);
    fn clear(&self);
}

pub trait Ctx<const N_ACCOUNTS: usize>: Debugger {
    type ContextState<P: Copy + 'static, M: Metrics>: CtxState<P, M, N_ACCOUNTS> + Send;
    #[cfg(not(target_os = "cuda"))]
    fn debug_streams(&self) -> &DebugStreams;
    fn to_state<P: Copy + 'static, M: Metrics>(
        &self,
        params: *const P,
        metrics: *const M::Snapshot,
        n_threads: u32,
        thread_id: u32,
        starting_balance: f32,
        round_trip_fee: f32,
        tick_size: f32,
    ) -> Self::ContextState<P, M>;
}

pub trait Metrics {
    type UpdateArgs;
    type PositionOpenedArgs;
    type PositionClosedArgs;
    type SnapshotArgs;
    type Snapshot: MetricSnapshot;
    fn new(starting_balance: f32) -> Self;
    fn balance(&self) -> f32;
    fn ignore(&self, ts_ns: u64, i: u32) -> bool;
    fn update(&mut self, d: &impl Debugger, ts_ns: u64, i: u32, args: Self::UpdateArgs);
    fn position_opened(
        &mut self,
        d: &impl Debugger,
        ts_ns: u64,
        i: u32,
        args: Self::PositionOpenedArgs,
    );
    fn position_closed(
        &mut self,
        d: &impl Debugger,
        ts_ns: u64,
        i: u32,
        args: Self::PositionClosedArgs,
    );
    fn snapshot(
        &self,
        d: &impl Debugger,
        ts_ns: u64,
        i: u32,
        args: Self::SnapshotArgs,
    ) -> Self::Snapshot;
}

pub trait MetricSnapshot {
    type Report;
    fn objective(&self) -> f32;
    fn set_p_value(&mut self, p_value: f32);
    fn to_report(&self) -> Self::Report;
}

pub type Context<const N_ACCOUNTS: usize> = ContextInner<N_ACCOUNTS>;

#[derive(Clone)]
pub struct ContextInner<const N_ACCOUNTS: usize> {
    #[cfg(not(target_os = "cuda"))]
    debug_streams: DebugStreams,
}

impl<const N_ACCOUNTS: usize> Ctx<N_ACCOUNTS> for ContextInner<N_ACCOUNTS> {
    type ContextState<P: Copy + 'static, M: Metrics> = ContextStateInner<P, M, N_ACCOUNTS>;

    #[inline(always)]
    fn to_state<P: Copy + 'static, M: Metrics>(
        &self,
        params: *const P,
        metrics: *const M::Snapshot,
        n_threads: u32,
        thread_id: u32,
        starting_balance: f32,
        round_trip_fee: f32,
        tick_size: f32,
    ) -> Self::ContextState<P, M> {
        ContextStateInner {
            params,
            metrics,
            n_threads,
            thread_id,
            starting_balance,
            round_trip_fee,
            tick_size,
            #[cfg(not(target_os = "cuda"))]
            debug_streams: self.debug_streams.clone(),
        }
    }

    #[cfg(not(target_os = "cuda"))]
    #[inline(always)]
    fn debug_streams(&self) -> &DebugStreams {
        &self.debug_streams
    }
}

impl<const N_ACCOUNTS: usize> Default for ContextInner<N_ACCOUNTS> {
    fn default() -> Self {
        Self {
            #[cfg(not(target_os = "cuda"))]
            debug_streams: sync::Arc::new(sync::RwLock::new(
                DebugStreamsInner {
                    streams: (0..N_ACCOUNTS).map(|_| Vec::new()).collect::<Vec<_>>(),
                    bounds_ns: Bounds::Auto,
                },
                || "debug_streams",
            )),
        }
    }
}

impl<'a, const N_ACCOUNTS: usize> DebuggerImpl<'a> for ContextInner<N_ACCOUNTS> {
    type SelectedDebugStream = NoDebugStream;

    #[inline(always)]
    fn debug(
        &'a self,
        account_id: usize,
        stream_id: usize,
        ts_ns: i64,
    ) -> Self::SelectedDebugStream {
        NoDebugStream
    }

    #[inline(always)]
    fn set_debug_bounds(&self, bounds_ns: Range<u64>) {}
    #[inline(always)]
    fn clear(&self) {
    }
}

impl<const N_ACCOUNTS: usize> Debugger for ContextInner<N_ACCOUNTS> {}

pub struct DebugContextState<P, M: Metrics, const N_ACCOUNTS: usize> {
    pub inner: ContextStateInner<P, M, N_ACCOUNTS>,
}

impl<'a, P, M: Metrics, const N_ACCOUNTS: usize> DebuggerImpl<'a>
    for DebugContextState<P, M, N_ACCOUNTS>
{
    #[cfg(not(target_os = "cuda"))]
    type SelectedDebugStream = ConcreteSelectedDebugStream<'a>;
    #[cfg(target_os = "cuda")]
    type SelectedDebugStream = ConcreteSelectedDebugStream;

    fn debug(
        &'a self,
        account_id: usize,
        stream_id: usize,
        ts_ns: i64,
    ) -> Self::SelectedDebugStream {
        #[cfg(not(target_os = "cuda"))]
        {
            ConcreteSelectedDebugStream {
                streams: &self.inner.debug_streams,
                account_id,
                stream_id,
                ts_ns,
            }
        }
        #[cfg(target_os = "cuda")]
        ConcreteSelectedDebugStream {}
    }

    fn set_debug_bounds(&self, bounds_ns: Range<u64>) {
        #[cfg(not(target_os = "cuda"))]
        {
            let mut streams = sync::write!(self.inner.debug_streams);
            streams.bounds_ns = Bounds::Range(bounds_ns);
        }
    }

    #[inline(always)]
    fn clear(&self) {
        #[cfg(not(target_os = "cuda"))]
        {
            let mut streams = sync::write!(self.inner.debug_streams);
            streams.streams.iter_mut().for_each(|x| x.clear());
        }
    }
}

impl<P: Copy + 'static, M: Metrics, const N_ACCOUNTS: usize> Debugger
    for DebugContextState<P, M, N_ACCOUNTS>
{
}

impl<P: Copy + 'static, M: Metrics, const N_ACCOUNTS: usize> CtxState<P, M, N_ACCOUNTS>
    for DebugContextState<P, M, N_ACCOUNTS>
{
    #[inline(always)]
    fn n_threads(&self) -> u32 {
        self.inner.n_threads()
    }

    #[inline(always)]
    fn thread_id(&self) -> u32 {
        self.inner.thread_id()
    }

    #[inline(always)]
    fn starting_balance(&self) -> f32 {
        self.inner.starting_balance()
    }

    #[inline(always)]
    fn round_trip_fee(&self) -> f32 {
        self.inner.round_trip_fee()
    }

    #[inline(always)]
    fn params(&self, account_id: usize) -> P {
        self.inner.params(account_id)
    }

    #[inline(always)]
    fn write_metrics(&mut self, account_id: usize, snapshot: <M as Metrics>::Snapshot) {
        self.inner.write_metrics(account_id, snapshot)
    }

    #[inline(always)]
    fn metrics_mut(&self, account_id: usize) -> &mut M::Snapshot {
        self.inner.metrics_mut(account_id)
    }

    #[inline(always)]
    fn tick_size(&self) -> f32 {
        self.inner.tick_size()
    }
}

impl<'a, P: 'a, M: Metrics, const N_ACCOUNTS: usize> CtxStateRef<'a, P>
    for DebugContextState<P, M, N_ACCOUNTS>
{
    type ParamsIterMut = core::iter::StepBy<core::iter::Skip<core::slice::IterMut<'a, P>>>;

    #[inline(always)]
    fn params_iter_mut(&'a mut self) -> Self::ParamsIterMut {
        self.inner.params_iter_mut()
    }
}

pub struct ContextStateInner<P, M: Metrics, const N_ACCOUNTS: usize> {
    params: *const P,
    metrics: *const M::Snapshot,
    n_threads: u32,
    thread_id: u32,
    starting_balance: f32,
    round_trip_fee: f32,
    tick_size: f32,
    #[cfg(not(target_os = "cuda"))]
    debug_streams: DebugStreams,
}

unsafe impl<P, M: Metrics, const N_ACCOUNTS: usize> Send for ContextStateInner<P, M, N_ACCOUNTS> {}
unsafe impl<P, M: Metrics, const N_ACCOUNTS: usize> Sync for ContextStateInner<P, M, N_ACCOUNTS> {}

impl<'a, P, M: Metrics, const N_ACCOUNTS: usize> DebuggerImpl<'a>
    for ContextStateInner<P, M, N_ACCOUNTS>
{
    type SelectedDebugStream = NoDebugStream;

    fn debug(
        &'a self,
        account_id: usize,
        stream_id: usize,
        ts_ns: i64,
    ) -> Self::SelectedDebugStream {
        NoDebugStream
    }

    fn set_debug_bounds(&self, bounds_ns: Range<u64>) {}
    #[inline(always)]
    fn clear(&self) {
    }
}

impl<P: Copy + 'static, M: Metrics, const N_ACCOUNTS: usize> Debugger
    for ContextStateInner<P, M, N_ACCOUNTS>
{
}

impl<P: Copy + 'static, M: Metrics, const N_ACCOUNTS: usize> CtxState<P, M, N_ACCOUNTS>
    for ContextStateInner<P, M, N_ACCOUNTS>
{
    #[inline(always)]
    fn n_threads(&self) -> u32 {
        self.n_threads
    }

    #[inline(always)]
    fn thread_id(&self) -> u32 {
        self.thread_id
    }

    #[inline(always)]
    fn starting_balance(&self) -> f32 {
        self.starting_balance
    }

    #[inline(always)]
    fn round_trip_fee(&self) -> f32 {
        self.round_trip_fee
    }

    #[inline(always)]
    fn params(&self, account_id: usize) -> P {
        unsafe {
            (*self
                .params
                .add((account_id as u32 * self.n_threads + self.thread_id) as usize))
        }
    }

    #[inline(always)]
    fn write_metrics(&mut self, account_id: usize, snapshot: <M as Metrics>::Snapshot) {
        unsafe {
            let offset = ((account_id as u32 * self.n_threads + self.thread_id) as usize);
            self.metrics.add(offset).write(snapshot);
        }
    }

    #[inline(always)]
    fn metrics_mut(&self, account_id: usize) -> &mut M::Snapshot {
        unsafe {
            let offset = ((account_id as u32 * self.n_threads + self.thread_id) as usize);
            &mut *(self.metrics.add(offset) as *mut M::Snapshot)
        }
    }

    fn tick_size(&self) -> f32 {
        self.tick_size
    }
}

impl<'a, P: 'a, M: Metrics, const N_ACCOUNTS: usize> CtxStateRef<'a, P>
    for ContextStateInner<P, M, N_ACCOUNTS>
{
    type ParamsIterMut = core::iter::StepBy<core::iter::Skip<core::slice::IterMut<'a, P>>>;

    #[inline(always)]
    fn params_iter_mut(&'a mut self) -> Self::ParamsIterMut {
        unsafe {
            let len = N_ACCOUNTS as u32 * self.n_threads;
            core::slice::from_raw_parts_mut(self.params as *mut P, len as usize)
                .iter_mut()
                .skip(self.thread_id as usize)
                .step_by(self.n_threads as usize)
        }
    }
}

#[derive(Clone, Default)]
pub struct DebugContext<const N_ACCOUNTS: usize> {
    pub inner: ContextInner<N_ACCOUNTS>,
}

impl<'a, const N_ACCOUNTS: usize> DebuggerImpl<'a> for DebugContext<N_ACCOUNTS> {
    #[cfg(not(target_os = "cuda"))]
    type SelectedDebugStream = ConcreteSelectedDebugStream<'a>;
    #[cfg(target_os = "cuda")]
    type SelectedDebugStream = ConcreteSelectedDebugStream;

    #[inline(always)]
    fn debug(
        &'a self,
        account_id: usize,
        stream_id: usize,
        ts_ns: i64,
    ) -> Self::SelectedDebugStream {
        #[cfg(not(target_os = "cuda"))]
        {
            ConcreteSelectedDebugStream {
                streams: &self.inner.debug_streams(),
                account_id,
                stream_id,
                ts_ns,
            }
        }
        #[cfg(target_os = "cuda")]
        ConcreteSelectedDebugStream {}
    }

    #[inline(always)]
    fn set_debug_bounds(&self, bounds_ns: Range<u64>) {
        #[cfg(not(target_os = "cuda"))]
        {
            let mut streams = sync::write!(self.inner.debug_streams());
            streams.bounds_ns = Bounds::Range(bounds_ns);
        }
    }

    #[inline(always)]
    fn clear(&self) {
        #[cfg(not(target_os = "cuda"))]
        {
            let mut streams = sync::write!(self.inner.debug_streams());
            streams.streams.iter_mut().for_each(|x| x.clear());
        }
    }
}

impl<const N_ACCOUNTS: usize> Debugger for DebugContext<N_ACCOUNTS> {}

impl<const N_ACCOUNTS: usize> Ctx<N_ACCOUNTS> for DebugContext<N_ACCOUNTS> {
    type ContextState<P: Copy + 'static, M: Metrics> = DebugContextState<P, M, N_ACCOUNTS>;

    #[inline(always)]
    fn to_state<P: Copy + 'static, M: Metrics>(
        &self,
        params: *const P,
        metrics: *const M::Snapshot,
        n_threads: u32,
        thread_id: u32,
        starting_balance: f32,
        round_trip_fee: f32,
        tick_size: f32,
    ) -> Self::ContextState<P, M> {
        DebugContextState {
            inner: self.inner.to_state(
                params,
                metrics,
                n_threads,
                thread_id,
                starting_balance,
                round_trip_fee,
                tick_size,
            ),
        }
    }

    #[cfg(not(target_os = "cuda"))]
    fn debug_streams(&self) -> &DebugStreams {
        &self.inner.debug_streams
    }
}

pub trait CtxStateRef<'a, P: 'a> {
    type ParamsIterMut: Iterator<Item = &'a mut P>;
    fn params_iter_mut(&'a mut self) -> Self::ParamsIterMut;
}

pub trait CtxState<P: Copy + 'static, M: Metrics, const N_ACCOUNTS: usize>:
    Debugger + for<'a> CtxStateRef<'a, P>
{
    fn n_threads(&self) -> u32;
    fn thread_id(&self) -> u32;
    fn starting_balance(&self) -> f32;
    fn round_trip_fee(&self) -> f32;
    fn tick_size(&self) -> f32;
    fn params(&self, account_id: usize) -> P;
    fn write_metrics(&mut self, account_id: usize, snapshot: M::Snapshot);
    fn metrics_mut(&self, account_id: usize) -> &mut M::Snapshot;
}

#[derive(Clone)]
pub enum Bounds {
    Auto,
    Range(Range<u64>),
    // WasRange(Range<u64>),
}
