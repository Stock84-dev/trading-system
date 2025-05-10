use alloc::string::String;
use alloc::sync::Arc;
use core::hash::Hasher;

use ahash::AHasher;

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
        ts_ns: u64,
    ) -> Self::SelectedDebugStream;
    /// Go to area of interest in plot window
    fn set_debug_bounds(&self, bounds_ns: Range<u64>);
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

#[macro_export]
macro_rules! val_as_f32_into_primitive {
    ($value:expr) => {
        ($value as f32).into()
    };
}

#[macro_export]
macro_rules! val_into_primitive {
    ($value:expr) => {
        $value.into()
    };
}

macro_rules! def_macros {
    ($name:ident, $options:ty, $into_primitive:ident) => {
        #[macro_export]
        /// Usage examples:
        /// ```
        /// points!(value, axis_id, plot_id, |x| x.color(Color::GREEN).name("points")),
        /// texts!((c2, "Hello, world!")),
        /// candles!((open, high, low, close), 0, 2),
        /// ```
        macro_rules! $name {
            ($value: expr) => {
                DebugField::<_, $options> {
                    name: stringify!($value),
                    plot: 0,
                    axis: 0,
                    options: |x| x,
                    primitive: $into_primitive!($value),
                }
            };
            ($value: expr,$axis: expr) => {
                DebugField::<_, $options> {
                    name: stringify!($value),
                    plot: 0,
                    axis: $axis,
                    options: |x| x,
                    primitive: $into_primitive!($value),
                }
            };
            ($value: expr,$axis: expr,$plot: expr) => {
                DebugField::<_, $options> {
                    name: stringify!($value),
                    axis: $axis,
                    plot: $plot,
                    options: |x| x,
                    primitive: $into_primitive!($value),
                }
            };
            ($value: expr,$axis: expr,$plot: expr,$builder: expr) => {
                DebugField::<_, $options> {
                    name: stringify!($value),
                    axis: $axis,
                    plot: $plot,
                    options: $builder,
                    primitive: $into_primitive!($value),
                }
            };
        }
    };
}

def_macros!(lines, LinesOptions, val_as_f32_into_primitive);
def_macros!(points, PointsOptions, val_as_f32_into_primitive);
def_macros!(bars, VerticalBarsOptions, val_as_f32_into_primitive);
def_macros!(candles, CandlesOptions, val_into_primitive);
def_macros!(texts, TextsOptions, val_into_primitive);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Color(pub u8, pub u8, pub u8, pub u8);

impl Color {
    pub const BLACK: Self = Self(0, 0, 0, 0xff);
    pub const BLUE: Self = Self(0, 0, 0xff, 0xff);
    pub const CYAN: Self = Self(0, 0xff, 0xff, 0xff);
    pub const GREEN: Self = Self(0, 0xff, 0, 0xff);
    pub const MAGENTA: Self = Self(0xff, 0, 0xff, 0xff);
    pub const RED: Self = Self(0xff, 0, 0, 0xff);
    pub const TRANSPARENT: Self = Self(0, 0, 0, 0);
    pub const WHITE: Self = Self(0xff, 0xff, 0xff, 0xff);
    pub const YELLOW: Self = Self(0xff, 0xff, 0, 0xff);

    #[inline(always)]
    pub fn from_name(name: &str) -> Self {
        let mut hasher = AHasher::default();
        hasher.write(name.as_bytes());
        let value = hasher.finish();
        let r = (value & 0xff) as u8;
        let g = ((value >> 8) & 0xff) as u8;
        let b = ((value >> 16) & 0xff) as u8;
        Self(r | 0x40, g | 0x40, b | 0x40, 0xff)
    }

    #[inline(always)]
    pub fn from_name_dark(name: &str) -> Self {
        let mut hasher = AHasher::default();
        hasher.write(name.as_bytes());
        let value = hasher.finish();
        let r = (value & 0xff) as u8;
        let g = ((value >> 8) & 0xff) as u8;
        let b = ((value >> 16) & 0xff) as u8;
        Self(r | 0x20, g | 0x20, b | 0x20, 0xff)
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum LineStyle {
    Solid,
    Dotted { spacing: f32 },
    Dashed { length: f32 },
}

#[derive(PartialEq, Clone)]
pub struct LinesOptions {
    pub width: f32,
    pub color: Color,
    pub style: LineStyle,
}

impl LinesOptions {
    #[inline(always)]
    pub fn new(name: &str) -> Self {
        Self {
            width: 1.,
            color: Color::from_name(name),
            style: LineStyle::Solid,
        }
    }
}

#[derive(Clone, Copy, PartialEq)]
pub enum PointShape {
    Circle,
    Diamond,
    Square,
    Cross,
    Plus,
    Up,
    Down,
    Left,
    Right,
    Asterisk,
}

#[derive(PartialEq, Clone)]
pub struct PointsOptions {
    pub shape: PointShape,
    pub color: Color,
    pub filled: bool,
    pub stems: Option<f32>,
    pub radius: f32,
}

impl PointsOptions {
    #[inline(always)]
    pub fn new(name: &str) -> Self {
        Self {
            shape: PointShape::Circle,
            color: Color::from_name(name),
            filled: true,
            stems: None,
            radius: 2.0,
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct VerticalBarsOptions {
    pub width: Option<f32>,
    pub fill_color: Color,
    pub outline_color: Color,
    pub outline_width: f32,
}

impl VerticalBarsOptions {
    #[inline(always)]
    pub fn new(name: &str) -> Self {
        Self {
            width: None,
            fill_color: Color::from_name(name),
            outline_color: Color::from_name_dark(name),
            outline_width: 1.,
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct CandlesOptions {
    pub up_fill_color: Color,
    pub up_outline_color: Color,
    pub down_fill_color: Color,
    pub down_outline_color: Color,
    pub color: Color,
}

impl CandlesOptions {
    #[inline(always)]
    pub fn new(name: &str) -> Self {
        Self {
            up_fill_color: Color(8, 153, 129, 0xff),
            up_outline_color: Color(8, 153, 129, 0xff),
            down_fill_color: Color(242, 54, 69, 0xff),
            down_outline_color: Color(242, 54, 69, 0xff),
            color: Color::from_name(name),
        }
    }
}

#[derive(PartialEq, Clone)]
pub struct TextsOptions {
    pub size: Option<f32>,
    pub extra_letter_spacing: f32,
    pub line_height: Option<f32>,
    pub family: Option<FontFamily>,
    pub text_style: Option<TextStyle>,
    pub color: Color,
    pub code: bool,
    pub strong: bool,
    pub weak: bool,
    pub underline: bool,
    pub strikethrough: bool,
    pub italics: bool,
    pub raised: bool,
    pub background_color: Color,
    pub align: (Align, Align),
}

impl TextsOptions {
    #[inline(always)]
    pub fn new(name: &str) -> Self {
        Self {
            size: None,
            extra_letter_spacing: 0.0,
            line_height: None,
            family: None,
            text_style: None,
            color: Color::from_name(name),
            code: false,
            strong: false,
            weak: false,
            underline: false,
            strikethrough: false,
            italics: false,
            raised: false,
            background_color: Color::TRANSPARENT,
            align: (Align::Center, Align::Center),
        }
    }
}

pub struct NoDebugStream;

pub trait SelectedDebugStream {
    fn add(&mut self, event: impl DebugPlotEvent) {}
}

impl SelectedDebugStream for NoDebugStream {
    #[inline(always)]
    fn add(&mut self, event: impl DebugPlotEvent) {}
}

pub struct NoDebugStreamWriter {}

impl NoDebugStreamWriter {
    #[inline(always)]
    pub fn add<T>(&mut self, event: T) {}
}

#[cfg(target_os = "cuda")]
pub struct ConcreteSelectedDebugStream {}

#[cfg(target_os = "cuda")]
impl SelectedDebugStream for ConcreteSelectedDebugStream {
    #[inline(always)]
    fn add(&mut self, event: impl DebugPlotEvent) {}
}

#[cfg(target_os = "cuda")]
pub struct ConcreteDebugStreamWriter {}

#[cfg(target_os = "cuda")]
impl ConcreteDebugStreamWriter {
    #[inline(always)]
    pub fn add<T>(&mut self, event: T) {}
}

pub trait Primitive: Send + Sync {
    fn y(&self) -> f32;
}

pub trait DebugStream {
    type Primitive: Primitive;
}

trait AddField<T: DebugStream> {
    unsafe fn add_field<F: FnMut(DebugOptions<T>) -> DebugOptions<T>>(
        &mut self,
        name: impl Into<String>,
        plot: usize,
        axis: usize,
        field: usize,
        options: F,
        value: T::Primitive,
    );
}

trait DebugPlotEvent {
    fn add(self, writer: &mut ConcreteDebugStreamWriter);
}

pub struct DebugOptions<O> {
    pub name: String,
    pub options: O,
}

pub struct DebugField<F: FnMut(DebugOptions<O>) -> DebugOptions<O>, O: DebugStream> {
    pub name: &'static str,
    pub plot: usize,
    pub axis: usize,
    pub options: F,
    pub primitive: O::Primitive,
}

impl<T> DebugOptions<T> {
    #[inline(always)]
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }
}

impl DebugOptions<LinesOptions> {
    #[inline(always)]
    pub fn width(mut self, width: f32) -> Self {
        self.options.width = width;
        self
    }

    #[inline(always)]
    pub fn color(mut self, color: Color) -> Self {
        self.options.color = color;
        self
    }

    #[inline(always)]
    pub fn style(mut self, style: LineStyle) -> Self {
        self.options.style = style;
        self
    }
}

impl DebugOptions<PointsOptions> {
    #[inline(always)]
    pub fn shape(mut self, shape: PointShape) -> Self {
        self.options.shape = shape;
        self
    }

    #[inline(always)]
    pub fn color(mut self, color: Color) -> Self {
        self.options.color = color;
        self
    }

    #[inline(always)]
    pub fn filled(mut self, filled: bool) -> Self {
        self.options.filled = filled;
        self
    }

    #[inline(always)]
    pub fn stems(mut self, stems: f32) -> Self {
        self.options.stems = Some(stems);
        self
    }

    #[inline(always)]
    pub fn radius(mut self, radius: f32) -> Self {
        self.options.radius = radius;
        self
    }
}

impl DebugOptions<VerticalBarsOptions> {
    #[inline(always)]
    pub fn width(mut self, width: f32) -> Self {
        self.options.width = Some(width);
        self
    }

    #[inline(always)]
    pub fn fill_color(mut self, fill_color: Color) -> Self {
        self.options.fill_color = fill_color;
        self
    }

    #[inline(always)]
    pub fn outline_color(mut self, outline_color: Color) -> Self {
        self.options.outline_color = outline_color;
        self
    }

    #[inline(always)]
    pub fn outline_width(mut self, outline_width: f32) -> Self {
        self.options.outline_width = outline_width;
        self
    }
}

impl DebugOptions<CandlesOptions> {
    #[inline(always)]
    pub fn up_fill_color(mut self, up_fill_color: Color) -> Self {
        self.options.up_fill_color = up_fill_color;
        self
    }

    #[inline(always)]
    pub fn up_outline_color(mut self, up_outline_color: Color) -> Self {
        self.options.up_outline_color = up_outline_color;
        self
    }

    #[inline(always)]
    pub fn down_fill_color(mut self, down_fill_color: Color) -> Self {
        self.options.down_fill_color = down_fill_color;
        self
    }

    #[inline(always)]
    pub fn down_outline_color(mut self, down_outline_color: Color) -> Self {
        self.options.down_outline_color = down_outline_color;
        self
    }

    #[inline(always)]
    pub fn color(mut self, color: Color) -> Self {
        self.options.color = color;
        self
    }
}

impl DebugOptions<TextsOptions> {
    #[inline(always)]
    pub fn color(mut self, color: Color) -> Self {
        self.options.color = color;
        self
    }

    #[inline(always)]
    pub fn align(mut self, align: (Align, Align)) -> Self {
        self.options.align = align;
        self
    }

    #[inline(always)]
    pub fn size(mut self, size: f32) -> Self {
        self.options.size = Some(size);
        self
    }

    #[inline(always)]
    pub fn extra_letter_spacing(mut self, extra_letter_spacing: f32) -> Self {
        self.options.extra_letter_spacing = extra_letter_spacing;
        self
    }

    #[inline(always)]
    pub fn line_height(mut self, line_height: f32) -> Self {
        self.options.line_height = Some(line_height);
        self
    }

    #[inline(always)]
    pub fn family(mut self, family: FontFamily) -> Self {
        self.options.family = Some(family);
        self
    }

    #[inline(always)]
    pub fn text_style(mut self, text_style: TextStyle) -> Self {
        self.options.text_style = Some(text_style);
        self
    }

    #[inline(always)]
    pub fn code(mut self, code: bool) -> Self {
        self.options.code = code;
        self
    }

    #[inline(always)]
    pub fn strong(mut self, strong: bool) -> Self {
        self.options.strong = strong;
        self
    }

    #[inline(always)]
    pub fn weak(mut self, weak: bool) -> Self {
        self.options.weak = weak;
        self
    }

    #[inline(always)]
    pub fn underline(mut self, underline: bool) -> Self {
        self.options.underline = underline;
        self
    }

    #[inline(always)]
    pub fn strikethrough(mut self, strikethrough: bool) -> Self {
        self.options.strikethrough = strikethrough;
        self
    }

    #[inline(always)]
    pub fn italics(mut self, italics: bool) -> Self {
        self.options.italics = italics;
        self
    }

    #[inline(always)]
    pub fn raised(mut self, raised: bool) -> Self {
        self.options.raised = raised;
        self
    }

    #[inline(always)]
    pub fn background_color(mut self, background_color: Color) -> Self {
        self.options.background_color = background_color;
        self
    }

    #[inline(always)]
    pub fn align_min_min(self) -> Self {
        self.align((Align::Min, Align::Min))
    }

    #[inline(always)]
    pub fn align_min_center(self) -> Self {
        self.align((Align::Min, Align::Center))
    }

    #[inline(always)]
    pub fn align_center_center(self) -> Self {
        self.align((Align::Center, Align::Center))
    }

    #[inline(always)]
    pub fn align_max_max(self) -> Self {
        self.align((Align::Max, Align::Max))
    }
}

#[derive(Clone, PartialEq)]
pub enum FontFamily {
    Proportional,
    Monospace,
    Name(Arc<str>),
}

#[derive(Clone, PartialEq)]
pub enum TextStyle {
    Small,
    Body,
    Monospace,
    Button,
    Heading,
    Name(Arc<str>),
}

#[derive(Clone, Copy, PartialEq)]
pub enum Align {
    Min,
    Center,
    Max,
}

impl DebugStream for PointsOptions {
    type Primitive = PointsPrimitive;
}

impl Primitive for PointsPrimitive {
    #[inline(always)]
    fn y(&self) -> f32 {
        self.value
    }
}

#[derive(Clone, Copy)]
pub struct PointsPrimitive {
    pub value: f32,
}

impl From<f32> for PointsPrimitive {
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self { value }
    }
}

#[derive(Clone, Copy)]
pub struct LinesPrimitive {
    pub value: f32,
}

impl From<f32> for LinesPrimitive {
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self { value }
    }
}

impl DebugStream for LinesOptions {
    type Primitive = LinesPrimitive;
}

impl Primitive for LinesPrimitive {
    #[inline(always)]
    fn y(&self) -> f32 {
        self.value
    }
}

impl DebugStream for VerticalBarsOptions {
    type Primitive = VerticalBarsPrimitive;
}

impl Primitive for VerticalBarsPrimitive {
    #[inline(always)]
    fn y(&self) -> f32 {
        self.value
    }
}

#[derive(Clone, Copy)]
pub struct VerticalBarsPrimitive {
    pub value: f32,
}

impl From<f32> for VerticalBarsPrimitive {
    #[inline(always)]
    fn from(value: f32) -> Self {
        Self { value }
    }
}

impl DebugStream for CandlesOptions {
    type Primitive = CandlesPrimitive;
}

impl Primitive for CandlesPrimitive {
    #[inline(always)]
    fn y(&self) -> f32 {
        self.close
    }
}

pub struct CandlesPrimitive {
    pub open: f32,
    pub high: f32,
    pub low: f32,
    pub close: f32,
}

impl From<(f32, f32, f32, f32)> for CandlesPrimitive {
    #[inline(always)]
    fn from((open, high, low, close): (f32, f32, f32, f32)) -> Self {
        Self {
            open,
            high,
            low,
            close,
        }
    }
}

impl DebugStream for TextsOptions {
    type Primitive = TextsPrimitive;
}

impl Primitive for TextsPrimitive {
    #[inline(always)]
    fn y(&self) -> f32 {
        self.value
    }
}

pub struct TextsPrimitive {
    pub value: f32,
    pub text: String,
}

impl<S: Into<String>> From<(f32, S)> for TextsPrimitive {
    #[inline(always)]
    fn from((value, text): (f32, S)) -> Self {
        Self {
            value,
            text: text.into(),
        }
    }
}

#[cfg(target_os = "cuda")]
impl<T> DebugPlotEvent for T {
    #[inline(always)]
    fn add(self, writer: &mut ConcreteDebugStreamWriter) {}
}

#[cfg(not(target_os = "cuda"))]
mod cpu {
    use super::*;
    use crate::{DebugStreams, DebugStreamsInner};

    pub struct ConcreteSelectedDebugStream<'a> {
        pub(crate) streams: &'a DebugStreams,
        pub(crate) account_id: usize,
        pub(crate) stream_id: usize,
        pub(crate) ts_ns: u64,
    }

    impl<'a> SelectedDebugStream for ConcreteSelectedDebugStream<'a> {
        #[inline(always)]
        fn add(&mut self, event: impl DebugPlotEvent) {
            // Taking a lock once event is constructed to shorten critical section
            let mut streams = sync::write!(self.streams);
            let account_stream = unsafe { streams.streams.get_unchecked_mut(self.account_id) };
            let delta = self.stream_id as isize - account_stream.len() as isize + 1;
            for _ in 0..delta {
                account_stream.push(DebugStreamData {
                    ts_ns: Vec::new(),
                    items: Vec::new(),
                });
            }
            let stream = unsafe { account_stream.get_unchecked_mut(self.stream_id) };
            stream.ts_ns.push(self.ts_ns);
            event.add(&mut ConcreteDebugStreamWriter {
                guard: streams,
                account_id: self.account_id,
                stream_id: self.stream_id,
            });
        }
    }

    pub struct ConcreteDebugStreamWriter<'a> {
        pub(crate) guard: sync::RwLockWriteGuard<'a, DebugStreamsInner>,
        pub(crate) account_id: usize,
        pub(crate) stream_id: usize,
    }

    impl<'a> ConcreteDebugStreamWriter<'a> {
        #[inline(always)]
        pub fn add(&mut self, event: impl DebugPlotEvent) {
            event.add(self);
        }
    }

    macro_rules! impl_add_field {
        ($stream:ty, $variant:ident) => {
            impl<'a> AddField<$stream> for ConcreteDebugStreamWriter<'a> {
                #[inline(always)]
                unsafe fn add_field<F: FnMut(DebugOptions<$stream>) -> DebugOptions<$stream>>(
                    &mut self,
                    name: impl Into<String>,
                    plot: usize,
                    axis: usize,
                    field: usize,
                    mut options: F,
                    value: <$stream as DebugStream>::Primitive,
                ) {
                    let items = unsafe {
                        &mut self
                            .guard
                            .streams
                            .get_unchecked_mut(self.account_id)
                            .get_unchecked_mut(self.stream_id)
                            .items
                    };
                    if field == items.len() {
                        let name = name.into();
                        let options = options(DebugOptions {
                            options: <$stream>::new(&name),
                            name,
                        });
                        items.push(PlotItem {
                            data: PlotItemData::$variant {
                                primitives: Vec::new(),
                                options: options.options,
                            },
                            plot,
                            axis,
                            name: options.name,
                        });
                    }
                    let field_data = unsafe { &mut items.get_unchecked_mut(field).data };
                    if let PlotItemData::$variant { primitives, .. } = field_data {
                        primitives.push(value);
                    } else {
                        unsafe {
                            core::hint::unreachable_unchecked();
                        }
                    };
                }
            }
        };
    }

    impl_add_field!(LinesOptions, Lines);
    impl_add_field!(PointsOptions, Points);
    impl_add_field!(VerticalBarsOptions, VerticalBars);
    impl_add_field!(CandlesOptions, Candles);
    impl_add_field!(TextsOptions, Texts);

    impl<F: FnMut(DebugOptions<O>) -> DebugOptions<O>, O: DebugStream> DebugPlotEvent
        for DebugField<F, O>
    where
        for<'a> ConcreteDebugStreamWriter<'a>: AddField<O>,
    {
        #[inline(always)]
        fn add(self, writer: &mut ConcreteDebugStreamWriter) {
            unsafe {
                writer.add_field(
                    self.name,
                    self.plot,
                    self.axis,
                    0,
                    self.options,
                    self.primitive,
                );
            }
        }
    }

    macro_rules! impl_debug_plot_event {
    ($(($n:ident, $o:ident)),*) => {
        impl<$($n: FnMut(DebugOptions<$o>) -> DebugOptions<$o>, $o: DebugStream),*> DebugPlotEvent
            for ($(DebugField<$n, $o>,)*)
        where
        $(for<'a> ConcreteDebugStreamWriter<'a>: AddField<$o>),*
        {
            #[inline(always)]
            fn add(self, writer: &mut ConcreteDebugStreamWriter) {
                let ($($n,)*) = self;
                let mut offset = 0;
                unsafe {
                    $(
                        writer.add_field(
                            $n.name,
                            $n.plot,
                            $n.axis,
                            offset,
                            $n.options,
                            $n.primitive,
                        );
                        offset += 1;
                    )*
                }
            }
        }
    };
}

    impl_debug_plot_event!((N0, O0));
    impl_debug_plot_event!((N0, O0), (N1, O1));
    impl_debug_plot_event!((N0, O0), (N1, O1), (N2, O2));
    impl_debug_plot_event!((N0, O0), (N1, O1), (N2, O2), (N3, O3));
    impl_debug_plot_event!((N0, O0), (N1, O1), (N2, O2), (N3, O3), (N4, O4));
    impl_debug_plot_event!((N0, O0), (N1, O1), (N2, O2), (N3, O3), (N4, O4), (N5, O5));
    impl_debug_plot_event!(
        (N0, O0),
        (N1, O1),
        (N2, O2),
        (N3, O3),
        (N4, O4),
        (N5, O5),
        (N6, O6)
    );
    impl_debug_plot_event!(
        (N0, O0),
        (N1, O1),
        (N2, O2),
        (N3, O3),
        (N4, O4),
        (N5, O5),
        (N6, O6),
        (N7, O7)
    );

    pub struct DebugStreamData {
        pub ts_ns: Vec<u64>,
        pub items: Vec<PlotItem>,
    }

    pub struct PlotItem {
        pub data: PlotItemData,
        pub plot: usize,
        pub axis: usize,
        pub name: String,
    }

    pub enum PlotItemData {
        Lines {
            primitives: Vec<LinesPrimitive>,
            options: LinesOptions,
        },
        Points {
            primitives: Vec<PointsPrimitive>,
            options: PointsOptions,
        },
        VerticalBars {
            primitives: Vec<VerticalBarsPrimitive>,
            options: VerticalBarsOptions,
        },
        Candles {
            primitives: Vec<CandlesPrimitive>,
            options: CandlesOptions,
        },
        Texts {
            primitives: Vec<TextsPrimitive>,
            options: TextsOptions,
        },
    }

    pub enum PlotItemOptionsRef<'a> {
        Lines(&'a LinesOptions),
        Points(&'a PointsOptions),
        VerticalBars(&'a VerticalBarsOptions),
        Candles(&'a CandlesOptions),
        Texts(&'a TextsOptions),
    }
}
