use std::ops::Range;

use egui::{Color32, Context, Painter, RichText, Stroke, TextStyle, Ui, WidgetText};
use egui_plot::{
    BoxElem, BoxPlot, BoxSpread, Line, Orientation, PlotBounds, PlotPoint, PlotPoints,
    PlotTransform, PlotUi, Text,
};
use emath::{Align2, Pos2};
use esl_gpu::{CandlesOptions, CandlesPrimitive, DebugStream, DebugStreamData, PlotItem};
use float_pretty_print::PrettyPrintFloat;
use parking_lot::RwLock;
use triomphe::Arc;

use crate::convert_color;
use crate::generic_series::{GenericSeries, Point, SeriesWidget, fold};

pub struct CandlesWidget;

impl SeriesWidget for CandlesWidget {
    type Fold = [BoxElem; 1];
    type Options = CandlesOptions;
    type Point = BoxElem;

    fn fold(
        &self,
        x: &[i64],
        y: &[<Self::Options as DebugStream>::Primitive],
        item: &PlotItem,
        bounds: &Range<f64>,
        n_points: usize,
    ) -> Self::Fold {
        let (min, max) = y.iter().fold((f32::MAX, f32::MIN), |(min, max), p| {
            (min.min(p.low), max.max(p.high))
        });
        let x_coord = x[x.len() / 2];
        let x_diff = x[x.len() - 1] - x[0];
        let next_x = x_coord + x_diff;
        [self.convert(
            item,
            bounds,
            n_points,
            x_coord,
            next_x,
            &CandlesPrimitive {
                open: y[0].open,
                high: max,
                low: min,
                close: y[y.len() - 1].close,
            },
        )]
    }

    fn primitives<'a>(
        &self,
        item: &'a PlotItem,
    ) -> &'a [<Self::Options as DebugStream>::Primitive] {
        match &item.data {
            esl_gpu::PlotItemData::Candles { primitives, .. } => primitives,
            _ => unsafe { std::hint::unreachable_unchecked() },
        }
    }

    #[inline(always)]
    fn convert(
        &self,
        item: &PlotItem,
        bounds: &Range<f64>,
        n_points: usize,
        x: i64,
        next_x: i64,
        primitive: &<Self::Options as DebugStream>::Primitive,
    ) -> Self::Point {
        let options = match &item.data {
            esl_gpu::PlotItemData::Candles { options, .. } => options,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        let width = options
            .width
            .map(|x| x as f64)
            .unwrap_or((bounds.end - bounds.start) / n_points as f64 / 2.);
        BoxElem {
            name: String::new(),
            orientation: Orientation::Vertical,
            argument: x as f64,
            spread: BoxSpread {
                lower_whisker: primitive.low as f64,
                quartile1: primitive.open.min(primitive.close) as f64,
                median: ((primitive.open + primitive.close) / 2.0) as f64,
                quartile3: primitive.open.max(primitive.close) as f64,
                upper_whisker: primitive.high as f64,
            },
            box_width: width,
            // box_width: ((next_x - x) as f64).abs(),
            // box_width: ((bounds.end - bounds.start) / n_points as f64).max(1.),
            whisker_width: 1.0,
            stroke: if primitive.close > primitive.open {
                Stroke::new(1., convert_color(options.up_outline_color))
            } else {
                Stroke::new(1., convert_color(options.down_outline_color))
            },
            fill: if primitive.close > primitive.open {
                convert_color(options.up_fill_color)
            } else {
                convert_color(options.down_fill_color)
            },
        }
    }

    fn appended(&self, points: &mut [Self::Point]) {
        let x_diff = (points.last().map(|x| x.argument).unwrap_or(0.)
            - points.first().map(|x| x.argument).unwrap_or(0.)) as f64
            / points.len() as f64;
        for i in 0..points.len() {
            points[i].box_width =
                points.get(i + 1).map(|p| p.argument - points[i].argument).unwrap_or(x_diff);
        }
    }

    fn show(&self, plot_ui: &mut PlotUi, item: &esl_gpu::PlotItem, data: Vec<BoxElem>) {
        let options = match &item.data {
            esl_gpu::PlotItemData::Candles { options, .. } => options,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        plot_ui.box_plot(
            BoxPlot::new(data)
                .color(convert_color(options.color))
                .name(item.name.clone())
                .element_formatter(Box::new(|_, _| String::new())),
        );
    }

    fn show_coordinates(
        &self,
        ui: &mut Ui,
        painter: &Painter,
        coordinates_position: &mut Pos2,
        transform: &PlotTransform,
        item: &PlotItem,
        point: &Self::Point,
    ) {
        let options = match &item.data {
            esl_gpu::PlotItemData::Candles { options, .. } => options,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        let up_fill_color: Color32 = convert_color(options.up_fill_color);
        let open;
        let close;
        if point.fill == up_fill_color {
            open = point.spread.quartile1;
            close = point.spread.quartile3;
        } else {
            open = point.spread.quartile3;
            close = point.spread.quartile1;
        }
        let high = point.spread.upper_whisker;
        let low = point.spread.lower_whisker;
        let text = format!(
            " {} {} {} {}",
            PrettyPrintFloat(open),
            PrettyPrintFloat(high),
            PrettyPrintFloat(low),
            PrettyPrintFloat(close)
        );
        let font_id = TextStyle::Monospace.resolve(ui.style());
        coordinates_position.x += painter
            .text(
                *coordinates_position,
                Align2::LEFT_TOP,
                text,
                font_id,
                convert_color(options.color),
            )
            .width();
    }
}

impl Point for BoxElem {
    fn x(&self) -> f64 {
        self.argument
    }

    fn y(&self) -> f64 {
        self.spread.median
    }
}
