use std::ops::Range;

use egui::{Context, Painter, Ui};
use egui_plot::{Line, PlotBounds, PlotPoint, PlotPoints, PlotTransform, PlotUi};
use emath::Pos2;
use esl_gpu::{DebugStream, DebugStreamData, LineStyle, LinesOptions, PlotItem, Primitive};
use parking_lot::RwLock;
use triomphe::Arc;

use crate::convert_color;
use crate::generic_series::{GenericSeries, Point, SeriesWidget, fold, show_coordinates};

pub struct LinesWidget;

impl SeriesWidget for LinesWidget {
    type Fold = [PlotPoint; 2];
    type Options = LinesOptions;
    type Point = PlotPoint;

    fn fold(
        &self,
        x: &[i64],
        y: &[<Self::Options as DebugStream>::Primitive],
        item: &PlotItem,
        bounds: &Range<f64>,
        n_points: usize,
    ) -> Self::Fold {
        fold(self, x, y, item, bounds, n_points)
    }

    fn primitives<'a>(
        &self,
        item: &'a PlotItem,
    ) -> &'a [<Self::Options as DebugStream>::Primitive] {
        match &item.data {
            esl_gpu::PlotItemData::Lines { primitives, .. } => primitives,
            _ => unreachable!(),
        }
    }

    fn convert(
        &self,
        item: &PlotItem,
        bounds: &Range<f64>,
        n_points: usize,
        x: i64,
        next_x: i64,
        y: &<Self::Options as DebugStream>::Primitive,
    ) -> Self::Point {
        PlotPoint::new(x as f64, y.y() as f64)
    }

    fn show(&self, plot_ui: &mut PlotUi, item: &esl_gpu::PlotItem, data: Vec<PlotPoint>) {
        let options = match &item.data {
            esl_gpu::PlotItemData::Lines { options, .. } => options,
            _ => unreachable!(),
        };
        plot_ui.line(
            Line::new(PlotPoints::Owned(data))
                .width(options.width)
                .color(convert_color(options.color))
                .style(match options.style {
                    LineStyle::Solid => egui_plot::LineStyle::Solid,
                    LineStyle::Dotted { spacing } => egui_plot::LineStyle::Dotted { spacing },
                    LineStyle::Dashed { length } => egui_plot::LineStyle::Dashed { length },
                })
                .name(item.name.clone()),
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
        let color = match &item.data {
            esl_gpu::PlotItemData::Lines { options, .. } => options.color,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        show_coordinates(
            ui,
            painter,
            coordinates_position,
            transform,
            item,
            point.y(),
            color,
        );
    }
}
