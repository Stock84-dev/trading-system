use std::ops::Range;

use egui::{Context, Painter, Stroke, Ui};
use egui_plot::{
    Bar, BarChart, Line, Orientation, PlotBounds, PlotPoint, PlotPoints, PlotTransform, PlotUi,
};
use emath::Pos2;
use esl_gpu::{DebugStream, DebugStreamData, PlotItem, Primitive, VerticalBarsOptions};
use parking_lot::RwLock;
use triomphe::Arc;

use crate::convert_color;
use crate::generic_series::{GenericSeries, Point, SeriesWidget, fold, show_coordinates};

pub struct VerticalBarsWidget;

impl SeriesWidget for VerticalBarsWidget {
    type Fold = [Bar; 2];
    type Options = VerticalBarsOptions;
    type Point = Bar;

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
            esl_gpu::PlotItemData::VerticalBars { primitives, .. } => primitives,
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
        y: &<Self::Options as DebugStream>::Primitive,
    ) -> Self::Point {
        let options = match &item.data {
            esl_gpu::PlotItemData::VerticalBars { options, .. } => options,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        Bar {
            name: String::new(),
            orientation: Orientation::Vertical,
            argument: x as f64,
            value: y.y() as f64,
            base_offset: None,
            bar_width: options
                .width
                .map(|x| x as f64)
                .unwrap_or(((bounds.end - bounds.start) / n_points as f64 / 2.).max(1.)),
            stroke: Stroke::new(options.outline_width, convert_color(options.outline_color)),
            fill: convert_color(options.fill_color),
        }
    }

    fn appended(&self, points: &mut [Self::Point]) {
        let x_diff = (points.last().map(|x| x.argument).unwrap_or(0.)
            - points.first().map(|x| x.argument).unwrap_or(0.)) as f64
            / points.len() as f64;
        for i in 0..points.len() {
            points[i].bar_width =
                points.get(i + 1).map(|p| p.argument - points[i].argument).unwrap_or(x_diff);
        }
    }

    fn show(&self, plot_ui: &mut PlotUi, item: &esl_gpu::PlotItem, data: Vec<Bar>) {
        let options = match &item.data {
            esl_gpu::PlotItemData::VerticalBars { options, .. } => options,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        plot_ui.bar_chart(
            BarChart::new(data)
                .color(convert_color(options.fill_color))
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
        let color = match &item.data {
            esl_gpu::PlotItemData::VerticalBars { options, .. } => options.outline_color,
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

impl Point for Bar {
    fn x(&self) -> f64 {
        self.argument
    }

    fn y(&self) -> f64 {
        self.value
    }
}
