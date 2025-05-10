use std::ops::Range;

use egui::{Color32, Context, Painter, Ui};
use egui_plot::{PlotBounds, PlotPoint, PlotPoints, PlotTransform, PlotUi};
use emath::Pos2;
use esl_gpu::{DebugStream, DebugStreamData, PlotItem, PointShape, PointsOptions, Primitive};
use parking_lot::RwLock;
use triomphe::Arc;

use crate::generic_series::{fold, show_coordinates, GenericSeries, Point, SeriesWidget};
use crate::convert_color;

pub struct PointsWidget;

impl SeriesWidget for PointsWidget {
    type Fold = [PlotPoint; 2];
    type Options = PointsOptions;
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
            esl_gpu::PlotItemData::Points { primitives, .. } => primitives,
            _ => unreachable!(),
        }
    }

    fn show(&self, plot_ui: &mut PlotUi, item: &esl_gpu::PlotItem, data: Vec<PlotPoint>) {
        let options = match &item.data {
            esl_gpu::PlotItemData::Points { options, .. } => options,
            _ => unreachable!(),
        };
        let mut plot = egui_plot::Points::new(PlotPoints::Owned(data))
            .shape(match options.shape {
                PointShape::Circle => egui_plot::MarkerShape::Circle,
                PointShape::Diamond => egui_plot::MarkerShape::Diamond,
                PointShape::Square => egui_plot::MarkerShape::Square,
                PointShape::Cross => egui_plot::MarkerShape::Cross,
                PointShape::Plus => egui_plot::MarkerShape::Plus,
                PointShape::Up => egui_plot::MarkerShape::Up,
                PointShape::Down => egui_plot::MarkerShape::Down,
                PointShape::Left => egui_plot::MarkerShape::Left,
                PointShape::Right => egui_plot::MarkerShape::Right,
                PointShape::Asterisk => egui_plot::MarkerShape::Asterisk,
            })
            .color(convert_color(options.color))
            .filled(options.filled)
            .radius(options.radius)
            .name(item.name.clone());
        if let Some(stems) = options.stems {
            plot = plot.stems(stems);
        }
        plot_ui.points(plot);
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
            esl_gpu::PlotItemData::Points { options, .. } => options.color,
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
}

impl Point for PlotPoint {
    fn x(&self) -> f64 {
        self.x
    }

    fn y(&self) -> f64 {
        self.y
    }
}
