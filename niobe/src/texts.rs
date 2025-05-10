use std::ops::Range;

use egui::{Context, Painter, RichText, Stroke, Ui, WidgetText};
use egui_plot::{
    Line, Orientation, PlotBounds, PlotPoint, PlotPoints, PlotTransform, PlotUi, Text,
};
use emath::{Align2, Pos2};
use esl_gpu::{
    DebugStream, DebugStreamData, FontFamily, PlotItem, TextStyle, TextsOptions, TextsPrimitive,
};
use parking_lot::RwLock;
use triomphe::Arc;

use crate::convert_color;
use crate::generic_series::{GenericSeries, Point, SeriesWidget, fold, show_coordinates};

#[derive(Clone)]
pub struct TextPoint {
    x: i64,
    y: f64,
    text: RichText,
}

pub struct TextsWidget;

impl SeriesWidget for TextsWidget {
    type Fold = [TextPoint; 2];
    type Options = TextsOptions;
    type Point = TextPoint;

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
            esl_gpu::PlotItemData::Texts { primitives, .. } => primitives,
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
            esl_gpu::PlotItemData::Texts { options, .. } => options,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        let mut text = RichText::new(primitive.text.clone())
            .color(convert_color(options.color))
            .extra_letter_spacing(options.extra_letter_spacing)
            .background_color(convert_color(options.background_color));
        if let Some(size) = options.size {
            text = text.size(size);
        }
        if let Some(line_height) = options.line_height {
            text = text.line_height(Some(line_height));
        }
        if let Some(family) = &options.family {
            text = text.family(match family {
                FontFamily::Proportional => egui::FontFamily::Proportional,
                FontFamily::Monospace => egui::FontFamily::Monospace,
                FontFamily::Name(name) => egui::FontFamily::Monospace,
            });
        }
        if let Some(text_style) = options.text_style.clone() {
            text = text.text_style(match text_style {
                TextStyle::Small => egui::style::TextStyle::Small,
                TextStyle::Body => egui::style::TextStyle::Body,
                TextStyle::Monospace => egui::style::TextStyle::Monospace,
                TextStyle::Button => egui::style::TextStyle::Button,
                TextStyle::Heading => egui::style::TextStyle::Heading,
                TextStyle::Name(name) => egui::style::TextStyle::Monospace,
            });
        }
        if options.code {
            text = text.code();
        }
        if options.strong {
            text = text.strong();
        }
        if options.weak {
            text = text.weak();
        }
        if options.underline {
            text = text.underline();
        }
        if options.strikethrough {
            text = text.strikethrough();
        }
        if options.italics {
            text = text.italics();
        }
        if options.raised {
            text = text.raised();
        }
        TextPoint {
            x,
            y: primitive.value as f64,
            text,
        }
    }

    fn show(&self, plot_ui: &mut PlotUi, item: &esl_gpu::PlotItem, data: Vec<TextPoint>) {
        let options = match &item.data {
            esl_gpu::PlotItemData::Texts { options, .. } => options,
            _ => unsafe { std::hint::unreachable_unchecked() },
        };
        fn convert_align(align: &esl_gpu::Align) -> emath::Align {
            match align {
                esl_gpu::Align::Min => emath::Align::Min,
                esl_gpu::Align::Center => emath::Align::Center,
                esl_gpu::Align::Max => emath::Align::Max,
            }
        }
        for point in data {
            plot_ui.text(
                Text::new(PlotPoint::new(point.x as f64, point.y), point.text.clone()).anchor(
                    Align2([
                        convert_align(&options.align.0),
                        convert_align(&options.align.1),
                    ]),
                ),
            );
        }
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
            esl_gpu::PlotItemData::Texts { options, .. } => options.color,
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

impl Point for TextPoint {
    fn x(&self) -> f64 {
        self.x as f64
    }

    fn y(&self) -> f64 {
        self.y
    }
}
