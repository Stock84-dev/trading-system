
#[cfg(not(target_os = "cuda"))]
impl Into<egui::style::TextStyle> for TextStyle {
    #[inline(always)]
    fn into(self) -> egui::style::TextStyle {
        match self {
            TextStyle::Small => egui::style::TextStyle::Small,
            TextStyle::Body => egui::style::TextStyle::Body,
            TextStyle::Monospace => egui::style::TextStyle::Monospace,
            TextStyle::Button => egui::style::TextStyle::Button,
            TextStyle::Heading => egui::style::TextStyle::Heading,
            TextStyle::Name(name) => egui::style::TextStyle::Monospace,
        }
    }
}

#[cfg(not(target_os = "cuda"))]
impl Into<egui::FontFamily> for FontFamily {
    #[inline(always)]
    fn into(self) -> egui::FontFamily {
        match self {
            FontFamily::Proportional => egui::FontFamily::Proportional,
            FontFamily::Monospace => egui::FontFamily::Monospace,
            FontFamily::Name(name) => egui::FontFamily::Monospace,
        }
    }
}
#[cfg(not(target_os = "cuda"))]
impl Into<emath::Align> for Align {
    #[inline(always)]
    fn into(self) -> emath::Align {
        match self {
            Align::Min => emath::Align::Min,
            Align::Center => emath::Align::Center,
            Align::Max => emath::Align::Max,
        }
    }
}
#[cfg(not(target_os = "cuda"))]
impl Into<egui_plot::LineStyle> for LineStyle {
    #[inline(always)]
    fn into(self) -> egui_plot::LineStyle {
        match self {
        }
    }
}
