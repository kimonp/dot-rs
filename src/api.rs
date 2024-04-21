//! Top level api methods for dot-rs.

use crate::{graph::{Graph, Snapshots}, svg::{SvgStyle, SVG}};

/// Given a valid dot string, return an svg string.
/// 
/// panics if dot string is invalid.
pub fn dot_to_svg(dot: &str) -> String {
    let mut graph = Graph::from(dot);
    
    graph.layout_nodes();
    
    let svg = SVG::new(graph, SvgStyle::Production);
    
    svg.to_string()
}

/// Given a valid dot string, return all the debug snapshots made during layout.
///
/// Returned as a vector of: (title, SVG)
pub fn dot_to_svg_debug_snapshots(dot: &str) -> Snapshots {
    let mut graph = Graph::from(dot);
    
    graph.layout_nodes();
    graph.get_debug_svg_snapshots()
}