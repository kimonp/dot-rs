//! Top level api methods for dot-rs.

use crate::{graph::Graph, svg::SVG};

/// Given a valid dot string, return an svg string.
/// 
/// panics if dot string is invalid.
pub fn dot_to_svg(dot: &str) -> String {
    let mut graph = Graph::from(dot);
    
    graph.layout_nodes();
    
    let svg = SVG::new(graph, false);
    
    svg.to_string()
}