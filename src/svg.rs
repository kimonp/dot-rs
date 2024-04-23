//! For transforming a graph to SVG.

use std::fmt::{Display, Error, Formatter};
use std::fs::File;
use std::io::prelude::*;
use std::result::Result;

use crate::graph::node::{Node, Point, Rect};
use crate::graph::Graph;

pub struct SVG {
    graph: Graph,
    style: SvgStyle,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SvgStyle {
    Production,
    MinCross,
    Simplex,
}

impl SvgStyle {
    fn show_virtual(&self) -> bool {
        *self != SvgStyle::Production
    }

    fn show_simplex(&self) -> bool {
        *self == SvgStyle::Simplex
    }

    fn oval_nodes(&self) -> bool {
        *self == SvgStyle::Simplex
    }

    fn show_coordinates(&self) -> bool {
        self.oval_nodes()
    }
}

impl SVG {
    pub fn new(graph: Graph, style: SvgStyle) -> Self {
        Self { graph, style }
    }

    /// Write out the graph as is to the given file name (with an svg suffix).
    pub fn write_to_file(&self, name: &str) {
        let svg = self.to_string();
        let mut file = File::create(format!("{name}.svg")).unwrap();
        file.write_all(svg.as_bytes()).unwrap();
    }
}

fn normalized_coords(node: &Node, node_rect: &Rect) -> (f64, f64) {
    let coords = node.coordinates().expect("All nodes must have coordinates");
    let point = node_rect.normalize(coords);

    (point.x() as f64, point.y() as f64)
}

impl Display for SVG {
    fn fmt(&self, fmt: &mut Formatter<'_>) -> Result<(), Error> {
        let style = self.style;
        // const DEFAULT_X: usize = 1;
        const DEFAULT_Y: i32 = 1;
        let node_rect = self.graph.graph_rect();

        let bezel = (self.graph.horizontal_node_separation() / 2) as f64;
        let height = node_rect.height() as f64;
        let width = node_rect.width() as f64;
        //let width = box_width * x_scale;
        // let height = box_height * y_scale;
        let px_size = 1_f64;

        let vb_width = width + (2.0 * bezel);
        let vb_height = height + (2.0 * bezel);

        // To set the coordinates within the svg We set the viewbox's height and width
        let mut svg = vec![
            format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{vb_width}" height="{vb_height}"
                    viewBox="-{bezel} -{bezel} {vb_width} {vb_height}"
                    preserveAspectRatio="xMinYMin meet"
                    >"#
            ),
            format!(
                r#"<defs>
                    <marker
                        id="arrow-head"
                        orient="auto"
                        viewBox="0 0 15 10"
                        refX="15" refY="5"
                        markerWidth="10" markerHeight="15"
                    >
                        <path d="M0,0 L15,5 L0,10 Z" fill="black"/>
                    </marker>
                </defs>"#
            ),
        ];

        let real_radius = (self.graph.horizontal_node_separation() / 4) as f64 * 0.8;
        let virtual_radius = if style.show_virtual() {
            real_radius
        } else {
            0.0
        };

        for edge in self.graph.edges_iter() {
            let reversed = edge.reversed;
            let (src_node_idx, dst_node_idx) = if reversed {
                (edge.dst_node, edge.src_node)
            } else {
                (edge.src_node, edge.dst_node)
            };
            let src_node = self.graph.get_node(src_node_idx);
            let (src_x, src_y) = normalized_coords(src_node, &node_rect);

            let dst_node = self.graph.get_node(dst_node_idx);
            let (mut dst_x, mut dst_y) = normalized_coords(dst_node, &node_rect);

            let slope = (src_y - dst_y) / (src_x - dst_x); // # TODO
            let theta = slope.atan();
            let show_dst = !dst_node.is_virtual() || style.show_virtual();
            let node_radius = if show_dst {
                // TODO: This is just an approximation if the node
                //       is shows as an ellipse with x_radius=1.5*y_radius
                let node_radius_x = real_radius * 1.5 * 0.8;

                if style.oval_nodes() {
                    node_radius_x
                } else {
                    real_radius
                }
            } else {
                virtual_radius
            };

            let y_offset = theta.sin() * node_radius;
            let x_offset = theta.cos() * node_radius;

            if (slope < 0.0 && !reversed) || (slope > 0.0 && reversed) {
                dst_x += x_offset;
                dst_y += y_offset;
            } else {
                dst_x -= x_offset;
                dst_y -= y_offset;
            }

            let line_color = if style.show_simplex() && !edge.in_spanning_tree() {
                "red"
            } else {
                "black"
            };
            let marker = if show_dst {
                r#"marker-end="url(#arrow-head)""#
            } else {
                ""
            };

            svg.push(format!(
                r#"<path {marker} d="M{src_x} {src_y} L{dst_x} {dst_y}" stroke="{line_color}" stroke-width="{px_size}px"/>"#
            ));

            if style.show_simplex() {
                let font_size = 12.0;
                let font_style = format!("font-size:{font_size}; text-anchor: left");
                let label_x = (src_x + dst_x) / 2.0 + (font_size / 3.0);
                let label_y = (src_y + dst_y) / 2.0 + (font_size / 3.0);
                let edge_label = if let Some(cut_value) = edge.cut_value {
                    format!("{cut_value}")
                } else {
                    "Null".to_string()
                };

                svg.push(format!(
                    r#"<text x="{label_x}" y="{label_y}" style="{font_style}">{edge_label}</text>"#
                ));
            }
        }

        for node in self.graph.nodes_iter() {
            let (x, y) = normalized_coords(node, &node_rect);
            let label = self.graph.get_node_label(node.name());
            let node_name = if let Some(label) = label {
                label
            } else {
                node.name()
            };
            let name = if style.show_coordinates() {
                format!(
                    "{}: {:?},{:?}",
                    node_name,
                    node.coordinates()
                        .unwrap_or(Point::new(DEFAULT_Y, DEFAULT_Y))
                        .x(),
                    node.coordinates()
                        .unwrap_or(Point::new(DEFAULT_Y, DEFAULT_Y))
                        .y(),
                )
            } else {
                node_name.to_string()
            };
            let oval_nodes = style.oval_nodes() || name.len() > 3;
            let font_size = if oval_nodes { 12.0 } else if node.is_virtual() { 14.0 } else  { 24.0 };
            let font_style = format!("font-size:{font_size}; text-anchor: middle");
            let label_x = x;
            let label_y = y as f32 + (font_size / 3.0);

            let show_node = !node.is_virtual() || style.show_virtual();
            let node_radius = if show_node {
                real_radius
            } else {
                virtual_radius
            };
            let fill = if node.is_virtual() {
                "lightgray"
            } else {
                "skyblue"
            };
            let node_style = format!("fill: {fill}; stroke: black; stroke-width: {px_size}px;");

            if oval_nodes {
                let node_radius_x = node_radius * 2.0;

                svg.push(format!(
                    r#"<ellipse cx="{x}" cy="{y}" ry="{node_radius}" rx="{node_radius_x}" style="{node_style}"/>"#
                ));
            } else {
                svg.push(format!(
                    r#"<circle cx="{x}" cy="{y}" r="{node_radius}" style="{node_style}"/>"#
                ));
            }

            // svg.push(format!(
            //     r#"<svg viewBox="{rect_x} {rect_y} {rect_width} {rect_height}">"#
            // ));
            // svg.push(format!(
            //     r#"<rect x="0" y="0" height="1" width="1" rx=".001" style="{rect_style}"/>"#
            // ));
            if show_node {
                svg.push(format!(
                    r#"<text x="{label_x}" y="{label_y}" style="{font_style}">{name}</text>"#
                ));
            }
            // svg.push("</svg>".to_string());

            // if true {
            //     break;
            // }
        }

        svg.push("</svg>".to_string());

        writeln!(fmt, "{}", svg.join("\n"))
    }
}
