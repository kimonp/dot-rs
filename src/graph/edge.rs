//! Represents an edge connecting two nodes within a graph.

use std::{cell::RefCell, fmt::Display};

/// Minimum allowed edge weight.  In future implementations, user could set this.
/// Edge weight could be used when drawing to determine the stroke width of an edge.
pub const MIN_EDGE_WEIGHT: u32 = 1;

/// Minimum allowed edge length.  In future implementations, user could set this.
/// See function of edge length below: edge_length()
pub const MIN_EDGE_LENGTH: i32 = 1;

// EdgeDisposition indicates whether a edge is incoming our outgoing with respect to a particular node.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EdgeDisposition {
    In,
    Out,
}

impl EdgeDisposition {
    pub fn opposite(&self) -> Self {
        match self {
            EdgeDisposition::In => EdgeDisposition::Out,
            EdgeDisposition::Out => EdgeDisposition::In,
        }
    }
}

impl Display for EdgeDisposition {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let disp = match self {
            EdgeDisposition::In => "in",
            EdgeDisposition::Out => "out",
        };
        write!(fmt, "{disp}")
    }
}

/// An edge connects to nodes in a graph, and points from src_node to dst_node.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Edge {
    /// Node that this edge points from.  This is an index into graph.nodes.
    pub src_node: usize,
    /// Node that this edge points to.  This is an index into graph.nodes.
    pub dst_node: usize,

    /// The following fields are used for rank calculation:

    /// Weight of the edge,
    pub weight: u32,

    /// If this edge is ignored internally while calculating node ranks.
    /// TODO: NOTHING CURRENTLY TAKES IGNORED INTO ACCOUNT
    pub ignored: bool,
    /// If this edge is reversed internally while calculating node ranks.
    pub reversed: bool,
    /// Used as port of the algorithm to rank the nodes of the graph.
    pub cut_value: Option<i32>,
    /// True if this edge is part of the feasible tree use to calculate cut values.
    in_spanning_tree: RefCell<bool>,
    /// Minimum length of this edge.  Critical for calculating slack.
    min_len: i32,
}

impl Edge {
    pub fn new(src_node: usize, dst_node: usize) -> Self {
        Self::new_with_details(src_node, dst_node, MIN_EDGE_LENGTH, MIN_EDGE_WEIGHT)
    }

    pub fn new_with_details(src_node: usize, dst_node: usize, min_len: i32, weight: u32) -> Self {
        Edge {
            src_node,
            dst_node,
            weight,
            ignored: false,
            reversed: false,
            cut_value: None,
            in_spanning_tree: RefCell::new(false),
            min_len,
        }
    }

    pub fn edge_omega_value(src_is_virtual: bool, dst_is_virtual: bool) -> u32 {
        match (src_is_virtual, dst_is_virtual) {
            (false, false) => 0_u32,
            (true, false) => 2,
            (false, true) => 2,
            (true, true) => 8,
        }
    }

    pub fn in_spanning_tree(&self) -> bool {
        *self.in_spanning_tree.borrow()
    }

    pub fn set_in_spanning_tree(&self, value: bool) {
        *self.in_spanning_tree.borrow_mut() = value;
    }

    pub fn min_len(&self) -> i32 {
        self.min_len
    }
}
