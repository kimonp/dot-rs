//! Reprecents an edge connecting two nodes within a graph.

/// Minimum allowed edge weight.  In future implementations, user could set this.
/// Edge weight could be used when drawing to deletemine the stroke width of an edge.
pub const MIN_EDGE_WEIGHT: u32 = 1;

/// Minimum allowed edge length.  In future implementations, user could set this.
/// See function of edge length below: edge_length()
pub const MIN_EDGE_LENGTH: u32 = 1;

// EdgeDisposition indicates whether a edge is incoming our outgoing with respect to a particular node.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum EdgeDisposition {
    In,
    Out,
}

/// An edge connects to nodes in a graph, and points from src_node to dst_node.
#[derive(Debug, Eq, PartialEq)]
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
    in_spanning_tree: bool,
}

impl Edge {
    pub fn new(src_node: usize, dst_node: usize) -> Self {
        Edge {
            src_node,
            dst_node,
            weight: MIN_EDGE_WEIGHT,
            ignored: false,
            reversed: false,
            cut_value: None,
            in_spanning_tree: false,
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
        self.in_spanning_tree
    }

    pub fn set_in_spanning_tree(&mut self, value: bool) {
        self.in_spanning_tree = value;
    }
}
