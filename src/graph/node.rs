//! Reprecents a node (vertice) within a graph.

use std::{collections::HashSet, fmt::Display};

use super::edge::EdgeDisposition;
use crate::graph::network_simplex::SimplexNodeTarget;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Point {
    x: u32,
    y: u32,
}

const NODE_MIN_SEP_X: u32 = 100;
const NODE_MIN_SEP_Y: u32 = 100;

impl Point {
    pub fn new(x: u32, y: u32) -> Self {
        Self { x, y }
    }

    pub fn x(&self) -> u32 {
        self.x
    }

    pub fn y(&self) -> u32 {
        self.y
    }

    pub fn set_x(&mut self, x: u32) -> u32 {
        self.x = x;

        x
    }

    #[allow(unused)]
    pub fn set_y(&mut self, y: u32) -> u32 {
        self.y = y;

        y
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum NodeType {
    /// Part of the graph: these are the only nodes that are printed
    Real,
    /// Used so that edges never traverse more than one rank.
    RankFiller,
    /// Used to help calculate the x coordiante
    XCoordCalc,
}

impl NodeType {
    /// Return true if the node is a virtual node type.
    pub fn is_virtual(&self) -> bool {
        match self {
            NodeType::Real => false,
            NodeType::RankFiller => true,
            NodeType::XCoordCalc => true,
        }
    }
}

#[derive(Debug, Eq, PartialEq, Clone, Copy)]
pub struct SpanningTreeData {
    // Graph index id of the edge connects to the parent of this node.  None if this node has no tree parent.
    edge_idx_to_parent: Option<usize>,
    // The minimal tree index of all nodes for which this node is an ancestor.
    sub_tree_idx_min: usize,
    // The maximum tree index of all nodes for which this node is an ancestor.
    sub_tree_idx_max: usize,
}

impl SpanningTreeData {
    fn new(
        edge_idx_to_parent: Option<usize>,
        sub_tree_idx_min: usize,
        sub_tree_idx_max: usize,
    ) -> SpanningTreeData {
        SpanningTreeData {
            edge_idx_to_parent,
            sub_tree_idx_min,
            sub_tree_idx_max,
        }
    }
    
    pub fn edge_idx_to_parent(&self) -> Option<usize> {
        self.edge_idx_to_parent
    }

    pub fn sub_tree_idx_min(&self) -> usize {
        self.sub_tree_idx_min
    }

    pub fn sub_tree_idx_max(&self) -> usize {
        self.sub_tree_idx_max
    }
}

// Represents the node element of a graph.  Sometimes called a vertice.
//
// Nodes are connected together via Edges.  Each node has a list of edges coming in and edges
// going out of this node.  Note that this means that each edge is represented twice: Once in
// the outgoing node, and once in the incoming node.
#[derive(Debug, Eq, PartialEq, Clone)]
pub struct Node {
    // Arbitrary name set by the user.  Duplicates are possible, and up to the user to control.
    pub(super) name: String,
    // Rank is computed using the network simplex algorithm.  Used to determine both vertical_rank
    // and coordianates.x.
    pub(super) simplex_rank: Option<u32>,
    /// Relative verticial ranking of this node.  Zero based, greater numbers are lower.
    pub(super) vertical_rank: Option<u32>,
    /// Position is the relative horizontal position of this node compared to
    /// other nodes in the same rank.  Zero based, greater numbers are farther right.
    pub(super) horizontal_position: Option<usize>,
    /// Coordinates (x,y) of the node within the graph.
    pub(super) coordinates: Option<Point>,
    /// Edges incoming to this node.  Each entry is a edge index into the graph's edges list.
    pub(super) in_edges: Vec<usize>,
    /// Edges outcoming from this node.  Each entry is a edge index into the graph's edges list.
    pub(super) out_edges: Vec<usize>,

    /// True if this node is part of the "spanning" tree under consideration.  Used during ranking.
    spanning_tree: Option<SpanningTreeData>,
    /// Added as a placeholder node during position assignement or other part of graphinc
    pub(super) node_type: NodeType,
}

impl Node {
    /// Return a new node which is not yet connected to a graph.
    pub(super) fn new(name: &str) -> Self {
        Node {
            name: name.to_string(),
            simplex_rank: None,
            vertical_rank: None,
            horizontal_position: None,
            coordinates: None,
            in_edges: vec![],
            out_edges: vec![],
            spanning_tree: None,
            node_type: NodeType::Real,
        }
    }

    /// Return true of the node is one of the virtual node types.
    pub(super) fn is_virtual(&self) -> bool {
        self.node_type.is_virtual()
    }

    pub(super) fn in_spanning_tree(&self) -> bool {
        self.spanning_tree.is_some()
    }

    pub(super) fn spaning_tree(&self) -> Option<SpanningTreeData> {
        self.spanning_tree
    }

    /// If this node is in the tree, return the graph index of the parent.
    ///
    /// Note that None is retuned if either this node is not a tree node,
    /// or it has no parent.
    pub(super) fn spanning_tree_parent_edge_idx(&self) -> Option<usize> {
        if let Some(tree_data) = self.spanning_tree {
            tree_data.edge_idx_to_parent
        } else {
            None
        }
    }

    /// Set the given node's tree data as a root node of the tree.
    /// 
    /// Typically, nodes with no in_edges are set as root nodes.
    pub(super) fn set_tree_root_node(&mut self) {
        self.spanning_tree = Some(SpanningTreeData::new(None, 0, 0));
    }

    pub(super) fn clear_tree_data(&mut self) {
        self.spanning_tree = None;
    }

    pub(super) fn set_tree_data(&mut self, parent: Option<usize>, min: usize, max: usize) {
        self.spanning_tree = Some(SpanningTreeData::new(parent, min, max));
    }

    /// Set the node as a tree node, but don't set the parent, and set min or max to zero.
    /// 
    /// Used by asyclic tree for tree nodes, but does not use the other data.
    /// This must be cleared by simplex for it runs.
    pub(super) fn set_empty_tree_node(&mut self) {
        self.set_tree_data(None, 0, 0);
    }

    pub(super) fn set_coordinates(&mut self, x: u32, y: u32) {
        self.coordinates = Some(Point::new(x, y));
    }

    /// Add either an in our out edge to the node.
    pub(super) fn add_edge(&mut self, edge: usize, disposition: EdgeDisposition) {
        match disposition {
            EdgeDisposition::In => &self.in_edges.push(edge),
            EdgeDisposition::Out => &self.out_edges.push(edge),
        };
    }

    /// Minimum separation of x coordiante from a point to this node.
    ///
    /// TODO: For now this is just a constant, but in future
    ///       each node could differ.
    pub(super) fn min_separation_x(&self) -> u32 {
        NODE_MIN_SEP_X
    }

    /// Minimum separation of y coordiante from a point to this node.
    ///
    /// TODO: For now this is just a constant, but in future
    ///       each node could differ.
    pub(super) fn min_separation_y(&self) -> u32 {
        NODE_MIN_SEP_Y
    }

    /// Remove edge indexes from node.
    pub(super) fn clear_edges(&mut self) {
        self.in_edges = vec![];
        self.out_edges = vec![];
    }

    /// Return the list of In our Out edges.
    pub(super) fn get_edges(&self, disposition: EdgeDisposition) -> &Vec<usize> {
        match disposition {
            EdgeDisposition::In => &self.in_edges,
            EdgeDisposition::Out => &self.out_edges,
        }
    }

    /// Return all in and out edges associated with a node.
    pub(super) fn get_all_edges(&self) -> impl Iterator<Item = &usize> {
        self.out_edges.iter().chain(self.in_edges.iter())
    }

    /// Swap an edge from in_edges to out_edges or vice versa, depending on disposition.
    pub(super) fn swap_edge_in_list(&mut self, edge_idx: usize, disposition: EdgeDisposition) {
        let local_idx =
            if let Some(local_idx) = self.find_internal_edge_index(edge_idx, disposition) {
                local_idx
            } else {
                panic!("Could not find edge {disposition:?}:{edge_idx} to reverse in src node.");
            };

        match disposition {
            EdgeDisposition::In => {
                self.in_edges.remove(local_idx);
                self.out_edges.push(edge_idx);
            }
            EdgeDisposition::Out => {
                self.out_edges.remove(local_idx);
                self.in_edges.push(edge_idx);
            }
        }
    }

    fn find_internal_edge_index(
        &self,
        edge_idx_to_find: usize,
        disposition: EdgeDisposition,
    ) -> Option<usize> {
        let edge_list = self.get_edge_list(disposition);

        for (internal_idx, edge_idx) in edge_list.iter().enumerate() {
            if edge_idx_to_find == *edge_idx {
                return Some(internal_idx);
            }
        }

        None
    }

    fn get_edge_list(&self, disposition: EdgeDisposition) -> &Vec<usize> {
        match disposition {
            EdgeDisposition::In => &self.in_edges,
            EdgeDisposition::Out => &self.out_edges,
        }
    }

    // Return true if none of the incoming edges to node are in the set scanned_edges.
    // * If any incoming edges are not scanned, return false
    // * If there are no incoming edges, return true
    pub(super) fn no_unscanned_in_edges(&self, scanned_edges: &HashSet<usize>) -> bool {
        for edge_idx in self.get_edges(EdgeDisposition::In) {
            if !scanned_edges.contains(edge_idx) {
                return false;
            }
        }
        true
    }

    /// True if there are no incoming edges to a node.
    pub(super) fn no_in_edges(&self) -> bool {
        self.get_edges(EdgeDisposition::In).is_empty()
    }

    /// True if there are no outgoing edges to a node.
    pub(super) fn no_out_edges(&self) -> bool {
        self.get_edges(EdgeDisposition::Out).is_empty()
    }

    /// Sets the simplex rank of a node.
    ///
    /// Rank corresponds to the vertical placement of a node.  The greater the rank,
    /// the lower the placement on a canvas.
    pub(super) fn set_simplex_rank(&mut self, rank: Option<u32>) {
        self.simplex_rank = rank;
    }

    /// Gets the simplex rank of a node.
    pub(super) fn simplex_rank(&self) -> Option<u32> {
        self.simplex_rank
    }

    pub(super) fn assign_simplex_rank(&mut self, target: SimplexNodeTarget) {
        let simplex_rank = self.simplex_rank.unwrap();

        match target {
            SimplexNodeTarget::VerticalRank => self.vertical_rank = Some(simplex_rank),
            SimplexNodeTarget::XCoordinate => {
                if let Some(mut coords) = self.coordinates {
                    coords.x = simplex_rank;
                } else {
                    self.coordinates = Some(Point::new(simplex_rank, 0));
                }
            }
        };
    }
}

impl Display for Node {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let coords = if let Some(coords) = self.coordinates {
            format!("({:3},{:3})", coords.x(), coords.y())
        } else {
            "        ".to_string()
        };
        let v_rank = if let Some(v_rank) = self.vertical_rank {
            format!("vr:{v_rank:2}")
        } else {
            "     ".to_string()
        };
        let pos = if let Some(h_pos) = self.horizontal_position {
            format!("pos:{h_pos:2}")
        } else {
            "      ".to_string()
        };
        write!(fmt, "{}: {v_rank} {coords} {pos}", &self.name)
    }
}

impl Ord for Node {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.simplex_rank.cmp(&other.simplex_rank)
    }
}

impl PartialOrd for Node {
    fn partial_cmp(&self, other: &Node) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
