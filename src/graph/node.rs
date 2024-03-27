//! Reprecents a node (vertice) within a graph.

use std::{cell::RefCell, collections::HashSet, fmt::Display};

use super::{
    edge::EdgeDisposition,
    edge::EdgeDisposition::{In, Out},
    network_simplex::sub_tree::SubTree,
};
use crate::graph::network_simplex::SimplexNodeTarget;

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub struct Point {
    x: i32,
    y: i32,
}

#[derive(Debug, Clone)]
pub struct Rect {
    min: Point,
    max: Point,
}

impl Rect {
    pub fn new(min: Point, max: Point) -> Self {
        Rect { min, max }
    }

    pub fn min(&self) -> Point {
        self.min
    }

    pub fn max(&self) -> Point {
        self.max
    }

    pub fn height(&self) -> i32 {
        println!("HEIGHT: {} vs {}", self.max.y(), self.min.y());
        self.max.y() - self.min.y()
    }

    pub fn width(&self) -> i32 {
        self.max.x() - self.min.x()
    }

    /// Normalize a point to be placed inside of this Rect.
    pub fn normalize(&self, point: Point) -> Point {
        Point::new(point.x() - self.min().x(), point.y() - self.min().y())
    }
}

/// Separation of nodes horizontally in pixels, assuming 72 pixels per inch.
pub(super) const NODE_MIN_SEP_X: i32 = 72;
const NODE_MIN_SEP_Y: i32 = 72;
pub(super) const NODE_START_HEIGHT: i32 = NODE_MIN_SEP_X / 4;

impl Point {
    pub fn new(x: i32, y: i32) -> Self {
        Self { x, y }
    }

    pub fn x(&self) -> i32 {
        self.x
    }

    pub fn y(&self) -> i32 {
        self.y
    }

    #[allow(unused)]
    pub fn set_x(&mut self, x: i32) -> i32 {
        self.x = x;

        x
    }

    pub fn set_y(&mut self, y: i32) -> i32 {
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

#[derive(Debug, Eq, PartialEq, Clone)]
pub struct SpanningTreeData {
    edge_idx_to_parent: Option<usize>,
    // minimum distance to the root node of all nodes under this node.
    tree_dist_min: Option<usize>,
    // maximum distance to the root node of all nodes under this node.
    tree_dist_max: Option<usize>,
    /// Reference to the subtree that this node is a member of.
    sub_tree: Option<SubTree>,
}

impl Display for SpanningTreeData {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let parent = self
            .edge_idx_to_parent()
            .map(|p| p.to_string())
            .unwrap_or("-".to_string());
        let min = self
            .tree_dist_min
            .map(|m| m.to_string())
            .unwrap_or("-".to_string());
        let max = self
            .tree_dist_max
            .map(|m| m.to_string())
            .unwrap_or("-".to_string());

        write!(fmt, "edge_to_parent:{parent} min:{min} max:{max}")
    }
}

impl SpanningTreeData {
    #[cfg(test)]
    fn new(
        edge_idx_to_parent: Option<usize>,
        sub_tree_idx_min: Option<usize>,
        sub_tree_idx_max: Option<usize>,
    ) -> SpanningTreeData {
        SpanningTreeData {
            edge_idx_to_parent,
            tree_dist_min: sub_tree_idx_min,
            tree_dist_max: sub_tree_idx_max,
            sub_tree: None,
        }
    }

    /// Graph index id of the edge that connects to the parent of this node.
    ///
    /// None if this node has no tree parent.
    pub fn edge_idx_to_parent(&self) -> Option<usize> {
        self.edge_idx_to_parent
    }

    /// The minimal distance to the root of all nodes for which this node is an ancestor.
    pub fn tree_dist_min(&self) -> Option<usize> {
        self.tree_dist_min
    }

    /// The maximum distance to the root of all nodes for which this node is an ancestor.
    pub fn tree_dist_max(&self) -> Option<usize> {
        self.tree_dist_max
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
    simplex_rank: RefCell<Option<i32>>,
    /// Relative verticial ranking of this node.  Zero based, greater numbers are lower.
    pub(super) vertical_rank: Option<i32>,
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
    spanning_tree: RefCell<Option<SpanningTreeData>>,
    /// Added as a placeholder node during position assignement or other part of graphinc
    pub(super) node_type: NodeType,
}

impl Node {
    /// Return a new node which is not yet connected to a graph.
    pub(super) fn new(name: &str) -> Self {
        Node {
            name: name.to_string(),
            simplex_rank: RefCell::new(None),
            vertical_rank: None,
            horizontal_position: None,
            coordinates: None,
            in_edges: vec![],
            out_edges: vec![],
            spanning_tree: RefCell::new(None),
            node_type: NodeType::Real,
        }
    }

    /// Return the x,y coordinates of the node if it has been set.
    ///
    /// Once a graph has been layed out, all node coordinates should
    /// be set.
    pub fn coordinates(&self) -> Option<Point> {
        self.coordinates
    }

    /// Remove edge_idx from either the in_edges or out_edges depending on disposition.
    ///
    /// Return the position of the removed edge, or None if it could not be found.
    pub fn remove_edge(&mut self, edge_idx: usize, disposition: EdgeDisposition) -> Option<usize> {
        let mut edges_iter = match disposition {
            In => self.in_edges.iter_mut(),
            Out => self.in_edges.iter_mut(),
        };
        if let Some(position) = edges_iter.position(|x| *x == edge_idx) {
            let edges = match disposition {
                In => &mut self.in_edges,
                Out => &mut self.out_edges,
            };

            Some(edges.swap_remove(position))
        } else {
            None
        }
    }

    /// Return true of the node is one of the virtual node types.
    pub fn is_virtual(&self) -> bool {
        self.node_type.is_virtual()
    }

    /// Return the name of the node.
    pub fn name(&self) -> &str {
        &self.name
    }

    pub(super) fn in_spanning_tree(&self) -> bool {
        self.spanning_tree.borrow().is_some()
    }

    pub(super) fn spanning_tree(&self) -> Option<SpanningTreeData> {
        self.spanning_tree.borrow().clone()
    }

    /// If this node is in the tree, return the graph index of the parent.
    ///
    /// Note that None is retuned if either this node is not a tree node,
    /// or it has no parent.
    pub(super) fn spanning_tree_parent_edge_idx(&self) -> Option<usize> {
        if let Some(tree_data) = self.spanning_tree() {
            tree_data.edge_idx_to_parent
        } else {
            None
        }
    }

    pub(super) fn tree_dist_max(&self) -> Option<usize> {
        if let Some(tree_data) = self.spanning_tree() {
            tree_data.tree_dist_max()
        } else {
            None
        }
    }

    pub(super) fn tree_dist_min(&self) -> Option<usize> {
        if let Some(tree_data) = self.spanning_tree() {
            tree_data.tree_dist_min()
        } else {
            None
        }
    }

    pub(super) fn set_tree_dist_min(&self, min: Option<usize>) {
        if let Some(data) = &mut self.spanning_tree.borrow_mut().as_mut() {
            data.tree_dist_min = min
        } else {
            panic!("trying to set tree_dist_min but node not in spanning tree");
        }
    }

    pub(super) fn set_tree_dist_max(&self, max: Option<usize>) {
        if let Some(data) = &mut self.spanning_tree.borrow_mut().as_mut() {
            data.tree_dist_max = max
        } else {
            panic!("trying to set tree_dist_max but node not in spanning tree");
        }
    }

    /// Set the given node's tree data as a root node of the tree.
    ///
    /// Typically, nodes with no in_edges are set as root nodes.
    #[cfg(test)]
    pub(super) fn set_tree_root_node(&self) {
        *self.spanning_tree.borrow_mut() = Some(SpanningTreeData::new(None, None, None));
    }

    pub(super) fn clear_tree_data(&self) {
        *self.spanning_tree.borrow_mut() = None;
    }

    pub(super) fn set_tree_data(
        &self,
        parent: Option<usize>,
        min: Option<usize>,
        max: Option<usize>,
    ) {
        *self.spanning_tree.borrow_mut() = Some(SpanningTreeData {
            edge_idx_to_parent: parent,
            tree_dist_min: min,
            tree_dist_max: max,
            sub_tree: self.sub_tree(),
        });
    }

    fn tree_data(&self) -> Option<SpanningTreeData> {
        self.spanning_tree.borrow().clone()
    }

    /// Return a internally mutable subtree if one is set for this node.
    pub(super) fn sub_tree(&self) -> Option<SubTree> {
        self.spanning_tree
            .borrow()
            .as_ref()
            .and_then(|data| data.sub_tree.clone())
    }

    /// Return true if this node has a sub_tree set.
    pub(super) fn has_sub_tree(&self) -> bool {
        self.sub_tree().is_some()
    }

    /// Return a internally mutable subtree if one is set for this node.
    pub(super) fn set_sub_tree(&self, sub_tree: SubTree) {
        if !self.in_spanning_tree() {
            self.set_empty_tree_node();
        }

        if let Some(data) = &mut self.spanning_tree.borrow_mut().as_mut() {
            data.sub_tree = Some(sub_tree);
        } else {
            panic!("trying to set sub_tree but node not in spanning tree");
        }
    }

    /// Set the node as a tree node, but don't set the parent, and set min or max to zero.
    ///
    /// Used by asyclic tree for tree nodes, but does not use the other data.
    /// This must be cleared by simplex for it runs.
    pub(super) fn set_empty_tree_node(&self) {
        self.set_tree_data(None, None, None);
    }

    pub(super) fn set_coordinates(&mut self, x: i32, y: i32) {
        self.coordinates = Some(Point::new(x, y));
    }

    // pub(super) fn set_y_coordinate(&mut self, y: i32) {
    //     let prev_x = self.coordinates().unwrap_or(Point::new(0,0)).x();
    //     self.set_coordinates(prev_x, y)
    // }

    /// Add either an in our out edge to the node.
    pub(super) fn add_edge(&mut self, edge: usize, disposition: EdgeDisposition) {
        match disposition {
            In => &self.in_edges.push(edge),
            Out => &self.out_edges.push(edge),
        };
    }

    /// Minimum separation of x coordiante from a point to this node.
    ///
    /// TODO: For now this is just a constant, but in future
    ///       each node could differ.
    pub(super) fn min_separation_x(&self) -> i32 {
        NODE_MIN_SEP_X
    }

    /// Minimum separation of y coordiante from a point to this node.
    ///
    /// TODO: For now this is just a constant, but in future
    ///       each node could differ.
    pub(super) fn min_separation_y(&self) -> i32 {
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
            In => &self.in_edges,
            Out => &self.out_edges,
        }
    }

    /// Return all in and out edges associated with a node.
    pub(super) fn get_all_edges(&self) -> impl Iterator<Item = &usize> {
        self.out_edges.iter().chain(self.in_edges.iter())
    }

    /// Return all in and out edges associated with a node in tuple that includes
    /// the disposition (in our out) of each edge.
    pub(super) fn get_all_edges_with_disposition(
        &self,
        out_first: bool,
    ) -> impl Iterator<Item = (&usize, EdgeDisposition)> {
        let (first_edges, second_edges, first_dir, second_dir) = if out_first {
            (&self.out_edges, &self.in_edges, Out, In)
        } else {
            (&self.in_edges, &self.out_edges, In, Out)
        };

        first_edges
            .iter()
            .map(move |edge_idx| (edge_idx, first_dir))
            .chain(
                second_edges
                    .iter()
                    .map(move |edge_idx| (edge_idx, second_dir)),
            )
    }

    /// Swap an edge from in_edges to out_edges or vice versa, depending on disposition.
    pub(super) fn swap_edge_in_list(&mut self, edge_idx: usize, disposition: EdgeDisposition) {
        let local_idx =
            if let Some(local_idx) = self.find_internal_edge_index(edge_idx, disposition) {
                local_idx
            } else {
                panic!("Could not find edge {disposition:?}:{edge_idx} to reverse in src node.");
            };

        println!("SWAPPING EDGE for node {}: {edge_idx}", self.name());
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

    // /// True if there are no outgoing edges to a node.
    // pub(super) fn no_out_edges(&self) -> bool {
    //     self.get_edges(EdgeDisposition::Out).is_empty()
    // }

    /// Sets the simplex rank of a node.
    ///
    /// Rank corresponds to the vertical placement of a node.  The greater the rank,
    /// the lower the placement on a canvas.
    pub(super) fn set_simplex_rank(&self, rank: Option<i32>) {
        *self.simplex_rank.borrow_mut() = rank;
    }

    /// Gets the simplex rank of a node.
    pub(super) fn simplex_rank(&self) -> Option<i32> {
        *self.simplex_rank.borrow()
    }

    pub(super) fn assign_simplex_rank(&mut self, target: SimplexNodeTarget) {
        let simplex_rank = self.simplex_rank().unwrap();

        match target {
            SimplexNodeTarget::VerticalRank => self.vertical_rank = Some(simplex_rank),
            SimplexNodeTarget::XCoordinate => {
                self.assign_x_coord(simplex_rank);
            }
        };
    }

    pub(super) fn assign_x_coord(&mut self, x: i32) {
        if let Some(coords) = self.coordinates {
            self.coordinates = Some(Point::new(x, coords.y()));
        } else {
            self.coordinates = Some(Point::new(x, 0));
        }
    }
}

impl Display for Node {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let tree = self.tree_data().map(|td| td.to_string());
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
        write!(fmt, "{} ({tree:?}): {v_rank} {coords} {pos}", &self.name)
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
