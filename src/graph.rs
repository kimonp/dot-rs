use std::{
    collections::{HashMap, HashSet},
    fmt::Display,
};

const MIN_EDGE_LENGTH: u32 = 1;
const MIN_EDGE_WEIGHT: u32 = 1;

/// Simplist posible representation of a graph until more is needed.
/// 
/// Chose to use indexed arrays to avoid interior mutability for now,
/// as well as maps or sets.  Both could change when I see the need.
#[derive(Debug)]
pub struct Graph {
    /// All nodes in the graph.
    nodes: Vec<Node>,
    /// All edges in the graph.
    edges: Vec<Edge>,
}

/// Enum used for calculating cut values for edges.
/// To do so, graph nodes need to be placed in either a head or tail component.
#[derive(Eq, PartialEq)]
enum CutSet {
    /// Head component of a cut set
    Head,
    /// Tail component of a cut set
    Tail,
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(unused)]
impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
            edges: vec![],
        }
    }

    pub fn get_node(&self, node_idx: usize) -> &Node {
        &self.nodes[node_idx]
    }

    pub fn get_node_mut(&mut self, node_idx: usize) -> &mut Node {
        &mut self.nodes[node_idx]
    }

    pub fn get_edge(&self, edge_idx: usize) -> &Edge {
        &self.edges[edge_idx]
    }

    pub fn get_edge_mut(&mut self, edge_idx: usize) -> &mut Edge {
        &mut self.edges[edge_idx]
    }

    // Return the node connected to this node by the given edge.
    pub fn get_connected_node(&self, node_idx: usize, edge_idx: usize) -> Option<usize> {
        let edge = &self.edges[edge_idx];

        if node_idx == edge.dst_node {
            Some(edge.src_node)
        } else if node_idx == edge.src_node {
            Some(edge.dst_node)
        } else {
            None
        }
    }

    /// Add a new node identified by name, and return the node's index in the graph.
    pub fn add_node(&mut self, name: &str) -> usize {
        let new_node = Node::new(name);
        let idx = self.nodes.len();
        self.nodes.push(new_node);

        idx
    }

    fn display_edge(&mut self, edge_idx: usize) -> String {
        let edge = self.get_edge(edge_idx);
        let src = &self.get_node(edge.src_node).name;
        let dst = &self.get_node(edge.dst_node).name;

        format!("{src} -> {dst}")
    }

    /// Add a new edge between two nodes, and return the edge's index in the graph.
    pub fn add_edge(&mut self, src_node: usize, dst_node: usize) -> usize {
        let new_edge = Edge::new(src_node, dst_node);
        let idx = self.edges.len();
        self.edges.push(new_edge);

        self.nodes[src_node].add_edge(idx, EdgeDisposition::Out);
        self.nodes[dst_node].add_edge(idx, EdgeDisposition::In);

        idx
    }

    #[allow(unused)]
    pub fn draw_graph(&mut self) {
        self.rank();
        self.ordering();
        self.position();
        self.make_splines()
    }

    /// Rank nodes in the graph using the network simplex algorithm described in [TSE93].
    pub fn rank(&mut self) {
        self.merge_edges_and_ignore_loops();

        self.set_feasible_tree();
        // while let Some(e) = self.leave_edge() {
        //     let f = self.enter_edge(e);
        //     self.exchange(e, f);
        // }
        // self.normalize();
        // self.balance();
    }

    fn merge_edges_and_ignore_loops(&mut self) {
        self.merge_edges();
        self.ignore_self_loops();
    }

    /// TODO: Consider doing this when the edge is added...
    fn ignore_self_loops(&mut self) {
        for edge in self.edges.iter_mut() {
            if edge.src_node == edge.dst_node {
                edge.ignored = true;
            }
        }
    }

    /// Merge any redundant edges by marking them ignored and adding their weight to the matching edge.
    ///
    /// TODO: Consider doing this when the edge is added...
    fn merge_edges(&mut self) {
        let mut duped_edges = vec![];
        let mut checked_edges = HashMap::new();

        // Find all duplicate edges (same src and dst)
        for (edge_idx, edge) in self.edges.iter().enumerate() {
            let route = (edge.src_node, edge.dst_node);

            if let Some(heavy_edge_idx) = checked_edges.get(&route) {
                duped_edges.push((*heavy_edge_idx, edge_idx));
            } else {
                checked_edges.insert(route, edge_idx);
            }
        }

        // Consolidate duped edges
        for (heavy_edge_idx, ignore_edge_idx) in duped_edges {
            let additional_weight = {
                let edge = self.get_edge_mut(ignore_edge_idx);
                edge.ignored = true;

                edge.weight
            };
            self.get_edge_mut(heavy_edge_idx).weight += additional_weight;
        }
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Sets a feasible tree within the given graph by setting feasible_tree_member on tree member nodes.
    ///
    /// Documentation from the paper:
    /// * The while loop code below ﬁnds an edge to a non-tree node that is adjacent to the tree, and adjusts the ranks of
    ///   the tree nodes to make this edge tight.
    ///   * As the edge was picked to have minimal slack, the resulting ranking is still feasible.
    ///   * Thus, on every iteration, the maximal tight tree gains at least one node, and the algorithm
    ///     eventually terminates with a feasible spanning tree.
    /// * This technique is essentially the one described by Sugiyama et al [STT]:
    ///   * Sugiyama, K., S. Tagawa and M. Toda, ‘‘Methods for Visual Understanding of Hierarchical System Structures,’’
    ///   * IEEE Transactions on Systems, Man, and Cybernetics SMC-11(2), February, 1981, pp. 109-125.
    ///
    /// ChatGPT:
    /// * In graph theory, an "edge incident on a tree" refers to an edge that connects a vertex of the tree to a vertex outside the tree.
    /// * A tree is a specific type of graph that is connected and acyclic, meaning it doesn't contain any cycles.
    ///   * The edges in a tree connect the vertices (nodes) in such a way that there is exactly one path between any two vertices.
    ///
    fn set_feasible_tree(&mut self) {
        self.init_rank();

        for node in self.nodes.iter_mut() {
            if node.no_out_edges() {
                node.feasible_tree_member = true;
            }
        }

        while self.tight_tree() < self.node_count() {
            // e = a non-tree edge incident on the tree with a minimal amount of slack
            // delta = slack(e);
            // if includent_node is e.head then delta = -delta
            // for v in Tree do v.rank = v.rank + delta;
            let edge_idx = self
                .get_min_incident_edge()
                .expect("No incident edges left!");
            let mut delta = self.slack(edge_idx).expect("Can't calculate slack on edge");

            if self.edge_head_is_incident(edge_idx) {
                delta = -delta;
            }

            for node in self.nodes.iter_mut() {
                if node.feasible_tree_member {
                    let cur_rank = node.rank.expect("Node does not have rank");
                    node.rank = Some(cur_rank + delta as u32)
                }
            }

            let node_idx = self
                .get_incident_node(edge_idx)
                .expect("Edge is not incident");
            self.get_node_mut(node_idx).feasible_tree_member = true;
            self.get_edge_mut(edge_idx).feasible_tree_member = true;

            println!("Set edge to tree: {}", self.display_edge(edge_idx));
        }
        self.init_cutvalues();
    }

    fn init_cutvalues(&mut self) {
        for edge_idx in 0..self.edges.len() {
            let edge = self.get_edge(edge_idx);
            if edge.feasible_tree_member {
                let (head_nodes, tail_nodes) = self.get_components(edge_idx);
                let cut_value = self.transition_weight_sum(&head_nodes, &tail_nodes);

                self.get_edge_mut(edge_idx).cut_value = Some(cut_value);

                println!(
                    "Set cut value for {}: {cut_value}\n  heads: {}\n  tails: {}",
                    self.display_edge(edge_idx),
                    self.display_component(&head_nodes),
                    self.display_component(&tail_nodes),
                )
            }
        }
    }

    /// Given a feasible spanning tree, we can associate an integer cut value with each tree edge as follows.
    /// * If the tree edge is deleted, the tree breaks into two connected components:
    ///   * the tail component containing the tail node of the edge,
    ///   * and the head component containing the head node.
    /// * The cut value is deﬁned as the sum of the weights of all edges from the tail component to the head component,
    ///   including the tree edge, minus the sum of the weights of all edges from the head component to the tail component.
    fn get_components(&self, edge_idx: usize) -> (HashSet<usize>, HashSet<usize>) {
        let head_component = self.collect_component_set(edge_idx, CutSet::Head, &HashSet::new());
        let tail_component = self.collect_component_set(edge_idx, CutSet::Tail, &head_component);

        (head_component, tail_component)
    }

    fn display_component(&self, comp: &HashSet<usize>) -> String {
        let mut node_names = vec![];
        for node_idx in comp {
            node_names.push(self.get_node(*node_idx).name.clone());
        }
        node_names.sort();

        format!("{node_names:?}",)
    }

    fn collect_component_set(
        &self,
        cut_edge_idx: usize,
        set: CutSet,
        opposite: &HashSet<usize>,
    ) -> HashSet<usize> {
        let mut component_set = HashSet::new();
        let cut_edge = self.get_edge(cut_edge_idx);

        let node_idx = if set == CutSet::Head {
            cut_edge.dst_node
        } else {
            cut_edge.src_node
        };

        let mut candidate_queue = vec![node_idx];
        while !candidate_queue.is_empty() {
            while let Some(node_idx) = candidate_queue.pop() {
                let node = self.get_node(node_idx);

                if node.feasible_tree_member {
                    component_set.insert(node_idx);

                    for edge_idx in node.get_all_edges() {
                        let edge = self.get_edge(*edge_idx);

                        if edge.feasible_tree_member && *edge_idx != cut_edge_idx {
                            let candidate_node_idx = self
                                .get_connected_node(node_idx, *edge_idx)
                                .expect("edge not connected");

                            if !component_set.contains(&candidate_node_idx)
                                && !opposite.contains(&candidate_node_idx)
                            {
                                candidate_queue.push(candidate_node_idx);
                            }
                        }
                    }
                }
            }
        }
        component_set
    }

    /// Given two sets of nodes, return the sum of all edges that move from the head set to the tail set.
    fn transition_weight_sum(
        &self,
        head_nodes: &HashSet<usize>,
        tail_nodes: &HashSet<usize>,
    ) -> i32 {
        let mut sum = 0;
        for edge in self.edges.iter() {
            if head_nodes.contains(&edge.src_node) && tail_nodes.contains(&edge.dst_node) {
                sum -= edge.weight as i32;
            } else if tail_nodes.contains(&edge.src_node) && head_nodes.contains(&edge.dst_node) {
                sum += edge.weight as i32;
            }
        }
        sum
    }

    // True if the head (src_node) of the given edge is not in the feasible tree.
    fn edge_head_is_incident(&self, edge_idx: usize) -> bool {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);

        !src_node.feasible_tree_member && dst_node.feasible_tree_member
    }

    /// edge_index is expected to span two nodes, one of which is in the tree, one of which is not.
    /// Return the index to the node which is not yet in the tree.
    fn get_incident_node(&self, edge_idx: usize) -> Option<usize> {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);

        if !src_node.feasible_tree_member && dst_node.feasible_tree_member {
            Some(edge.src_node)
        } else if src_node.feasible_tree_member && !dst_node.feasible_tree_member {
            Some(edge.dst_node)
        } else {
            None
        }
    }

    /// Documentation from the paper:
    /// * The function tight_tree() ﬁnds a maximal tree of tight edges containing some ﬁxed node.
    ///   * tight_tree() returns the number of nodes in the tree.
    /// * Note that such a maximal tree is just a spanning tree for the subgraph induced by all nodes reachable from the
    ///   ﬁxed node in the underlying undirected graph using only tight edges.
    ///   * An edge is "tight" if its slack is zero.
    ///     * The "slack" of an edge is the difference of its length and its minimum length.
    ///     * Thus a edge is "tight" if its length == its minimum length
    ///       * QUESTION: Is "its minimum length" == MIN_EDGE_LENGTH or just the minmum it can be in a tree?
    ///         * If they meant MIN_EDGE_LENGTH, wouldn't they have said "THE minimum edge length"?
    /// * In particular, all such (feasible) trees have the same number of nodes.
    ///
    /// ChatGPT:
    /// * In graph theory, a "maximal tree" is a spanning tree within a given graph that includes the maximum possible
    ///   number of edges while still being a tree.
    ///   * A spanning tree of a graph is a subgraph that is a tree and includes all the vertices of the original graph.
    ///   * A spanning tree is said to be "maximal" if no additional edges can be added to it without creating a cycle.
    ///
    /// Additional papar details:
    /// * A feasible ranking is one satisfying the length constraints l(e) ≥ δ(e) for all e.
    ///   * Thus, a ranking where the all edge rankings are > min_length().  Thus no rank < 1
    ///   * l(e) = length(e) = rank(e1)-rank(e2) = rank_diff(e)
    ///     * length l(e) of e = (v,w) is deﬁned as λ(w) − λ(v)
    ///     * λ(w) − λ(v) = rank(w) - rank(v)
    ///   * δ(e) = min_length(e) = 1 unless requested by user
    /// * Given any ranking, not necessarily feasible, the "slack" of an edge is the difference of its length and its
    ///   minimum length.
    ///   * QUESTION: Is "its minimum length" == MIN_EDGE_LENGTH or just the minmum it can be in a tree?
    ///   * A(0) -> B(1) -> C(2)
    ///      \--------------/
    ///   * vs:
    ///   * A(0) -> B(1) -> C(1)
    ///      \--------------/
    /// * Thus, a ranking is feasible if the slack of every edge is non-negative.
    /// * An edge is "tight" if its slack is zero.
    ///
    /// Return the count of nodes that are in the current feasible tree under consideration.
    ///
    /// tight_tree is used durint the node ranking phase.
    ///
    /// TODO: make this O(1) by keeping track of the count of nodes which are currently "feasible".
    fn tight_tree(&self) -> usize {
        self.nodes
            .iter()
            .filter(|node| node.feasible_tree_member)
            .count()
    }

    /// Return an edge with the smallest slack of any edge which is incident to the tree.
    ///
    /// Incident to the tree means one point of the edge points to a node that is in the tree,
    /// and the other point points to a node that it not within the tree.
    ///
    /// TODO: Make more effecient by keeping a list of incident nodes
    fn get_min_incident_edge(&self) -> Option<usize> {
        let mut candidate = None;
        let mut candidate_slack = i32::MAX;

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.feasible_tree_member {
                for edge_idx in node.get_all_edges() {
                    let connected_node_idx = self
                        .get_connected_node(node_idx, *edge_idx)
                        .expect("Edge not connected");

                    if !self.get_node(connected_node_idx).feasible_tree_member {
                        let slack = self.slack(*edge_idx).expect("Can't calculate slack");

                        if candidate.is_none() || slack < candidate_slack {
                            candidate = Some(*edge_idx);
                            candidate_slack = slack;
                        }
                    }
                }
            }
        }
        candidate
    }

    /// Get an edge that spans a node which is in the feasible tree with another node that is not.
    fn get_next_feasible_edge(&self) -> Option<usize> {
        for node in self.nodes.iter() {
            if node.feasible_tree_member {
                for edge_idx in &node.out_edges {
                    let dst_node = self.get_edge(*edge_idx).dst_node;

                    if !self.get_node(dst_node).feasible_tree_member {
                        return Some(*edge_idx);
                    }
                }
            }
        }
        None
    }

    /// An edge is "feasible" if both it's nodes have been ranked, and rank_diff > MIN_EDGE_LEN.
    fn edge_is_feasible(&self, edge_idx: usize) -> bool {
        if let Some(diff) = self.edge_length(edge_idx) {
            diff > MIN_EDGE_LENGTH as i32
        } else {
            false
        }
    }

    // The slack of an edge is the difference of its length and its minimum length.
    ///
    /// An edge is "tight" if it's slack is zero.
    fn slack(&self, edge_idx: usize) -> Option<i32> {
        self.edge_length(edge_idx)
            .map(|len| len - (MIN_EDGE_LENGTH as i32))
    }

    /// edge_length() is the rank difference between src and dst nodes of the edge.
    fn edge_length(&self, edge_idx: usize) -> Option<i32> {
        self.rank_diff(edge_idx)
    }

    /// rank_diff returns the difference in rank between the source edge and the dst edge.
    ///
    /// Documentation from the paper:
    ///   * l(e) = length(e) = rank(e.dst_node)-rank(e.src_node) = rank_diff(e)
    ///     * length l(e) of e = (v,w) is deﬁned as λ(w) − λ(v)
    ///     * λ(w) − λ(v) = rank(v) - rank(w)
    fn rank_diff(&self, edge_idx: usize) -> Option<i32> {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);

        match (src_node.rank, dst_node.rank) {
            (Some(src), Some(dst)) => Some((dst as i32) - (src as i32)),
            _ => None,
        }
    }

    /// Documentation from the paper:
    /// * Nodes with no unscanned in-edges are placed in a queue.
    ///   * CLARIFICATION: Frist place all nodes with no in-edges in a queue.
    /// * As nodes are taken off the queue, they are assigned the least rank
    ///   that satisfies their in-edges, and their out-edges are marked as scanned.
    /// * In the simplist case, where minLength() == 1 for all edges, this corresponds
    ///   to viewing the graph as a poset (partially ordered set) and assigning the
    ///   the minimal elements to rank 0.  These nodes are removed from the poset and the
    ///   new set of minimal elements are assigned rank 1, etc.
    ///
    //// TODO: Don't we have to remove redundant edges and ensure the graph is
    ///        not circular to befor we even start this?
    ///
    ///   * In my implementation:
    ///     * Initialize:
    ///       * All nodes ranks are set to: None
    ///       * All nodes without in edges are placed on the queue.
    ///       * TODO: What if there a no nodes with no incoming edges to begin with?
    ///     * While the queue is not empty:
    ///       * For each node in the queue:
    ///         * If the rank is None:
    ///           * set rank to the current rank
    ///           * for each outgoing edge on newly ranked node:
    ///             * if the node pointed to by the outgoing edge no longer has any unscanned
    ///               edges, place it on the queue.
    ///
    /// Example:
    ///  a - b - c
    ///   \-----/
    /// Should rank: 0:a 2:b 3:c, because c is only ranked on the third round.
    fn init_rank(&mut self) {
        let mut nodes_to_rank = Vec::new();
        let mut scanned_edges = HashSet::new();

        // Initialize the queue with all nodes with no incoming edges (since no edges
        // are scanned yet)
        for (index, node) in self.nodes.iter_mut().enumerate() {
            node.set_rank(None);
            node.feasible_tree_member = false;

            if node.no_in_edges() {
                nodes_to_rank.push(index);
            }
        }

        let mut cur_rank = 0;
        while !nodes_to_rank.is_empty() {
            let mut next_nodes_to_rank = Vec::new();
            while let Some(node_idx) = nodes_to_rank.pop() {
                let node = self.get_node_mut(node_idx);

                if node.rank.is_none() {
                    node.set_rank(Some(cur_rank));

                    for edge_idx in node.out_edges.clone() {
                        scanned_edges.insert(edge_idx);

                        let edge = self.get_edge(edge_idx);
                        let dst_node = self.get_node(edge.dst_node);
                        if dst_node.no_unscanned_in_edges(&scanned_edges) {
                            next_nodes_to_rank.push(edge.dst_node);
                        }
                    }
                }
            }

            cur_rank += 1;
            next_nodes_to_rank
                .iter()
                .for_each(|idx| nodes_to_rank.push(*idx));
        }
    }

    fn leave_edge(&self) -> Option<usize> {
        todo!()
    }

    fn enter_edge(&self, edge: usize) -> usize {
        todo!()
    }

    fn exchange(&self, e1: usize, e2: usize) -> usize {
        todo!()
    }

    fn normalize(&mut self) {
        todo!();
    }

    fn balance(&mut self) {
        todo!();
    }

    fn ordering(&mut self) {
        todo!();
    }

    fn position(&mut self) {
        todo!();
    }

    fn make_splines(&mut self) {
        todo!();
    }
}

impl Display for Graph {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        for edge in &self.edges {
            let src = &self.nodes[edge.src_node];
            let dst = &self.nodes[edge.dst_node];
            let line = if let Some(val) = edge.cut_value {
                format!(" {:2} ", val)
            } else if edge.feasible_tree_member {
                "----".to_string()
            } else {
                " - -".to_string()
            };

            let _ = writeln!(fmt, "{src} -{line}> {dst}");
        }
        Ok(())
    }
}

// Node of a graph.  Sometimes called a vertice.
#[derive(Debug, Eq, PartialEq)]
pub struct Node {
    // Arbitrary name set by the user.  Duplicates are possible, and up to the user to control.
    name: String,
    // Rank is computed as part of the graphing process.
    rank: Option<u32>,
    // Edges incoming to this node.  Each entry is a edge index into the graph's edges list.
    in_edges: Vec<usize>,
    // Edges outcoming from this node.  Each entry is a edge index into the graph's edges list.
    out_edges: Vec<usize>,

    // True if this node is part of the "feasible" tree under consideration.  Used during ranking.
    feasible_tree_member: bool,
}

// Whether a edge is incoming our outgoing with respect to a particular node.
#[derive(Eq, PartialEq)]
enum EdgeDisposition {
    In,
    Out,
}

impl Node {
    pub fn new(name: &str) -> Self {
        Node {
            name: name.to_string(),
            rank: None,
            in_edges: vec![],
            out_edges: vec![],
            feasible_tree_member: false,
        }
    }

    /// Add either an in our out edge to the node.
    fn add_edge(&mut self, edge: usize, disposition: EdgeDisposition) {
        match disposition {
            EdgeDisposition::In => &self.in_edges.push(edge),
            EdgeDisposition::Out => &self.out_edges.push(edge),
        };
    }

    /// Return the list of In our Out edges.
    fn get_edges(&self, disposition: EdgeDisposition) -> &Vec<usize> {
        match disposition {
            EdgeDisposition::In => &self.in_edges,
            EdgeDisposition::Out => &self.out_edges,
        }
    }

    /// Return all in and out edges associated with a node.
    fn get_all_edges(&self) -> impl Iterator<Item = &usize> {
        self.out_edges.iter().chain(self.in_edges.iter())
    }

    // Return true if none of the incoming edges to node are in the set scanned_edges.
    // * If any incoming edges are not scanned, return false
    // * If there are no incoming edges, return true
    fn no_unscanned_in_edges(&self, scanned_edges: &HashSet<usize>) -> bool {
        for edge_idx in self.get_edges(EdgeDisposition::In) {
            if !scanned_edges.contains(edge_idx) {
                return false;
            }
        }
        true
    }

    /// True if there are no incoming edges to a node.
    fn no_in_edges(&self) -> bool {
        self.get_edges(EdgeDisposition::In).is_empty()
    }

    /// True if there are no outgoing edges to a node.
    fn no_out_edges(&self) -> bool {
        self.get_edges(EdgeDisposition::Out).is_empty()
    }

    /// Sets the rank of a node.
    /// 
    /// Rank corresponds to the vertical placement of a node.  The greater the rank,
    /// the lower the placement on a canvas.
    fn set_rank(&mut self, rank: Option<u32>) {
        self.rank = rank;
    }
}

impl Display for Node {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(fmt, "{}: {:?}", &self.name, self.rank)
    }
}

/// An edge connects to nodes in a graph, and points from src_node to dst_node.
#[derive(Debug, Eq, PartialEq)]
pub struct Edge {
    /// Node that this edge points from.  This is an index into graph.nodes.
    src_node: usize,
    /// Node that this edge points to.  This is an index into graph.nodes.
    dst_node: usize,

    /// The following fields are used for rank calculation:

    /// Weight of the edge,
    weight: u32,
    /// If this edge is ignored internally while calculating node ranks.
    ignored: bool,
    /// If this edge is reversed internally while calculating node ranks.
    reversed: bool,
    /// Used as port of the algorithm to rank the nodes of the graph.
    cut_value: Option<i32>,
    /// True if this edge is part of the feasible tree use to calculate cut values.
    feasible_tree_member: bool,
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
            feasible_tree_member: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ops::Range;

    /// Additional test only functions for Graph.
    impl Graph {
        /// Add multiple nodes with names given from a range of characters.
        ///
        /// Returns a HashMap of the created node names and node indexes to be
        /// used with add_edges().
        ///
        /// * Nodes must be named after a single character.
        /// * The range is inclusive only of the left side.  So 'a'..'d' incluses: a, b, c but NOT d.
        fn add_nodes(&mut self, range: Range<char>) -> HashMap<String, usize> {
            let mut index_map = HashMap::new();

            for name in range {
                index_map.insert(name.to_string(), self.add_node(&name.to_string()));
            }

            index_map
        }

        /// Add a list of edges to the graph, given a map of node names to node indexes.
        ///
        /// To be used with add_nodes().
        fn add_edges(&mut self, edges: &[(&str, &str)], node_map: &HashMap<String, usize>) {
            for (src_node, dst_node) in edges {
                let src_node_idx = *node_map.get(&src_node.to_string()).unwrap();
                let dst_node_idx = *node_map.get(&dst_node.to_string()).unwrap();

                self.add_edge(src_node_idx, dst_node_idx);
            }
        }

        /// Returns a node index given a node name.
        ///
        /// Expensive for large data sets: O(n)
        fn name_to_node_idx(&self, name: &str) -> Option<usize> {
            for (node_idx, node) in self.nodes.iter().enumerate() {
                if name == node.name {
                    return Some(node_idx);
                }
            }
            None
        }

        /// Configures the named node by setting the rank and making the node a feasible tree member.
        ///
        /// Expensive for large data sets: O(n)
        fn configure_node(&mut self, name: &str, rank: u32) {
            let node_idx = self.name_to_node_idx(name).unwrap();
            let node = self.get_node_mut(node_idx);

            node.rank = Some(rank);
            node.feasible_tree_member = true;
        }

        /// Get the edge that has src_node == src_name, dst_node == dst_name.
        ///
        /// Expensive for large data sets: O(e*n)
        fn get_named_edge(&self, src_name: &str, dst_name: &str) -> &Edge {
            for edge in &self.edges {
                let src_node = self.get_node(edge.src_node);
                let dst_node = self.get_node(edge.dst_node);

                if src_node.name == src_name && dst_node.name == dst_name {
                    return edge;
                }
            }
            panic!("Could not find requested edge: {src_name} -> {dst_name}");
        }

        // Set the ranks given in example 2-3 (a)
        fn configure_example_2_3_a() -> (Graph, Vec<(&'static str, &'static str, i32)>) {
            let mut graph = example_graph_from_paper_2_3();
            graph.configure_node("a", 0);
            graph.configure_node("b", 1);
            graph.configure_node("c", 2);
            graph.configure_node("d", 3);
            graph.configure_node("h", 4);

            graph.configure_node("e", 2);
            graph.configure_node("f", 2);
            graph.configure_node("g", 3);

            // Set feasible edges given in example 2-3 (a)
            let e_idx = graph.name_to_node_idx("e").unwrap();
            let f_idx = graph.name_to_node_idx("f").unwrap();
            for edge in graph.edges.iter_mut() {
                if edge.dst_node != e_idx && edge.dst_node != f_idx {
                    edge.feasible_tree_member = true;
                }
            }

            // cutvalues expected in example 2-3 (a)
            (
                graph,
                vec![
                    ("a", "b", 3),
                    ("b", "c", 3),
                    ("c", "d", 3),
                    ("d", "h", 3),
                    ("e", "g", 0),
                    ("f", "g", 0),
                    ("g", "h", -1),
                ],
            )
        }

        // Set the ranks given in example 2-3 (b)
        fn configure_example_2_3_b() -> (Graph, Vec<(&'static str, &'static str, i32)>) {
            let mut graph = example_graph_from_paper_2_3();
            graph.configure_node("a", 0);
            graph.configure_node("b", 1);
            graph.configure_node("c", 2);
            graph.configure_node("d", 3);
            graph.configure_node("h", 4);

            graph.configure_node("e", 1);
            graph.configure_node("f", 1);
            graph.configure_node("g", 2);

            // Set feasible edges given in example 2-3 (b)
            let g_idx = graph.name_to_node_idx("g").unwrap();
            let f_idx = graph.name_to_node_idx("f").unwrap();
            for edge in graph.edges.iter_mut() {
                edge.feasible_tree_member = !(edge.src_node == g_idx || edge.dst_node == f_idx);
            }

            // cutvalues expected in example 2-3 (b)
            (
                graph,
                vec![
                    ("a", "b", 2),
                    ("b", "c", 2),
                    ("c", "d", 2),
                    ("d", "h", 2),
                    ("a", "e", 1),
                    ("e", "g", 1),
                    ("f", "g", 0),
                ],
            )
        }

        fn assert_expected_cutvals(&self, expected_cutvals: Vec<(&str, &str, i32)>) {
            for (src_name, dst_name, cut_val) in expected_cutvals {
                let edge = self.get_named_edge(src_name, dst_name);

                assert_eq!(edge.cut_value, Some(cut_val), "unexpected cut_value");
            }
        }
    }

    #[test]
    fn test_add_edge() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");

        graph.add_edge(a_idx, b_idx);

        {
            let node_a = &graph.nodes[a_idx];
            let node_b = &graph.nodes[b_idx];

            assert_eq!(node_a.in_edges, vec![]);
            assert_eq!(node_a.out_edges, vec![0]);
            assert_eq!(node_b.in_edges, vec![0]);
            assert_eq!(node_b.out_edges, vec![]);
        }

        let c_idx = graph.add_node("C");
        graph.add_edge(a_idx, c_idx);
        graph.add_edge(b_idx, c_idx);

        {
            let node_a = &graph.nodes[a_idx];
            let node_b = &graph.nodes[b_idx];
            let node_c = &graph.nodes[c_idx];

            assert_eq!(node_a.in_edges, vec![]);
            assert_eq!(node_a.out_edges, vec![0, 1]);
            assert_eq!(node_b.in_edges, vec![0]);
            assert_eq!(node_b.out_edges, vec![2]);
            assert_eq!(node_c.in_edges, vec![1, 2]);
            assert_eq!(node_c.out_edges, vec![]);
        }
    }

    #[test]
    fn test_rank() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");
        let d_idx = graph.add_node("D");

        graph.add_edge(a_idx, b_idx);
        graph.add_edge(a_idx, c_idx);
        graph.add_edge(b_idx, d_idx);
        graph.add_edge(c_idx, d_idx);

        graph.rank();

        println!("{graph}");

        assert_eq!(graph.nodes[a_idx].rank, Some(0));
        assert_eq!(graph.nodes[b_idx].rank, Some(1));
        assert_eq!(graph.nodes[c_idx].rank, Some(1));
        assert_eq!(graph.nodes[d_idx].rank, Some(2));
    }

    #[test]
    fn test_rank_scaning() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");

        graph.add_edge(a_idx, b_idx);
        graph.add_edge(a_idx, c_idx);
        graph.add_edge(b_idx, c_idx);

        graph.rank();
        println!("{graph}");

        assert_eq!(graph.nodes[a_idx].rank, Some(0));
        assert_eq!(graph.nodes[b_idx].rank, Some(1));
        assert_eq!(graph.nodes[c_idx].rank, Some(2));
    }

    #[test]
    fn test_cutvalues() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");
        let d_idx = graph.add_node("D");
        let e_idx = graph.add_node("E");
        let f_idx = graph.add_node("F");
        let g_idx = graph.add_node("G");
        let h_idx = graph.add_node("H");

        graph.add_edge(a_idx, b_idx);
        graph.add_edge(a_idx, e_idx);
        graph.add_edge(a_idx, f_idx);

        graph.add_edge(b_idx, c_idx);

        graph.add_edge(e_idx, g_idx);
        graph.add_edge(f_idx, g_idx);
        graph.add_edge(c_idx, d_idx);

        graph.add_edge(g_idx, h_idx);
        graph.add_edge(d_idx, h_idx);

        graph.rank();
        println!("{graph}");
    }

    #[test]
    fn test_merge_edges() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");

        let e1 = graph.add_edge(a_idx, b_idx);
        let e2 = graph.add_edge(a_idx, b_idx);
        let e3 = graph.add_edge(a_idx, b_idx);
        let e4 = graph.add_edge(a_idx, b_idx);

        graph.merge_edges();

        let heavy_edge = graph.get_edge(e1);
        assert_eq!(heavy_edge.weight, MIN_EDGE_WEIGHT * 4);
        assert!(!heavy_edge.ignored);
        for edge in [e2, e3, e4] {
            let ignored_edge = graph.get_edge(edge);

            assert_eq!(ignored_edge.weight, MIN_EDGE_WEIGHT);
            assert!(ignored_edge.ignored);
        }
    }

    #[test]
    // An incident edge is one that has one point in a tree node, and the other
    // in a non-tree node (thus "incident" to the tree).
    fn test_get_min_incident_edge() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");

        let e1 = graph.add_edge(a_idx, b_idx);
        let e2 = graph.add_edge(a_idx, c_idx);

        graph.init_rank();

        println!("{graph}");

        graph.get_node_mut(a_idx).feasible_tree_member = true;
        let min_edge_idx = graph.get_min_incident_edge();
        assert_eq!(min_edge_idx, Some(e1));

        graph.get_node_mut(b_idx).feasible_tree_member = true;
        let min_edge_idx = graph.get_min_incident_edge();
        assert_eq!(min_edge_idx, Some(e2));

        graph.get_node_mut(c_idx).feasible_tree_member = true;
        let min_edge_idx = graph.get_min_incident_edge();
        assert_eq!(min_edge_idx, None);
    }

    #[test]
    fn test_edge_head_is_incident() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");

        let e1 = graph.add_edge(a_idx, b_idx);
        let e2 = graph.add_edge(a_idx, c_idx);
        let e3 = graph.add_edge(b_idx, c_idx);
        let e4 = graph.add_edge(b_idx, a_idx);

        graph.init_rank();

        println!("{graph}");

        graph.get_node_mut(a_idx).feasible_tree_member = true;
        // Graph:(A) <-> B
        //         |    |
        //         |    v
        //          \-> C
        //
        // head is incident only for: B->A
        assert!(!graph.edge_head_is_incident(e1), "A -> B");
        assert!(!graph.edge_head_is_incident(e2), "A -> C");
        assert!(!graph.edge_head_is_incident(e3), "B -> C");
        assert!(graph.edge_head_is_incident(e4), "B -> A");
    }

    /// * l(e) = length(e) = rank(e.dst_node)-rank(e.src_node) = rank_diff(e)
    ///   * length l(e) of e = (v,w) is deﬁned as λ(w) − λ(v)
    ///   * λ(w) − λ(v) = rank(w) - rank(v)
    #[test]
    fn test_set_edge_length() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");

        let a_b = graph.add_edge(a_idx, b_idx);
        let a_c = graph.add_edge(a_idx, c_idx);
        let b_c = graph.add_edge(b_idx, c_idx);
        // let c_a = graph.add_edge(c_idx, a_idx);

        // A: rank(0)
        // B: rank(1)
        // C: rank(2)

        assert_eq!(graph.edge_length(a_b), None);

        graph.init_rank();
        println!("{graph}");

        assert_eq!(graph.edge_length(a_b), Some(1));
        assert_eq!(graph.edge_length(a_c), Some(2));
        assert_eq!(graph.edge_length(b_c), Some(1));
        // assert_eq!(graph.edge_length(c_a), Some(-2));
    }

    #[test]
    fn test_set_edge_slack() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");

        let a_b = graph.add_edge(a_idx, b_idx);
        let a_c = graph.add_edge(a_idx, c_idx);
        let b_c = graph.add_edge(b_idx, c_idx);
        // let c_a = graph.add_edge(c_idx, a_idx);

        // A: rank(0)
        // B: rank(1)
        // C: rank(2)

        assert_eq!(graph.slack(a_b), None);

        graph.init_rank();
        println!("{graph}");

        assert_eq!(graph.slack(a_b), Some(0));
        assert_eq!(graph.slack(a_c), Some(1));
        assert_eq!(graph.slack(b_c), Some(0));
        // assert_eq!(graph.edge_length(c_a), Some(-2));
    }

    fn example_graph_from_paper_2_3() -> Graph {
        let mut graph = Graph::new();
        let node_map = graph.add_nodes('a'..'i');
        let edges = vec![
            ("a", "b"),
            ("b", "c"),
            ("c", "d"),
            ("d", "h"),
            ("a", "f"),
            ("f", "g"),
            ("g", "h"),
            ("a", "e"),
            ("e", "g"),
        ];
        graph.add_edges(&edges, &node_map);

        graph
    }

    #[test]
    fn test_init_cut_values_2_3_a() {
        let (mut graph, expected_cutvals) = Graph::configure_example_2_3_a();

        graph.init_cutvalues();
        println!("{graph}");
        graph.assert_expected_cutvals(expected_cutvals);
    }

    #[test]
    fn setup_init_cut_values_2_3_b() {
        let (mut graph, expected_cutvals) = Graph::configure_example_2_3_b();

        graph.init_cutvalues();
        println!("{graph}");
        graph.assert_expected_cutvals(expected_cutvals);
    }

    #[test]
    fn test_set_feasible_tree() {
        let mut graph = example_graph_from_paper_2_3();
        graph.set_feasible_tree();

        println!("{graph}");
    }
}
