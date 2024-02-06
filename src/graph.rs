//! Implement a graph that can be drawn using the algorithm described in the 1993 paper:
//! "A Technique for Drawing Directed Graphs" by Gansner, Koutsofios, North and Vo
//!
//! This paper is referred to as simply "the paper" below.

mod crossing_lines;
mod edge;
mod node;
mod rank_orderings;

use rank_orderings::RankOrderings;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    mem::replace,
};

use self::{
    edge::{Edge, EdgeDisposition, MIN_EDGE_LENGTH},
    node::Node,
    rank_orderings::AdjacentRank,
};

/// Determines what variable on each node which is set by the network simplex algorithm.
#[derive(Debug, Clone, Copy)]
enum SimplexNodeTarget {
    /// The rank of each node, which corresponds with to the ultimate y position of the node
    Rank,
    /// The X coordinate of each node
    XCoordinate,
}

/// Simplist posible representation of a graph until more is needed.
///
/// I chose to use indexed arrays to avoid interior mutability for now,
/// as well as requiring any maps or sets, because initially it was unclear to me what
/// would be optimal.  Both could be addressed on a future refactor.
///
/// TODO:
/// * Does not handle disconnected nodes, and does not enforce that all nodes must
///   be connected.  Thus, if you add nodes that are not connected, only the connected
///   nodes will be graphed in graph_node.
/// * Individual edge node loops are set to "ignore", and edges are merged to add weight,
/// * with the other edge set to ignore, but the graph_node() does not take into account
///   ignored nodes.
/// * Add an error type and remove all unwrap(), expect() and panic() code.
/// * Make runtime effecient
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

    /// The graph function described in "A Technique for Drawing Directed Graphs"
    pub fn draw_graph(&mut self) {
        self.rank();
        self.ordering();
        // self.position();
        // self.make_splines()
    }

    /// Return the node indexed by node_idx.
    pub fn get_node(&self, node_idx: usize) -> &Node {
        &self.nodes[node_idx]
    }

    /// Return a mutable node indexed by node_idx.
    pub fn get_node_mut(&mut self, node_idx: usize) -> &mut Node {
        &mut self.nodes[node_idx]
    }

    /// Return the edge indexed by edge_idx.
    pub fn get_edge(&self, edge_idx: usize) -> &Edge {
        &self.edges[edge_idx]
    }

    /// Return mutable edge indexed by edge_idx.
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

    fn get_rank_adjacent_edges(
        &self,
        node_idx: usize,
    ) -> impl Iterator<Item = (usize, AdjacentRank)> + '_ {
        let node = self.get_node(node_idx);
        let node_rank = node.simplex_rank;

        node.get_all_edges().cloned().filter_map(move |edge_idx| {
            if let Some(other_node_idx) = self.get_connected_node(node_idx, edge_idx) {
                let other_node = self.get_node(other_node_idx);

                if let (Some(n1), Some(n2)) = (node_rank, other_node.simplex_rank) {
                    let diff = n1 as i64 - n2 as i64;

                    if diff == -1 {
                        Some((edge_idx, AdjacentRank::Below))
                    } else if diff == 1 {
                        Some((edge_idx, AdjacentRank::Above))
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            }
        })
    }

    fn get_rank_adjacent_nodes(&self, node_idx: usize) -> (Vec<usize>, Vec<usize>) {
        let mut above_nodes = vec![];
        let mut below_nodes = vec![];

        for (edge_idx, direction) in self.get_rank_adjacent_edges(node_idx) {
            let edge = self.get_edge(edge_idx);
            let adj_node_idx = if edge.src_node == node_idx {
                edge.dst_node
            } else {
                edge.src_node
            };

            match direction {
                AdjacentRank::Above => above_nodes.push(adj_node_idx),
                AdjacentRank::Below => below_nodes.push(adj_node_idx),
            }
        }
        (above_nodes, below_nodes)
    }

    /// Add a new node identified by name, and return the node's index in the graph.
    pub fn add_node(&mut self, name: &str) -> usize {
        let new_node = Node::new(name);
        let idx = self.nodes.len();
        self.nodes.push(new_node);

        idx
    }

    /// Add a new node (marked as virtual=true)
    pub fn add_virtual_node(&mut self) -> usize {
        let idx = self.add_node("V");
        self.get_node_mut(idx).virtual_node = true;

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

    /// Rank nodes in the tree using the network simplex algorithm.
    pub fn rank(&mut self) {
        self.merge_edges();
        self.make_asyclic();
        self.network_simplex_ranking(SimplexNodeTarget::Rank);
    }

    /// Rank nodes in the graph using the network simplex algorithm.
    ///
    /// Note that this general algorithm is used for more than one purpose for layout:
    /// * Placing the nodes into horizontal ranks
    /// * Determining the X axis of nodes (when additional auxiliary nodes and edges are added)
    ///
    /// Network Simplex Algorithm: From Wikipedia, the free encyclopedia:
    ///
    ///   In mathematical optimization, the network simplex algorithm is a graph theoretic
    ///   specialization of the simplex algorithm. The algorithm is usually formulated in
    ///   terms of a minimum-cost flow problem. The network simplex method works very well
    ///   in practice, typically 200 to 300 times faster than the simplex method applied
    ///   to general linear program of same dimensions.
    ///     
    ///   The basis (of network simplex) is represented as a rooted spanning tree of the
    ///   underlying network, in which variables are represented by arcs (e.g. edges), and the
    ///   simplex multipliers by node potentials (e.g. any particular node value). At each
    ///   iteration, an entering variable is selected by some pricing strategy, based on the
    ///   dual multipliers (node potentials), and forms a cycle with the arcs of the tree.
    ///   The leaving variable is the arc of the cycle with the least augmenting flow.
    ///   The substitution of entering for leaving arc, and the reconstruction of the tree
    ///   is called a pivot. When no non-basic arc remains eligible to enter, the optimal
    ///   solution has been reached.
    ///     
    ///   So for this implementation:
    ///     * "arcs" are edges
    ///     * "simplex multipliers" or "node potentials" are "cut values"
    ///
    /// Documentation from the paper: pages 8-9
    /// * described in [TSE93]
    ///
    /// * Figure below describes our version of the network simplex algorithm:
    ///   * leave_edge returns a tree edge with a negative cut value, or nil if there is none,
    ///     meaning the solution is optimal.  Any edge with a negative cut value may be selected as the
    ///     edge to remove.
    ///   * enter_edge ﬁnds a non-tree edge to replace e.  This is done by breaking the edge e, which
    ///     divides the tree into a head and tail component. All edges going from the head component to
    ///     the tail are considered, with an edge of minimum slack being chosen.  This is necessary to
    ///     maintain feasibility.
    /// fn network_simplex_rank() {
    ///     feasible_tree();
    ///     while (e = leave_edge()) ≠ nil {
    ///         f = enter_edge(e);
    ///         exchange(e,f);
    ///     }
    ///     normalize();
    ///     balance();
    /// }
    ///
    fn network_simplex_ranking(&mut self, target: SimplexNodeTarget) {
        self.set_feasible_tree();
        while let Some(neg_cut_edge_idx) = self.leave_edge() {
            let non_tree_edge_idx = self
                .enter_edge(neg_cut_edge_idx)
                .expect("No negative cut values found!");
            self.exchange(neg_cut_edge_idx, non_tree_edge_idx);
        }
        self.normalize();
        // self.balance();
        self.assign_simplex_rank(target);
    }

    /// After running the network simplex algorithm, assign the result to each node.
    fn assign_simplex_rank(&mut self, target: SimplexNodeTarget) {
        for node in self.nodes.iter_mut() {
            node.assign_simplex_rank(target);
        }
    }

    /// make_asyclic() removes cycles from the graph.
    /// * starting with "source nodes" (nodes with only outgoing edges) it does a depth first search (DFS).
    ///   * "visited" nodes have their "tree_member" attribute set
    ///   * when the DFS finds an edge pointing to a visited node, it reverses the direction of the edge,
    ///     and sets the "reversed" attribute of the edge (so it can be depicted as it was originally)
    ///     * Trying to reverse an already reversed edge is an error
    /// * Before finishing, all nodes are checked to see if they have been visited.
    ///   * If one is found, start a new DFS using this node.
    ///   * Repeat until all nodes have been visited
    ///
    /// Documentation from the paper: page 6: 2.1: Making the graph asyclic
    /// * A graph must be acyclic to have a consistent rank assignment.
    /// * Because the input graph may contain cycles, a preprocessing step detects cycles and
    ///   breaks them by reversing certain edges [RDM].
    ///   * Of course these edges are only reversed internally; arrowheads in the drawing show
    ///     the original direction.
    /// * A useful procedure for breaking cycles is based on depth-ﬁrst search.
    ///   * Edges are searched in the "natural order" of the graph input, starting from some
    ///     source or sink nodes if any exist.
    ///     - a source node is a node in a directed graph that has no incoming edges.
    ///     - a sink node is a node in a directed graph that has no outgoing edges.
    ///   * Depth-ﬁrst search partitions edges into two sets: tree edges and non-tree edges [AHU].
    ///     * The tree deﬁnes a partial order on nodes.
    ///     * Given this partial order, the non-tree edges further partition into three sets:
    ///       cross edges, forward edges, and back edges.
    ///       * Cross edges connect unrelated nodes in the partial order.
    ///       * Forward edges connect a node to some of its descendants.
    ///       * Back edges connect a descendant to some of its ancestors.
    ///    * It is clear that adding forward and cross edges to the partial order does not create cycles.
    ///    * Because reversing back edges makes them into forward edges, all cycles are broken by this procedure.
    fn make_asyclic(&mut self) {
        self.ignore_node_loops();

        let mut queue = self.get_source_nodes();
        self.set_asyclic_tree(&mut queue);

        let mut start = 0;
        while let Some(non_tree_node_idx) = self.get_next_non_tree_node_idx(start) {
            queue.push_front(non_tree_node_idx);
            self.set_asyclic_tree(&mut queue);

            start = non_tree_node_idx + 1;
        }
    }

    /// Ignore individual edges that loop to and from the same node.
    fn ignore_node_loops(&mut self) {
        self.edges
            .iter_mut()
            .filter(|edge| edge.src_node == edge.dst_node)
            .for_each(|edge| edge.ignored = true);
    }

    /// Beginning with start, return the first index that is not yet marked as part of the tree.
    fn get_next_non_tree_node_idx(&self, start: usize) -> Option<usize> {
        for (index, node) in self.nodes.iter().skip(start).enumerate() {
            let node_idx = start + index;
            if !node.tree_node {
                return Some(node_idx);
            }
        }
        None
    }

    /// Return a queue of nodes that don't have incoming edges (source nodes).
    fn get_source_nodes(&self) -> VecDeque<usize> {
        let mut queue = VecDeque::new();

        for (node_idx, node) in self
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| n.no_in_edges())
        {
            queue.push_back(node_idx);
        }
        queue
    }

    /// Given a queue of source nodes, mark a tree of asyclic nodes.
    /// * Do a depth first search starting fron the source nodes
    /// * Mark any nodes visited as "tree_node"
    /// * If any edges point to a previously visted node, reverse those edges.
    fn set_asyclic_tree(&mut self, queue: &mut VecDeque<usize>) {
        while let Some(node_idx) = queue.pop_front() {
            let node = self.get_node_mut(node_idx);
            node.tree_node = true;

            let node = self.get_node(node_idx);
            let mut edges_to_reverse = Vec::new();
            for edge_idx in node.out_edges.iter().cloned() {
                let edge = self.get_edge(edge_idx);
                let dst_node = self.get_node(edge.dst_node);

                if !dst_node.tree_node {
                    queue.push_back(edge.dst_node);
                } else {
                    edges_to_reverse.push(edge_idx);
                }
            }
            for edge_idx in edges_to_reverse {
                self.reverse_edge(edge_idx);
            }
        }
    }

    fn reverse_edge(&mut self, edge_idx_to_reverse: usize) {
        let (src_node_idx, dst_node_idx) = {
            let edge = self.get_edge(edge_idx_to_reverse);

            (edge.src_node, edge.dst_node)
        };

        // Swap the references in src and dst nodes
        self.get_node_mut(src_node_idx)
            .swap_edge_in_list(edge_idx_to_reverse, EdgeDisposition::Out);
        self.get_node_mut(dst_node_idx)
            .swap_edge_in_list(edge_idx_to_reverse, EdgeDisposition::In);

        // Reverse the edge itself and set it to "reversed"
        let edge = self.get_edge_mut(edge_idx_to_reverse);
        edge.src_node = replace(&mut edge.dst_node, edge.src_node);
        edge.reversed = true;
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
    /// Documentation from the paper: pages 8-9
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
    /// Additional papar details: page 7
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
    fn set_feasible_tree(&mut self) {
        self.init_rank();

        for node in self.nodes.iter_mut() {
            node.tree_node = node.no_out_edges();
        }

        while self.tight_tree() < self.node_count() {
            // e = a non-tree edge incident on the tree with a minimal amount of slack
            // delta = slack(e);
            // if includent_node is e.head then delta = -delta
            // for v in Tree do v.rank = v.rank + delta;
            let edge_idx = self
                .get_min_incident_edge()
                .expect("No incident edges left!");
            let mut delta = if let Some(delta) = self.simplex_slack(edge_idx) {
                if self.edge_head_is_incident(edge_idx) {
                    -delta
                } else {
                    delta
                }
            } else {
                panic!("Can't calculate slack on edge {edge_idx}");
            };

            for node in self.nodes.iter_mut().filter(|node| node.tree_node) {
                let cur_rank = node.simplex_rank.expect("Node does not have rank");
                node.simplex_rank = Some(cur_rank + delta as u32)
            }

            let node_idx = self
                .get_incident_node(edge_idx)
                .expect("Edge is not incident");
            self.get_node_mut(node_idx).tree_node = true;
            self.get_edge_mut(edge_idx).feasible_tree_member = true;
        }
        self.init_cutvalues();
    }

    /// Calculate the cutvalues of all edges that are part of the current feasible tree.
    ///
    /// Documentation from the paper:
    /// * The init_cutvalues function computes the cut values of the tree edges.
    ///   * For each tree edge, this is computed by marking the nodes as belonging to the head or tail component,
    ///   * and then performing the sum of the signed weights of all edges whose head and tail are in different components,
    ///     * the sign being negative for those edges going from the head to the tail component
    ///
    /// Optimization TODOs from the paper:
    /// * In a naive implementation, initial cut values can be found by taking every tree edge in turn,
    ///   breaking it, labeling each node according to whether it belongs to the head or tail component,
    ///   and performing the sum.
    ///   * This takes O(V E) time.
    ///   * To reduce this cost, we note that the cut values can be computed using information local to an edge
    ///     if the search is ordered from the leaves of the feasible tree inward.
    ///     * It is trivial to compute the cut value of a tree edge with one of its endpoints a leaf in the tree,
    ///       since either the head or the tail component consists of a single node.
    ///     * Now, assuming the cut values are known for all the edges incident on a given node except one, the
    ///       cut value of the remaining edge is the sum of the known cut values plus a term dependent only on
    ///       the edges incident to the given node.
    ///
    /// * Another valuable optimization, similar to a technique described in [Ch], is to perform a postorder traversal
    ///   of the tree, starting from some ﬁxed root node v root, and labeling each node v with its postorder traversal
    ///   number lim(v), the least number low(v) of any descendant in the search, and the edge parent(v) by which the
    ///   node was reached (see ﬁgure 2-5).
    ///   * This provides an inexpensive way to test whether a node lies in the head or tail component of a tree edge,
    ///     and thus whether a non-tree edge crosses between the two components.
    fn init_cutvalues(&mut self) {
        for edge_idx in 0..self.edges.len() {
            let edge = self.get_edge(edge_idx);
            if edge.feasible_tree_member {
                let (head_nodes, tail_nodes) = self.get_components(edge_idx);
                let cut_value = self.transition_weight_sum(&head_nodes, &tail_nodes);

                self.get_edge_mut(edge_idx).cut_value = Some(cut_value);
                // println!(
                //     "Set cut value for {}: {cut_value}\n  heads: {}\n  tails: {}",
                //     self.display_edge(edge_idx),
                //     self.display_component(&head_nodes),
                //     self.display_component(&tail_nodes),
                // )
            }
        }
    }

    /// Get the head an tail components needed to calculate edge cut values.
    ///
    /// Documentation from the paper: page 8
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

    #[cfg(test)]
    fn display_component(&self, comp: &HashSet<usize>) -> String {
        let mut node_names = vec![];
        for node_idx in comp {
            node_names.push(self.get_node(*node_idx).name.clone());
        }
        node_names.sort();

        format!("{node_names:?}",)
    }

    /// Collect a head or tail cutset component.
    ///
    /// * Given an initial node, consider all edges which are in the feasible tree
    ///   * ignore the cut edge
    ///   * if the edge is not yet in this cut set or the opposite cut set, add it.
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

                component_set.insert(node_idx);

                for edge_idx in node.get_all_edges().filter(|edge_idx| {
                    let edge_idx = **edge_idx;

                    edge_idx != cut_edge_idx && self.get_edge(edge_idx).feasible_tree_member
                }) {
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

        !src_node.tree_node && dst_node.tree_node
    }

    /// edge_index is expected to span two nodes, one of which is in the tree, one of which is not.
    /// Return the index to the node which is not yet in the tree.
    fn get_incident_node(&self, edge_idx: usize) -> Option<usize> {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);

        if !src_node.tree_node && dst_node.tree_node {
            Some(edge.src_node)
        } else if src_node.tree_node && !dst_node.tree_node {
            Some(edge.dst_node)
        } else {
            None
        }
    }

    /// Return the count of nodes that are in the current feasible tree under consideration.
    ///
    /// tight_tree is used during the node ranking phase.
    ///
    /// TODO: make this O(1) by keeping track of the count of nodes which are currently "feasible".
    ///
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
    fn tight_tree(&self) -> usize {
        self.nodes.iter().filter(|node| node.tree_node).count()
    }

    /// Return an edge with the smallest slack of any edge which is incident to the tree.
    ///
    /// Incident to the tree means one point of the edge points to a node that is in the tree,
    /// and the other point points to a node that it not within the tree.
    ///
    /// TODO: Make more effecient by keeping a list of incident nodes
    ///
    /// Optimization TODO from the paper:
    /// * The network simplex is also very sensitive to the choice of the negative edge to replace.
    /// * We observed that searching cyclically through all the tree edges, instead of searching from the
    ///   beginning of the list of tree edges every time, can save many iterations.
    fn get_min_incident_edge(&self) -> Option<usize> {
        let mut candidate = None;
        let mut candidate_slack = i32::MAX;

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.tree_node {
                for edge_idx in node.get_all_edges() {
                    let connected_node_idx = self
                        .get_connected_node(node_idx, *edge_idx)
                        .expect("Edge not connected");

                    if !self.get_node(connected_node_idx).tree_node {
                        let slack = self
                            .simplex_slack(*edge_idx)
                            .expect("Can't calculate slack");

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
            if node.tree_node {
                for edge_idx in &node.out_edges {
                    let dst_node = self.get_edge(*edge_idx).dst_node;

                    if !self.get_node(dst_node).tree_node {
                        return Some(*edge_idx);
                    }
                }
            }
        }
        None
    }

    /// An edge is "feasible" if both it's nodes have been ranked, and rank_diff > MIN_EDGE_LEN.
    fn edge_is_feasible(&self, edge_idx: usize) -> bool {
        if let Some(diff) = self.simplex_edge_length(edge_idx) {
            diff > MIN_EDGE_LENGTH as i32
        } else {
            false
        }
    }

    /// Returns the slack of and edge for the network simplex algorithm.
    ///
    /// The slack of an edge is the difference of its length and its minimum length.
    ///
    /// An edge is "tight" if it's slack is zero.
    fn simplex_slack(&self, edge_idx: usize) -> Option<i32> {
        self.simplex_edge_length(edge_idx).map(|len| {
            if len > 0 {
                len - (MIN_EDGE_LENGTH as i32)
            } else {
                len + (MIN_EDGE_LENGTH as i32)
            }
        })
    }

    /// Returns the simplex length for an edge in the network simplex algorithm.
    ///
    /// simplex_edge_length() is the simplex rank difference between src and dst nodes of the edge.
    fn simplex_edge_length(&self, edge_idx: usize) -> Option<i32> {
        self.simplex_rank_diff(edge_idx)
    }

    /// simplex_rank_diff() returns the difference in rank between the source edge and the dst edge.
    ///
    /// Documentation from the paper:
    ///   * l(e) = length(e) = rank(e.dst_node)-rank(e.src_node) = rank_diff(e)
    ///     * length l(e) of e = (v,w) is deﬁned as λ(w) − λ(v)
    ///     * λ(w) − λ(v) = rank(v) - rank(w)
    fn simplex_rank_diff(&self, edge_idx: usize) -> Option<i32> {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);

        match (src_node.simplex_rank, dst_node.simplex_rank) {
            (Some(src), Some(dst)) => Some((dst as i32) - (src as i32)),
            _ => None,
        }
    }

    /// Documentation from the paper:
    /// * Nodes with no unscanned in-edges are placed in a queue.
    ///   * CLARIFICATION: First place all nodes with no in-edges in a queue.
    /// * As nodes are taken off the queue, they are assigned the least rank
    ///   that satisfies their in-edges, and their out-edges are marked as scanned.
    /// * In the simplist case, where minLength() == 1 for all edges, this corresponds
    ///   to viewing the graph as a poset (partially ordered set) and assigning the
    ///   the minimal elements to rank 0.  These nodes are removed from the poset and the
    ///   new set of minimal elements are assigned rank 1, etc.
    ///
    //// TODO: Don't we have to remove redundant edges and ensure the graph is not circular
    ///  before we even start this?
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
            node.set_simplex_rank(None);
            node.tree_node = false;

            if node.no_in_edges() {
                nodes_to_rank.push(index);
            }
        }

        let mut cur_rank = 0;
        while !nodes_to_rank.is_empty() {
            let mut next_nodes_to_rank = Vec::new();
            while let Some(node_idx) = nodes_to_rank.pop() {
                let node = self.get_node_mut(node_idx);

                if node.simplex_rank.is_none() {
                    node.set_simplex_rank(Some(cur_rank));

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

    /// If any edge has a negative cut value, return the first one found.
    ///
    /// Otherwise, return None.
    fn leave_edge(&self) -> Option<usize> {
        for (edge_idx, edge) in self.edges.iter().enumerate() {
            if let Some(cut_value) = edge.cut_value {
                if cut_value < 0 {
                    return Some(edge_idx);
                }
            }
        }
        None
    }

    /// Given an edge with a negative cut value, return a non-tree edge to replace it.
    ///
    /// Documentation from paper:
    /// * enter_edge ﬁnds a non-tree edge to replace e.
    ///   * This is done by breaking the edge e, which divides the tree into a head and tail component.
    ///   * All edges going from the head component to the tail are considered, with an edge of minimum slack being chosen.
    ///   * This is necessary to maintain feasibility.
    fn enter_edge(&self, neg_cut_edge_idx: usize) -> Option<usize> {
        let replacement_edge_idx = 0;
        let (head_nodes, tail_nodes) = self.get_components(neg_cut_edge_idx);

        let mut min_slack = i32::MAX;
        let mut replacement_edge_idx = None;

        for (edge_idx, edge) in self.edges.iter().enumerate() {
            if head_nodes.contains(&edge.src_node) && tail_nodes.contains(&edge.dst_node) {
                let edge_slack = self.simplex_slack(edge_idx).expect("Can't calculate slack");

                if edge_slack < min_slack {
                    replacement_edge_idx = Some(edge_idx)
                }
            }
        }

        replacement_edge_idx
    }

    /// Documentation from paper:
    /// * The edges are exchanged, updating the tree and cut values.
    fn exchange(&mut self, neg_cut_edge_idx: usize, non_tree_edge_idx: usize) {
        {
            let neg_cut_edge = self.get_edge_mut(neg_cut_edge_idx);
            neg_cut_edge.feasible_tree_member = false;

            let non_tree_edge = self.get_edge_mut(non_tree_edge_idx);
            non_tree_edge.feasible_tree_member = true;
        }

        self.init_cutvalues();
    }

    /// Set the least rank of the tree to zero.
    /// * finding the current least rank
    /// * subtracking the least rank from all ranks
    ///
    /// Documentation from paper:
    /// The solution is normalized setting the least rank to zero.
    fn normalize(&mut self) {
        if let Some(min_node) = self.nodes.iter().min() {
            if let Some(least_rank) = min_node.simplex_rank {
                for node in self.nodes.iter_mut() {
                    if let Some(rank) = node.simplex_rank {
                        node.simplex_rank = Some(rank - least_rank);
                    }
                }
            }
        }
    }

    /// TODO:
    ///
    /// Documentation from paper: page 9
    /// * Nodes having equal in- and out-edge weights and multiple feasible ranks are moved to a feasible rank
    ///   with the fewest nodes.
    ///   * The purpose is to reduce crowding and improve the aspect ratio of the drawing,
    ///     following principle A4.
    ///   * The adjustment does not change the cost of the rank assignment.
    ///   * Nodes are adjusted in a greedy fashion, which works sufﬁciently well.
    ///   * Globally balancing ranks is considered in a forthcoming paper [GNV2]:
    ///     "On the Rank Assignment Problem"
    ///     * Unclear if this paper was ever created or submitted
    fn balance(&mut self) {
        todo!();
    }

    /// Documentation from paper: page 14
    ///
    /// * TODO: In an actual implementation, one might prefer an adaptive strategy that
    ///   iterates as long as the solution has improved at least a few percent
    ///   over the last several iterations.
    ///   * Also if no position changes, we can break the loop because it will never improve.
    /// ordering {
    ///     order = init_order();
    ///     best = order;
    ///     for i=0 to max_iterations {
    ///         wmedian(order, i)
    ///         transpose(order)
    ///         if crossing(order) < crossing(best) {
    ///             best = order;
    ///         }
    ///     }
    ///     return best
    /// }
    fn ordering(&mut self) -> RankOrderings {
        const MAX_ITERATIONS: usize = 24;
        let order = self.init_order();
        let mut best = order.clone();

        for i in 0..MAX_ITERATIONS {
            // println!("Ordering pass {i}: cross count: {}", order.crossing_count());

            order.weighted_median(i);
            // println!(
            //     "  After weighed_median: {}\n{order}",
            //     order.crossing_count()
            // );

            order.transpose();
            // println!("  After transpose: {}\n{order}", order.crossing_count());

            let new_crossing_count = order.crossing_count();
            if new_crossing_count < best.crossing_count() {
                best = order.clone();

                if new_crossing_count == 0 {
                    break;
                }
            }

            self.set_node_positions(&best)
        }
        // println!(
        //     "-- Final order (crosses: {}): --\n{best}",
        //     best.crossing_count()
        // );

        best
    }

    /// Set the relative positions of each node from the given orderings.
    fn set_node_positions(&mut self, orderings: &RankOrderings) {
        let node_positions = orderings.nodes().borrow();
        for (node_idx, node) in self.nodes.iter_mut().enumerate() {
            if let Some(node_pos) = node_positions.get(&node_idx) {
                node.horizontal_position = Some(node_pos.borrow().position());
            } else {
                panic!("Node {node_idx} not found in rank orderings");
            }
        }
    }

    /// Set the initial ordering of the nodes, and return a RankOrderings object to optimize node orderings.
    fn init_order(&mut self) -> RankOrderings {
        let mut order = self.get_initial_ordering();

        self.fill_rank_gaps(&order);
        self.set_adjacent_nodes_in_ranks(&order);

        order
    }

    /// Edges between nodes more than one rank apart are replaced by chains of virtual nodes.
    ///
    /// After runing, no edge spans more than one rank.
    ///
    /// Documentation from paper: page 13
    /// * After rank assignment, edges between nodes more than one rank apart are
    ///   replaced by chains of unit length edges between temporary or "virtual" nodes.
    /// * The virtual nodes are placed on the intermediate ranks, converting the original
    ///   graph into one whose edges connect only nodes on adjacent ranks.
    /// * Self- edges are ignored in this pass, and multi-edges are merged as in the previous pass
    fn fill_rank_gaps(&mut self, order: &RankOrderings) {
        for (rank, rank_order) in order.iter() {
            for node_idx in rank_order.borrow().iter() {
                let node_edges = self
                    .get_node(*node_idx)
                    .get_all_edges()
                    .cloned()
                    .collect::<Vec<usize>>();

                for edge_idx in node_edges {
                    if let Some(slack) = self.simplex_slack(edge_idx) {
                        if slack != 0 {
                            self.replace_edge_with_virtual_chain(edge_idx, *rank, slack, order);
                        }
                    }
                }
            }
        }
    }

    fn replace_edge_with_virtual_chain(
        &mut self,
        edge_idx: usize,
        rank: u32,
        slack: i32,
        order: &RankOrderings,
    ) {
        let mut remaining_slack = slack;
        let reverse_edge = slack < 0;

        let mut cur_edge_idx = edge_idx;
        let mut new_rank = rank;
        while remaining_slack != 0 {
            new_rank = if reverse_edge {
                new_rank - 1
            } else {
                new_rank + 1
            };

            let virt_node_idx = self.add_virtual_node();
            self.get_node_mut(virt_node_idx).simplex_rank = Some(new_rank);

            let old_edge = self.get_edge_mut(cur_edge_idx);
            let orig_dst = replace(&mut old_edge.dst_node, virt_node_idx);

            cur_edge_idx = self.add_edge(virt_node_idx, orig_dst);
            order.add_node_idx_to_existing_rank(new_rank, virt_node_idx);

            remaining_slack += if reverse_edge { 1 } else { -1 };
        }
    }

    /// Return an initial ordering of the graph ranks.
    ///
    /// * The initial ordering is a map of ranks.
    /// * Each rank is a set of NodePositions.
    ///   * The position of each node in the rank is in the NodePosition, as well
    ///     as the node_idx.
    ///
    /// * Start with the nodes in the minimal rank (presumably rank 0)
    ///  * Do a depth first seach by following edges that point to nodes that
    ///    have not yet been assigned an ordering
    ///    * When we find a node that has no edges that have not been assigned
    ///      * Add it to the rank_order BTreeMap under it's given rank
    ///      * Mark it assigned
    ///    * When we find a node that has edges that have not yet been assigned
    ///      * push the found node back onto the front of the queue.
    ///      * push the all the unassinged nodes the node's edges point to on the front of the queue
    /// * Continue until the queue in empty and return the rank order.
    ///
    /// Documentation from paper: page 14
    /// init_order initially orders the nodes in each rank.
    /// * This may be done by a depth-ﬁrst or breadth-ﬁrst search starting with vertices of minimum rank.
    ///   * Vertices are assigned positions in their ranks in left-to-right order as the search progresses.
    ///     * This strategy ensures that the initial ordering of a tree has no crossings.
    ///     * This is important because such crossings are obvious, easily- avoided "mistakes."
    fn get_initial_ordering(&mut self) -> RankOrderings {
        let mut rank_order = RankOrderings::new();
        let mut dfs_queue = self.get_min_rank_nodes();
        let mut assigned = HashSet::new();

        while let Some(node_idx) = dfs_queue.pop_front() {
            let node = self.get_node(node_idx);
            let unassigned_dst_nodes = node
                .out_edges
                .iter()
                .cloned()
                .filter_map(|edge_idx| {
                    let edge = self.get_edge(edge_idx);

                    if assigned.get(&edge.dst_node).is_none() {
                        Some(edge.dst_node)
                    } else {
                        None
                    }
                })
                .collect::<Vec<usize>>();

            if unassigned_dst_nodes.is_empty() {
                if let Some(rank) = node.simplex_rank {
                    assigned.insert(node_idx);
                    rank_order.add_node_idx_to_rank(rank, node_idx);
                }
            } else {
                dfs_queue.push_front(node_idx);
                for node_idx in unassigned_dst_nodes {
                    dfs_queue.push_front(node_idx);
                }
            }
        }
        rank_order
    }

    /// The graph is reponsible for setting adjacent nodes in the rank_order once all nodes have been added to it.
    fn set_adjacent_nodes_in_ranks(&self, rank_order: &RankOrderings) {
        for (node_idx, node_position) in rank_order.nodes().borrow().iter() {
            let (above_adj, below_adj) = self.get_rank_adjacent_nodes(*node_idx);

            rank_order.set_adjacent_nodes(*node_idx, &above_adj, &below_adj);
        }
    }

    /// Return a VecDequeue of nodes which have minimum rank.
    ///
    /// * Assumes that the graph has been ranked
    fn get_min_rank_nodes(&self) -> VecDeque<usize> {
        let mut min_rank_nodes = VecDeque::new();
        let min_rank = self
            .nodes
            .iter()
            .min()
            .and_then(|min_node| min_node.simplex_rank);

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.simplex_rank == min_rank {
                min_rank_nodes.push_back(node_idx);
            }
        }
        min_rank_nodes
    }

    /// Return a hash map of rank -> vec<node_idx> as well as the minimum rank
    fn get_rank_map(&self) -> (Option<u32>, HashMap<u32, Vec<usize>>) {
        let mut ranks: HashMap<u32, Vec<usize>> = HashMap::new();
        let mut min_rank = None;

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if let Some(rank) = node.simplex_rank {
                if let Some(level) = ranks.get_mut(&rank) {
                    level.push(node_idx);
                } else {
                    ranks.insert(rank, vec![node_idx]);
                }

                min_rank = if let Some(min_rank) = min_rank {
                    Some(u32::min(min_rank, rank))
                } else {
                    Some(rank)
                };
            }
        }

        (min_rank, ranks)
    }

    /// Generic Network simplex:
    ///
    /// fn network_simplex_example() {
    ///     network.init()
    ///     while network.not_optimized(network.optimization_value()) {
    ///         iterate();
    ///     }
    ///  }
    ///  
    /// Documentation from paper: 4.2 Optimal Node Placement page 20
    /// * The method involves constructing an auxiliary graph as illustrated in ﬁgure 4-2.
    /// * This transformation is the graphical analogue of the algebraic transformation mentioned above
    ///   for removing the absolute values from the optimization problem.
    /// * The nodes of the auxiliary graph G′ are the nodes of the original graph G plus,
    ///   for every edge e in G, there is a new node n.  There are two kinds of edges in G′:
    ///    * One edge class encodes the cost of the original edges.
    ///      Every edge e=(u,v) in G is replaced by two edges (new_v_node, u) and (new_v_node, v)
    ///      with δ=0 and ω=ω(e) Ω(e).
    ///      * δ(e) = MinLen(edge(v, w)) = minimum length
    ///      * ω(e) = weight of edge e
    ///      * Ω(e) = function to bias for straighter long lines (see below)
    ///    * The other class of edges separates nodes in the same rank.  If v is the left neighbor of w,
    ///      then G′ has an edge f=e(v,w) with δ(f)=ρ(v,w) and ω(f)=0.
    ///      * This edge forces the nodes to be sufﬁciently separated but does not affect the cost of the layout.
    /// * We can now consider the level assignment problem on G′, which can be solved using the network
    ///   simplex method.
    ///   * Any solution of the positioning problem on G corresponds to a solution of the level assignment
    ///     problem on G′ with the same cost. This is achieved by assigning each new_v_node the value
    ///     min (x_u, x_v), using the notation of ﬁgure 4-2 and where x_u and x_v are the X coordinates assigned
    ///     to u and v in G.
    ///   * Conversely, any level assignment in G′ induces a valid positioning in G. In addition, in an optimal
    ///     level assignment, one of e_u or e_v must have length 0, and the other has length |x_u − x_v|. This
    ///     means the cost of an original edge (u ,v) in G equals the sum of the cost of the two edges e_u, e_v
    ///     in G′ and, globally, the two solutions have the same cost.
    ///   * Thus, optimality of G′ implies optimality for G and solving G′ gives us a solution for G.
    fn position(&mut self) {
        self.init_node_coordinates();
        let aux_graph = self.create_positioning_aux_graph();

        let mut improved = false;
        while !improved {
            let cur_value = aux_graph.graph_coordinate_optimization_value();
            // aux_graph.optimize_coordinates();
            let new_value = aux_graph.graph_coordinate_optimization_value();

            if cur_value < new_value {
                improved = true;
            } else {
                improved = false;
            }
        }
    }

    /// Initialize the x and y coordinates of all nodes.
    fn init_node_coordinates(&mut self) {
        for (node_idx, mut node) in self.nodes.iter().enumerate() {
            let min_x = node.min_seperation_x();
            let min_y = node.min_seperation_y();

            if let Some(mut coords) = node.coordinates {
                if let Some(position) = node.horizontal_position {
                    coords.set_x(position as u32 * min_x);
                } else {
                    panic!("Node {node_idx}: position not set");
                }

                if let Some(rank) = node.simplex_rank {
                    coords.set_y(rank as u32 * min_y);
                } else {
                    panic!("Node {node_idx}: rank not set");
                }
            }
        }
    }

    /// Documentation from paper: 4.2 Optimal Node Placement page 20
    /// * The method involves constructing an auxiliary graph as illustrated in ﬁgure 4-2.
    /// * This transformation is the graphical analogue of the algebraic transformation mentioned above
    ///   for removing the absolute values from the optimization problem.
    /// * The nodes of the auxiliary graph G′ are the nodes of the original graph G plus,
    ///   for every edge e in G, there is a new node n.  There are two kinds of edges in G′:
    ///    * One edge class encodes the cost of the original edges.
    ///      Every edge e=(u,v) in G is replaced by two edges (new_v_node, u) and (new_v_node, v)
    ///      with δ=0 and ω=ω(e) Ω(e).
    ///      * δ(e) = MinLen(edge(v, w)) = minimum length
    ///      * ω(e) = weight of edge e
    ///      * Ω(e) = function to bias for straighter long lines (see below)
    ///    * The other class of edges separates nodes in the same rank.  If v is the left neighbor of w,
    ///      then G′ has an edge f=e(v,w) with δ(f)=ρ(v,w) and ω(f)=0.
    ///      * This edge forces the nodes to be sufﬁciently separated but does not affect the cost of the layout.
    ///      
    ///  QUESTIONS: ☑ ☐ ☒
    ///  * For the new graph G':
    ///    * Are virtual nodes copied too? (almost certianly yes)
    ///  * Are the new nodes in graph G virtual nodes? (probably no, otherwise they would have said so)
    ///    * Are the new nodes in the same rank their source nodes?
    ///    * Do they have the same positions?
    ///  * Each old edge is replaced by two new edges, so the old edges are not in G'? (probably yes?)
    ///  * It seems the "second" class of edges which connect adjacent edges of the same rank
    ///
    /// * We can now consider the level assignment problem on G′, which can be solved using the network
    ///   simplex method.
    ///   * Any solution of the positioning problem on G corresponds to a solution of the
    ///     level assignment problem on G′ with the same cost.
    ///   * This is achieved by assigning each n_e the value min (x_u, x_v), using the notation of ﬁgure 4-2
    ///     and where x_u and x_v are the X coordinates assigned to u and v in G.
    ///   * Conversely, any level assignment in G′ induces a valid positioning in G.
    ///   * In addition, in an optimal level assignment, one of e_u or e must have length 0, and the other
    ///     has length |x_u − x_v|. This means the cost of an original edge (u, v) in G equals the sum
    ///     of the cost of the two edges e_y, e_v in G′ and, globally, the two solutions have the same cost.
    ///   * Thus, optimality of G′ implies optimality for G and solving G ′ gives us a solution for G.
    fn create_positioning_aux_graph(&self) -> Graph {
        let mut aux_graph = Graph::new();

        /// Add all the existing edges to the current graph without edges.
        for node in self.nodes.iter() {
            let mut new_node = node.clone();
            new_node.clear_edges();

            aux_graph.nodes.push(new_node)
        }

        // Add the new node and edges between the new node, as per figure 4-2 in paper
        // Note that we are adding 2 edges and one node for every edge, but not the orignal edge itself.
        for edge in self.edges.iter() {
            let src_node_idx = edge.src_node;
            let dst_node_idx = edge.dst_node;
            let src_node = self.get_node(src_node_idx);
            let dst_node = self.get_node(dst_node_idx);

            // QUESTION: Should the new node be a virtual node?  Unclear from text.
            let new_node_idx =
                aux_graph.add_node(&format!("{}-{}", &src_node.name, &dst_node.name));
            let omega = Edge::edge_omega_value(src_node.virtual_node, dst_node.virtual_node);
            let new_weight = edge.weight * omega;

            let new_edge_idx_1 = aux_graph.add_edge(new_node_idx, src_node_idx);
            {
                let new_edge_1 = aux_graph.get_edge_mut(new_edge_idx_1);
                new_edge_1.weight = new_weight;
            }
            let new_edge_idx_2 = aux_graph.add_edge(new_node_idx, dst_node_idx);
            {
                let new_edge_2 = aux_graph.get_edge_mut(new_edge_idx_2);
                new_edge_2.weight = new_weight;
            }

            /// TODO: deal with unwraps();
            let new_node = aux_graph.get_node_mut(new_node_idx);
            let src_x = src_node.coordinates.unwrap().x();
            let dst_x = dst_node.coordinates.unwrap().x();

            /// ...assigning each n_e the value min(x_u, x_v), using the notation of ﬁgure 4-2
            /// and where x_u and x_v are the X coordinates assigned to u and v in G.
            new_node.coordinates.unwrap().set_x(src_x.min(dst_x));

            new_node.simplex_rank = src_node.simplex_rank; // WHY? Probably rank does not matter because rank is not in the optimization formula
        }

        // Add edges between adjacent nodes in a rank, as per figure 4-2 in the paper.
        // TODO

        aux_graph
    }

    fn make_splines(&mut self) {
        todo!();
    }

    /// Return a value for the graph used to optimize the graph for the selection of x coordiantes for nodes.
    ///
    /// Documentation from paper: page 17
    fn graph_coordinate_optimization_value(&self) -> u32 {
        self.edges
            .iter()
            .map(|edge| self.edge_coordinate_optimization_value(edge))
            .sum()
    }

    /// Return a value for an edge used to optimize the graph for the selection of x coordiantes for nodes.
    ///
    /// Documentation from paper: page 17
    /// * For edge = (v,w): Ω(e)*ω(e)*|Xw − Xv|
    /// * Subject to: Xb − Xa ≥ ρ(a,b)
    ///   * ρ is a function on pairs of adjacent nodes in the same rank giving the minimum separation
    ///     between their center points
    fn edge_coordinate_optimization_value(&self, edge: &Edge) -> u32 {
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);
        let omega = Edge::edge_omega_value(src_node.virtual_node, dst_node.virtual_node);
        let weight = edge.weight;
        let w_x = src_node.coordinates.unwrap().x();
        let v_x = dst_node.coordinates.unwrap().y();
        let x_diff = w_x.abs_diff(v_x);

        self.min_node_distance().max(omega * weight * x_diff)
    }

    /// TODO: min_node_distance should be determined by graph context, and perhaps the specific nodes
    ///       involved
    fn min_node_distance(&self) -> u32 {
        const MIN_NODE_DISTANCE: u32 = 100; // pixels...

        MIN_NODE_DISTANCE
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

#[cfg(test)]
mod tests {
    use tests::edge::MIN_EDGE_WEIGHT;

    use super::*;
    use std::ops::RangeInclusive;

    /// Additional test only functions for Graph to make graph construction testing easier.
    impl Graph {
        /// Add multiple nodes with names given from a range of characters.
        ///
        /// Returns a HashMap of the created node names and node indexes to be
        /// used with add_edges().
        ///
        /// * Nodes must be named after a single character.
        /// * The range is inclusive only of the left side.  So 'a'..'d' incluses: a, b, c but NOT d.
        fn add_nodes(&mut self, range: RangeInclusive<char>) -> HashMap<String, usize> {
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
        pub fn name_to_node_idx(&self, name: &str) -> Option<usize> {
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

            node.simplex_rank = Some(rank);
            node.tree_node = true;
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

        graph.get_node_mut(a_idx).tree_node = true;
        let min_edge_idx = graph.get_min_incident_edge();
        assert_eq!(min_edge_idx, Some(e1));

        graph.get_node_mut(b_idx).tree_node = true;
        let min_edge_idx = graph.get_min_incident_edge();
        assert_eq!(min_edge_idx, Some(e2));

        graph.get_node_mut(c_idx).tree_node = true;
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

        graph.get_node_mut(a_idx).tree_node = true;
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

        assert_eq!(graph.simplex_edge_length(a_b), None);

        graph.init_rank();
        println!("{graph}");

        assert_eq!(graph.simplex_edge_length(a_b), Some(1));
        assert_eq!(graph.simplex_edge_length(a_c), Some(2));
        assert_eq!(graph.simplex_edge_length(b_c), Some(1));
        // assert_eq!(graph.edge_length(c_a), Some(-2));
    }

    #[test]
    fn test_set_edge_simplex_slack() {
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

        assert_eq!(graph.simplex_slack(a_b), None);

        graph.init_rank();
        println!("{graph}");

        assert_eq!(graph.simplex_slack(a_b), Some(0));
        assert_eq!(graph.simplex_slack(a_c), Some(1));
        assert_eq!(graph.simplex_slack(b_c), Some(0));
        // assert_eq!(graph.edge_length(c_a), Some(-2));
    }

    pub fn example_graph_from_paper_2_3() -> Graph {
        let mut graph = Graph::new();
        let node_map = graph.add_nodes('a'..='h');
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
    fn test_get_source_nodes_single() {
        let mut graph = Graph::new();
        let node_map = graph.add_nodes('a'..='d');
        let edges = vec![("a", "b"), ("c", "d"), ("d", "c")];
        graph.add_edges(&edges, &node_map);

        let source_nodes = graph.get_source_nodes();
        let source_nodes = source_nodes.iter().cloned().collect::<Vec<usize>>();

        assert_eq!(source_nodes, vec![0]);
    }

    #[test]
    fn test_get_source_nodes_double() {
        let mut graph = Graph::new();
        let node_map = graph.add_nodes('a'..='c');
        let edges = vec![("a", "b"), ("c", "b")];
        graph.add_edges(&edges, &node_map);

        let source_nodes = graph.get_source_nodes();
        let source_nodes = source_nodes.iter().cloned().collect::<Vec<usize>>();

        assert_eq!(source_nodes, vec![0, 2]);
    }

    /// Test that two simple cyclic graphs are both made asyclic.
    #[test]
    fn test_make_asyclic() {
        let mut graph = Graph::new();

        let node_map = graph.add_nodes('a'..='d');
        let edges = vec![("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")];
        graph.add_edges(&edges, &node_map);

        graph.make_asyclic();

        println!("{graph}");

        let a_b = graph.get_edge(0);
        let b_a = graph.get_edge(1);
        let c_d = graph.get_edge(2);
        let d_c = graph.get_edge(3);

        assert!(!a_b.reversed);
        assert!(b_a.reversed);
        assert!(!c_d.reversed);
        assert!(d_c.reversed);

        assert_eq!(a_b.src_node, *node_map.get("a").unwrap());
        assert_eq!(a_b.dst_node, *node_map.get("b").unwrap());
        assert_eq!(b_a.src_node, *node_map.get("a").unwrap());
        assert_eq!(b_a.dst_node, *node_map.get("b").unwrap());
    }

    #[test]
    fn test_get_min_rank() {
        let mut graph = Graph::new();
        let node_map = graph.add_nodes('a'..='d');
        let edges = vec![("a", "b"), ("b", "a"), ("c", "d"), ("c", "a")];
        graph.add_edges(&edges, &node_map);

        graph.rank();

        let min_rank = graph.get_min_rank_nodes();
        let min_rank = min_rank.iter().cloned().collect::<Vec<usize>>();

        assert_eq!(
            min_rank,
            vec![*node_map.get("c").unwrap()],
            "min node should be 'c'"
        );
    }

    #[test]
    fn test_get_initial_ordering() {
        let mut graph = Graph::new();
        let node_map = graph.add_nodes('a'..='e');
        let edges = vec![("a", "b"), ("b", "c"), ("b", "d"), ("c", "e"), ("d", "e")];
        graph.add_edges(&edges, &node_map);

        graph.rank();

        println!("{graph}");
        let order = graph.get_initial_ordering();

        println!("{order:?}");
    }

    #[test]
    fn test_rank() {
        let mut graph = example_graph_from_paper_2_3();

        graph.rank();

        println!("{graph}");
    }

    // #[test]
    // fn test_rank() {
    //     let mut graph = Graph::new();
    //     let a_idx = graph.add_node("A");
    //     let b_idx = graph.add_node("B");
    //     let c_idx = graph.add_node("C");
    //     let d_idx = graph.add_node("D");

    //     graph.add_edge(a_idx, b_idx);
    //     graph.add_edge(a_idx, c_idx);
    //     graph.add_edge(b_idx, d_idx);
    //     graph.add_edge(c_idx, d_idx);

    //     graph.rank();

    //     println!("{graph}");

    //     assert_eq!(graph.nodes[a_idx].rank, Some(0));
    //     assert_eq!(graph.nodes[b_idx].rank, Some(1));
    //     assert_eq!(graph.nodes[c_idx].rank, Some(1));
    //     assert_eq!(graph.nodes[d_idx].rank, Some(2));
    // }

    // #[test]
    // fn test_rank_scaning() {
    //     let mut graph = Graph::new();
    //     let a_idx = graph.add_node("A");
    //     let b_idx = graph.add_node("B");
    //     let c_idx = graph.add_node("C");

    //     graph.add_edge(a_idx, b_idx);
    //     graph.add_edge(a_idx, c_idx);
    //     graph.add_edge(b_idx, c_idx);

    //     graph.rank();
    //     println!("{graph}");

    //     assert_eq!(graph.nodes[a_idx].rank, Some(0));
    //     assert_eq!(graph.nodes[b_idx].rank, Some(1));
    //     assert_eq!(graph.nodes[c_idx].rank, Some(2));
    // }

    #[test]
    fn test_fill_rank_gaps() {
        let (mut graph, _expected_cutvals) = Graph::configure_example_2_3_a();
        graph.init_cutvalues();
        let order = graph.get_initial_ordering();

        println!("{graph}");
        graph.fill_rank_gaps(&order);
        println!("{graph}");

        for (edge_idx, _edge) in graph.edges.iter().enumerate() {
            if let Some(len) = graph.simplex_edge_length(edge_idx) {
                assert!(len.abs() <= MIN_EDGE_LENGTH as i32)
            }
        }
    }

    #[test]
    fn test_draw_graph() {
        let mut graph = example_graph_from_paper_2_3();

        println!("{graph}");
        graph.draw_graph();
        println!("{graph}");
    }
}
