
//! Methods for graph that implement the network simplex algorithm.
//! 
//! Network simplex is a general algorithm to find the minimum cost
//! through a network.
//!
//! <https://en.wikipedia.org/wiki/Network_simplex_algorithm>

use std::collections::HashSet;
use super::{edge::MIN_EDGE_LENGTH, Graph};

/// Determines what variable on each node which is set by the network simplex algorithm.
#[derive(Debug, Clone, Copy)]
pub(crate) enum SimplexNodeTarget {
    /// The rank of each node, which corresponds with to the ultimate y position of the node
    VerticalRank,
    /// The X coordinate of each node
    XCoordinate,
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

impl Graph {
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
    pub(super) fn network_simplex_ranking(&mut self, target: SimplexNodeTarget) {
        self.set_feasible_tree_for_simplex();
        while let Some(neg_cut_edge_idx) = self.leave_edge_for_simplex() {
            let non_tree_edge_idx = self
                .enter_edge_for_simplex(neg_cut_edge_idx)
                .expect("No negative cut values found!");
            self.exchange(neg_cut_edge_idx, non_tree_edge_idx);
        }
        self.normalize_simplex_rank();
        // self.balance();
        self.assign_simplex_rank(target);
    }

    /// After running the network simplex algorithm, assign the result to each node.
    ///
    /// The value assigned to depends on the given target.
    pub(super) fn assign_simplex_rank(&mut self, target: SimplexNodeTarget) {
        for node in self.nodes.iter_mut() {
            node.assign_simplex_rank(target);
        }
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
    fn set_feasible_tree_for_simplex(&mut self) {
        self.init_simplex_rank();

        for node in self.nodes.iter_mut() {
            node.tree_node = node.no_out_edges();
        }

        while self.tight_simplex_tree() < self.node_count() {
            // e = a non-tree edge incident on the tree with a minimal amount of slack
            // delta = slack(e);
            // if includent_node is e.head then delta = -delta
            // for v in Tree do v.rank = v.rank + delta;
            let edge_idx = self
                .get_min_incident_edge()
                .expect("No incident edges left!");
            let delta = if let Some(delta) = self.simplex_slack(edge_idx) {
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
    pub(super) fn init_cutvalues(&mut self) {
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
    fn tight_simplex_tree(&self) -> usize {
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
    #[allow(unused)]
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
    #[allow(unused)]
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
    pub(super) fn simplex_slack(&self, edge_idx: usize) -> Option<i32> {
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
    pub(super) fn simplex_edge_length(&self, edge_idx: usize) -> Option<i32> {
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
    pub(super) fn init_simplex_rank(&mut self) {
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
    fn leave_edge_for_simplex(&self) -> Option<usize> {
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
    fn enter_edge_for_simplex(&self, neg_cut_edge_idx: usize) -> Option<usize> {
        let (head_nodes, tail_nodes) = self.get_components(neg_cut_edge_idx);

        let min_slack = i32::MAX;
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
    fn normalize_simplex_rank(&mut self) {
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
}

#[cfg(test)]
mod tests {
    use super::*;
    
    impl Graph {
        fn assert_expected_cutvals(&self, expected_cutvals: Vec<(&str, &str, i32)>) {
            for (src_name, dst_name, cut_val) in expected_cutvals {
                let edge = self.get_named_edge(src_name, dst_name);

                assert_eq!(edge.cut_value, Some(cut_val), "unexpected cut_value");
            }
        }

        #[allow(unused)]
        fn display_component(&self, comp: &HashSet<usize>) -> String {
            let mut node_names = vec![];
            for node_idx in comp {
                node_names.push(self.get_node(*node_idx).name.clone());
            }
            node_names.sort();

            format!("{node_names:?}",)
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

        graph.init_simplex_rank();

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

        graph.init_simplex_rank();

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

        graph.init_simplex_rank();
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

        graph.init_simplex_rank();
        println!("{graph}");

        assert_eq!(graph.simplex_slack(a_b), Some(0));
        assert_eq!(graph.simplex_slack(a_c), Some(1));
        assert_eq!(graph.simplex_slack(b_c), Some(0));
        // assert_eq!(graph.edge_length(c_a), Some(-2));
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

}