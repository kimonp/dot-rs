//! Methods for graph that implement the network simplex algorithm.
//!
//! Network simplex is a general algorithm to find the minimum cost
//! through a network.
//!
//! <https://en.wikipedia.org/wiki/Network_simplex_algorithm>

use super::{
    edge::{EdgeDisposition, MIN_EDGE_LENGTH},
    Graph,
};
use std::collections::HashSet;

mod spanning_tree;

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
        self.init_cutvalues();

        let mut start_idx = 0;
        while let Some(neg_cut_edge_idx) = self.leave_edge_for_simplex(start_idx) {
            let non_tree_edge_idx = self
                .enter_edge_for_simplex(neg_cut_edge_idx)
                .expect("No negative cut values found!");

            println!("Exchanging {neg_cut_edge_idx} with {non_tree_edge_idx}");
            self.exchange(neg_cut_edge_idx, non_tree_edge_idx);

            start_idx = neg_cut_edge_idx + 1;
        }
        self.normalize_simplex_rank();
        self.balance(target);
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
            if edge.in_spanning_tree() {
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

                    edge_idx != cut_edge_idx && self.get_edge(edge_idx).in_spanning_tree()
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

        !src_node.in_spanning_tree() && dst_node.in_spanning_tree()
    }

    /// edge_index is expected to span two nodes, one of which is in the tree, one of which is not.
    /// Return the index to the node which is not yet in the tree.
    fn get_incident_node(&self, edge_idx: usize) -> Option<usize> {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);

        if !src_node.in_spanning_tree() && dst_node.in_spanning_tree() {
            Some(edge.src_node)
        } else if src_node.in_spanning_tree() && !dst_node.in_spanning_tree() {
            Some(edge.dst_node)
        } else {
            None
        }
    }

    /// Return an edge with the smallest slack of any edge which is incident to the tree.
    ///
    /// Incident to the tree means one point of the edge points to a node that is in the tree,
    /// and the other point points to a node that it not within the tree.
    ///
    /// TODO: Make more efficient by keeping a list of incident nodes
    ///
    /// Optimization TODO from the paper:
    /// * The network simplex is also very sensitive to the choice of the negative edge to replace.
    /// * We observed that searching cyclically through all the tree edges, instead of searching from the
    ///   beginning of the list of tree edges every time, can save many iterations.
    fn get_min_incident_edge(&self) -> Option<usize> {
        let mut candidate = None;
        let mut candidate_slack = i32::MAX;

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.in_spanning_tree() {
                for edge_idx in node.get_all_edges() {
                    let connected_node_idx = self
                        .get_connected_node(node_idx, *edge_idx)
                        .expect("Edge not connected");

                    if !self.get_node(connected_node_idx).in_spanning_tree() {
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
            if node.in_spanning_tree() {
                for edge_idx in &node.out_edges {
                    let dst_node = self.get_edge(*edge_idx).dst_node;

                    if !self.get_node(dst_node).in_spanning_tree() {
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

        // Lets not assume the tree fields are clear to begin with
        for edge in self.edges.iter_mut() {
            edge.set_in_spanning_tree(false)
        }

        // Initialize the queue with all nodes with no incoming edges (since no edges
        // are scanned yet)
        for (index, node) in self.nodes.iter_mut().enumerate() {
            node.set_simplex_rank(None);
            node.clear_tree_data();

            if node.no_in_edges() {
                nodes_to_rank.push(index);
            }
        }

        while !nodes_to_rank.is_empty() {
            let mut next_nodes_to_rank = Vec::new();
            while let Some(node_idx) = nodes_to_rank.pop() {
                let node = self.get_node(node_idx);
                if node.simplex_rank.is_none() {
                    let mut new_rank = 0;

                    for edge_idx in node.in_edges.clone() {
                        let edge = self.get_edge(edge_idx);
                        let src_node = self.get_node(edge.src_node);

                        new_rank = if let Some(src_rank) = src_node.simplex_rank() {
                            new_rank.max(src_rank + MIN_EDGE_LENGTH)
                        } else {
                            new_rank
                        };
                    }
                    for edge_idx in node.out_edges.clone() {
                        scanned_edges.insert(edge_idx);

                        let edge = self.get_edge(edge_idx);
                        let dst_node = self.get_node(edge.dst_node);
                        if dst_node.no_unscanned_in_edges(&scanned_edges) {
                            next_nodes_to_rank.push(edge.dst_node);
                        }
                    }
                    self.get_node_mut(node_idx).set_simplex_rank(Some(new_rank));
                }
            }

            next_nodes_to_rank
                .iter()
                .for_each(|idx| nodes_to_rank.push(*idx));
        }
    }

    // pub(super) fn init_simplex_rank_old(&mut self) {
    //     let mut nodes_to_rank = Vec::new();
    //     let mut scanned_edges = HashSet::new();

    //     // Initialize the queue with all nodes with no incoming edges (since no edges
    //     // are scanned yet)
    //     for (index, node) in self.nodes.iter_mut().enumerate() {
    //         node.set_simplex_rank(None);
    //         node.set_tree_node(false);

    //         if node.no_in_edges() {
    //             nodes_to_rank.push(index);
    //         }
    //     }

    //     let mut cur_rank = 0;
    //     while !nodes_to_rank.is_empty() {
    //         let mut next_nodes_to_rank = Vec::new();
    //         while let Some(node_idx) = nodes_to_rank.pop() {
    //             let node = self.get_node_mut(node_idx);

    //             if node.simplex_rank.is_none() {
    //                 node.set_simplex_rank(Some(cur_rank));

    //                 for edge_idx in node.out_edges.clone() {
    //                     scanned_edges.insert(edge_idx);

    //                     let edge = self.get_edge(edge_idx);
    //                     let dst_node = self.get_node(edge.dst_node);
    //                     if dst_node.no_unscanned_in_edges(&scanned_edges) {
    //                         next_nodes_to_rank.push(edge.dst_node);
    //                     }
    //                 }
    //             }
    //         }

    //         cur_rank += 1;
    //         next_nodes_to_rank
    //             .iter()
    //             .for_each(|idx| nodes_to_rank.push(*idx));
    //     }
    // }

    /// If any edge has a negative cut value, return the first one found starting with start_idx.
    /// Otherwise, return None.
    ///
    /// start_idx is used to support a finding in the paper that network simplex effeciency is very sensitive to
    /// the choice of edge via leave_edge(). From the paper:
    ///   The network simplex is also very sensitive to the choice of the negative edge to replace.  We observed
    ///   that searching cyclically through all the tree edges, instead of searching from the beginning of the
    ///   list of tree edges every time, can save many iterations
    fn leave_edge_for_simplex(&self, start_idx: usize) -> Option<usize> {
        // Search starting at start_idx
        let mut edge_idx = start_idx;

        while edge_idx < self.edges.len() {
            let edge = self.get_edge(edge_idx);
            println!("Checking from {edge_idx}: {:?}", edge.cut_value);

            if let Some(cut_value) = edge.cut_value {
                if cut_value < 0 {
                    return Some(edge_idx);
                }
            }
            edge_idx += 1;
        }
        // Search starting at zero up to start_idx
        println!("Checking from 0-{start_idx}");
        edge_idx = 0;
        while edge_idx < start_idx {
            let edge = self.get_edge(edge_idx);

            if let Some(cut_value) = edge.cut_value {
                if cut_value < 0 {
                    return Some(edge_idx);
                }
            }
            edge_idx += 1;
        }
        None
    }

    /// Given a tree edge, return the non-tree edge with the lowest remaining cut-value.
    ///
    /// Documentation from paper:
    /// * enter_edge ﬁnds a non-tree edge to replace e.
    ///   * This is done by breaking the edge e, which divides the tree into a head and tail component.
    ///   * All edges going from the head component to the tail are considered, with an edge of minimum slack being chosen.
    ///   * This is necessary to maintain feasibility.
    fn enter_edge_for_simplex(&self, tree_edge_idx: usize) -> Option<usize> {
        let (head_nodes, tail_nodes) = self.get_components(tree_edge_idx);

        let mut min_slack = i32::MAX;
        let mut replacement_edge_idx = None;

        for (edge_idx, edge) in self.edges.iter().enumerate() {
            if head_nodes.contains(&edge.src_node) && tail_nodes.contains(&edge.dst_node) {
                let edge_slack = self.simplex_slack(edge_idx).expect("Can't calculate slack");

                if edge_slack < min_slack {
                    replacement_edge_idx = Some(edge_idx);
                    min_slack = edge_slack;
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
            neg_cut_edge.set_in_spanning_tree(false);

            let non_tree_edge = self.get_edge_mut(non_tree_edge_idx);
            non_tree_edge.set_in_spanning_tree(true);
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

    /// Balance nodes either top to bottom or left to right
    fn balance(&mut self, target: SimplexNodeTarget) {
        match target {
            SimplexNodeTarget::VerticalRank => self.balance_top_bottom(),
            SimplexNodeTarget::XCoordinate => self.balance_left_right(),
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
    ///  
    /// TB_balance() function in lib/common/ns.c in GraphViz 9.0:
    ///   Try to improve the top-bottom balance of the number of nodes in ranks.
    ///   * Only look at non-virtual nodes
    ///   * Get a count of how many nodes are in each rank.
    ///   * Sort nodes so that we can step through them by most popular rank to least popular.
    ///   * For each node, ranked by rank population (higest to lowest)
    ///     * consider all the ranks between the highest rank and lowest
    ///       rank the node is connected to another node.
    ///     * If any of those ranks has fewer nodes then where it is currently,
    ///       move it there
    fn balance_top_bottom(&mut self) {
        // TODO!
    }

    ///
    ///From GraphViz 9.0.0:
    /// * Initializes DFS range attributes (par, low, lim) over tree nodes such that:
    /// * ND_par(n) - parent tree edge
    /// * ND_low(n) - min DFS index for nodes in sub-tree (>= 1)
    /// * ND_lim(n) - max DFS index for nodes in sub-tree
    ///
    fn balance_left_right(&mut self) {
        for (tree_edge_idx, tree_edge) in self
            .edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| edge.in_spanning_tree())
        {
            if tree_edge.cut_value == Some(0) {
                if let Some(replace_edge_idx) = self.enter_edge_for_simplex(tree_edge_idx) {
                    if let Some(delta) = self.simplex_slack(replace_edge_idx) {
                        if delta > 1 {
                            let edge = self.get_edge(replace_edge_idx);
                            let src_node = self.get_node(edge.src_node);
                            let dst_node = self.get_node(edge.dst_node);

                            // ND_lim(n) is the max depth first index for nodes in the subtree of node n.
                            // * How many steps from the root is the node connected to n that is farthest
                            //   farthest from the root.
                            // So if src_node is connected to nodes farther from the root than src_node:
                            //   reduce by delta/2 the rank of dst_node and all nodes under it
                            // else
                            //   increase by delta/2 the rank of dst_node and all nodes under it
                            //
                            // if ND_lim(src_node) < ND_lim(dst_node) {
                            //     // rerank(n) subtracts delta to the rank of all subtree nodes:
                            //     //
                            //     // * reduces the rank of src_node by delta
                            //     // * for each outgoing edge of the tree from n
                            //     //   * if edge is not heading to the parent of n
                            //     //      * call rerank(e.dst_node, delta)
                            //     // * for each incoming edge of the tree from n
                            //     //   * if edge is not heading to the parent of n
                            //     //      * call rerank(e.src_node, delta)
                            //     rerank(src_node, delta / 2);
                            // } else {
                            //     rerank(dst_node, -delta / 2)
                            // }
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Graph {
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
        fn tree_node_count(&self) -> usize {
            self.nodes
                .iter()
                .filter(|node| node.in_spanning_tree())
                .count()
        }

        fn assert_expected_cutvals(&self, expected_cutvals: Vec<(&str, &str, i32)>) {
            for (src_name, dst_name, cut_val) in expected_cutvals {
                let (edge, _) = self.get_named_edge(src_name, dst_name);

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

        graph.get_node_mut(a_idx).set_tree_root_node();
        let min_edge_idx = graph.get_min_incident_edge();
        assert_eq!(min_edge_idx, Some(e1));

        graph.get_node_mut(b_idx).set_tree_root_node();
        let min_edge_idx = graph.get_min_incident_edge();
        assert_eq!(min_edge_idx, Some(e2));

        graph.get_node_mut(c_idx).set_tree_root_node();
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

        graph.get_node_mut(a_idx).set_tree_root_node();
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

    #[test]
    fn setup_init_cut_values_2_3_extended() {
        let (mut graph, expected_cutvals) = Graph::configure_example_2_3_extended();

        graph.init_cutvalues();
        println!("{graph}");
        graph.assert_expected_cutvals(expected_cutvals);
    }

    #[test]
    fn test_leave_edge_for_simplex() {
        let (mut graph, _expected_cutvals) = Graph::configure_example_2_3_extended();
        graph.set_feasible_tree_for_simplex();
        graph.init_cutvalues();

        let (_, neg_edge1_idx) = graph.get_named_edge("g", "h");
        let (_, neg_edge2_idx) = graph.get_named_edge("l", "h");

        let neg_edge = graph.leave_edge_for_simplex(0);
        assert_eq!(neg_edge, Some(neg_edge1_idx));

        let neg_edge2 = graph.leave_edge_for_simplex(neg_edge.unwrap() + 1);
        assert_eq!(neg_edge2, Some(neg_edge2_idx));

        // Expect it to wrap around
        let neg_edge3 = graph.leave_edge_for_simplex(neg_edge2.unwrap() + 1);
        assert_eq!(neg_edge3, Some(neg_edge1_idx));
    }

    /// enter_edge_for_simplex is supposed to find the next edge with the minimum cut value.
    ///
    /// Given a specific example, we know which edges we expect it to return.
    #[test]
    fn test_enter_edge_for_simplex() {
        let (mut graph, _expected_cutvals) = Graph::configure_example_2_3_extended();
        graph.set_feasible_tree_for_simplex();
        graph.init_cutvalues();

        assert_eq!(
            graph.tree_node_count(),
            graph.node_count(),
            "all nodes should be tree nodes"
        );

        let (_, neg_edge1_idx) = graph.get_named_edge("e", "g");
        let (_, neg_edge2_idx) = graph.get_named_edge("i", "l");

        let neg_edge = graph.leave_edge_for_simplex(0).unwrap();
        let neg_edge2 = graph.leave_edge_for_simplex(neg_edge + 1).unwrap();

        let new_edge = graph.enter_edge_for_simplex(neg_edge).unwrap();
        assert_eq!(new_edge, neg_edge1_idx);

        // let new_edge = graph.get_edge(new_edge);
        // let src_node = graph.get_node(new_edge.src_node);
        // let dst_node = graph.get_node(new_edge.dst_node);
        // println!("{graph}");
        // println!("Next edge 1: {} -> {}", src_node, dst_node);

        let new_edge2 = graph.enter_edge_for_simplex(neg_edge2).unwrap();
        assert_eq!(new_edge2, neg_edge2_idx);

        // let new_edge2 = graph.get_edge(new_edge2);
        // let src_node = graph.get_node(new_edge2.src_node);
        // let dst_node = graph.get_node(new_edge2.dst_node);
        // println!("Next edge 2: {} -> {}", src_node, dst_node);
    }

    #[test]
    fn test_network_simplex_ranking() {
        let mut graph = Graph::example_graph_from_paper_2_3_extended();

        graph.network_simplex_ranking(SimplexNodeTarget::VerticalRank);
    }
}
