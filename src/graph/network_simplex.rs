//! Methods for graph that implement the network simplex algorithm.
//!
//! Network simplex is a general algorithm to find the minimum cost
//! through a network.
//!
//! <https://en.wikipedia.org/wiki/Network_simplex_algorithm>

use super::{
    edge::{
        EdgeDisposition,
        EdgeDisposition::{In, Out},
    },
    Graph,
};
use std::collections::{HashSet, VecDeque};

mod heap;
pub(super) mod spanning_tree;
pub(super) mod cutvalue;
pub(crate) mod sub_tree;

/// Determines what variable on each node which is set by the network simplex algorithm.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum SimplexNodeTarget {
    /// The rank of each node, which corresponds with to the ultimate y position of the node
    VerticalRank,
    /// The X coordinate of each node
    XCoordinate,
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
        self.set_feasible_tree_for_simplex(target == SimplexNodeTarget::VerticalRank);

        let mut start_idx = 0;
        while let Some(neg_cut_edge_idx) = self.leave_edge_for_simplex(start_idx) {
            println!("About to enter: {neg_cut_edge_idx}");

            if let Some((selected_edge_idx, selected_slack)) =
                self.enter_edge_for_simplex(neg_cut_edge_idx)
            {
                // println!(
                //     "Exchanging edges with slack {selected_slack}\n  remove from tree: {}\n       add to tree: {}",
                //     self.edge_to_string(neg_cut_edge_idx),
                //     self.edge_to_string(selected_edge_idx),
                // );
                self.rerank_for_simplex(neg_cut_edge_idx, selected_slack);
                self.adjust_cutvalues_and_exchange_for_simplex(neg_cut_edge_idx, selected_edge_idx);

                start_idx = neg_cut_edge_idx + 1;
            } else {
                // println!("No negative cut values!");
                break;
            }
        }
        // self.print_nodes("after simplex loop");

        // self.normalize_simplex_rank(); GraphViz only does this in the "default" case (not TB or LR. We don't have that target)
        self.balance_for_simplex(target);
        self.assign_simplex_rank(target);
        if target == SimplexNodeTarget::XCoordinate {
            // self.print_nodes("network_simplex_ranking after balance_left_right END");
        }
    }

    /// Rerank nodes based on which edge was removed from the tree during a loop of network simplex.
    ///
    /// This makes edges "tight" as result of removing an edge with a negative cut value.
    fn rerank_for_simplex(&self, prev_tree_edge_idx: usize, delta: i32) {
        if delta > 0 {
            let edge = self.get_edge(prev_tree_edge_idx);
            let src_idx = edge.src_node; // tail
            let dst_idx = edge.dst_node; // head

            let src_node_in_cnt = self.node_tree_edges(src_idx, In).len();
            let src_node_out_cnt = self.node_tree_edges(src_idx, Out).len();
            let size = (src_node_in_cnt + src_node_out_cnt) as i32;

            let (rerank_idx, rerank_delta) = if size == 1 {
                (src_idx, delta)
            } else {
                let dst_node_in_cnt = self.node_tree_edges(dst_idx, In).len();
                let dst_node_out_cnt = self.node_tree_edges(dst_idx, Out).len();
                let size = (dst_node_in_cnt + dst_node_out_cnt) as i32;

                if size == 1 {
                    (dst_idx, -delta)
                } else if let (Some(src_tree), Some(dst_tree)) = (
                    self.get_node(src_idx).spanning_tree(),
                    self.get_node(dst_idx).spanning_tree(),
                ) {
                    if src_tree.tree_dist_max() < dst_tree.tree_dist_max() {
                        (src_idx, delta)
                    } else {
                        (dst_idx, -delta)
                    }
                } else {
                    panic!("Not all nodes in spanning tree!");
                }
            };
            self.rerank_by_tree(rerank_idx, rerank_delta);
        }
    }

    /// After running the network simplex algorithm, assign the result to each node.
    ///
    /// The value assigned to depends on the given target.
    pub(super) fn assign_simplex_rank(&mut self, target: SimplexNodeTarget) {
        for node in self.nodes.iter_mut() {
            node.assign_simplex_rank(target);
        }
    }

    /// Returns the slack of and edge for the network simplex algorithm.
    ///
    /// The slack of an edge is the difference of its length and its minimum length.
    ///
    /// An edge is "tight" if it's slack is zero.
    pub(super) fn simplex_slack(&self, edge_idx: usize) -> Option<i32> {
        self.simplex_edge_length(edge_idx).map(|len| {
            let min_len = self.get_edge(edge_idx).min_len();
            if len > 0 {
                len - min_len
            } else {
                len + min_len
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

        match (src_node.simplex_rank(), dst_node.simplex_rank()) {
            (Some(src), Some(dst)) => Some(dst - src),
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
        let mut nodes_to_rank = VecDeque::new();
        let mut scanned_edges = HashSet::new();

        // self.print_nodes("before init_simplex_rank()");

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
                nodes_to_rank.push_back(index);
            }
        }

        while let Some(node_idx) = nodes_to_rank.pop_front() {
            let node = self.get_node(node_idx);
            // println!("  init_simplex_rank(): Processing: {}", self.node_to_string(node_idx));

            if node.simplex_rank().is_none() {
                let mut new_rank = 0;

                for edge_idx in node.in_edges.clone() {
                    let edge = self.get_edge(edge_idx);
                    let src_node = self.get_node(edge.src_node);

                    new_rank = if let Some(src_rank) = src_node.simplex_rank() {
                        new_rank.max(src_rank + edge.min_len())
                    } else {
                        new_rank.max(edge.min_len())
                    };
                }
                for edge_idx in node.out_edges.clone() {
                    scanned_edges.insert(edge_idx);

                    let edge = self.get_edge(edge_idx);
                    let dst_node = self.get_node(edge.dst_node);
                    if dst_node.no_unscanned_in_edges(&scanned_edges) {
                        // println!("  init_simples_rank():    queueing: {}", self.get_node(edge.dst_node).name);
                        nodes_to_rank.push_back(edge.dst_node)
                    }
                }
                self.get_node_mut(node_idx).set_simplex_rank(Some(new_rank));
            }
        }

        // self.print_nodes("after init_simplex_rank()");
    }

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
        for edge_idx in start_idx..self.edges.len() {
            let edge = self.get_edge(edge_idx);
            if let Some(cut_value) = edge.cut_value {
                if cut_value < 0 {
                    return Some(edge_idx);
                }
            }
        }
        // Now start at zero
        for edge_idx in 0..start_idx {
            let edge = self.get_edge(edge_idx);
            if let Some(cut_value) = edge.cut_value {
                if cut_value < 0 {
                    return Some(edge_idx);
                }
            }
        }

        None
    }

    /// Given a tree edge, return the non-tree edge with the lowest remaining cut-value.
    ///
    /// * Determines if we will be searching In or Out from the src_node or the dst_node respectively
    ///   based on the whichever has the smallest subtree max.
    /// * Passes the search_node and min/max to the recursive select_edge_for_simplex(), which will return
    ///   an: Option<candinate_node>
    /// * If the candidate_node is set, look up it's slack and return both.
    ///
    /// Documentation from paper:
    /// * enter_edge ﬁnds a non-tree edge to replace e.
    ///   * This is done by breaking the edge e, which divides the tree into a head and tail component.
    ///   * All edges going from the head component to the tail are considered, with an edge of minimum slack being chosen.
    ///   * This is necessary to maintain feasibility.
    fn enter_edge_for_simplex(&self, tree_edge_idx: usize) -> Option<(usize, i32)> {
        let edge = self.get_edge(tree_edge_idx);
        let (disposition, search_node_idx, min, max) = {
            let src_node = self.get_node(edge.src_node);
            let dst_node = self.get_node(edge.dst_node);
            let src_node_max = src_node.tree_dist_max().expect("must have subtree_max");
            let dst_node_max = dst_node.tree_dist_max().expect("must have subtree_max");

            if src_node_max < dst_node_max {
                let src_node_min = src_node.tree_dist_min().expect("must have subtree_min");

                (In, edge.src_node, src_node_min, src_node_max)
            } else {
                let dst_node_min = dst_node.tree_dist_min().expect("must have subtree_min");

                (Out, edge.dst_node, dst_node_min, dst_node_max)
            }
        };

        // println!(
        //     "Going to select: {disposition}: search_node {min}-{max}: {}",
        //     self.node_to_string(search_node_idx)
        // );

        self.select_edge_for_simplex(disposition, search_node_idx, min, max, None)
            .map(|selected_edge| {
                // println!("Final selection: {}", self.edge_to_string(selected_edge));
                (
                    selected_edge,
                    self.simplex_slack(selected_edge)
                        .expect("selected edge must have slack"),
                )
            })
    }

    /// Recursively find the first edge that satifies the edge selection critiera for network simplex.
    ///
    /// Selection criteria:
    /// * Edge must not currently be in the spanning tree.
    /// * If disposition is In:
    ///   * The src_node (tail) of the edge must have a sub_tree_max >= min and <= max
    /// * If disposition is Out:
    ///   * The dst_node (tail) of the edge must have a sub_tree_max >= min and <= max
    fn select_edge_for_simplex(
        &self,
        disposition: EdgeDisposition,
        search_node_idx: usize,
        min: usize,
        max: usize,
        candidate_idx: Option<usize>,
    ) -> Option<usize> {
        let mut candidate_idx = candidate_idx;
        let search_node = self.get_node(search_node_idx);
        let search_node_tree_dist_max = search_node
            .tree_dist_max()
            .expect("Search node must have subtree");

        // println!(
        //     "   select_edge_for_simplex: search_node: {}",
        //     self.node_to_string(search_node_idx)
        // );
        for edge_idx in search_node.get_edges(disposition).iter().cloned() {
            let edge = self.get_edge(edge_idx);
            let node_idx = match disposition {
                In => edge.src_node,
                Out => edge.dst_node,
            };
            let node_tree_dist_max = self
                .get_node(node_idx)
                .tree_dist_max()
                .expect("Candidate node must have a subtree");
            // println!(
            //     "   select_edge_for_simplex: checking edge {disposition}: {}",
            //     self.edge_to_string(edge_idx)
            // );

            if !edge.in_spanning_tree() {
                let slack = self.simplex_slack(edge_idx).expect("Edge must have slack");
                let candidate_slack = candidate_idx
                    .map(|edge_idx| self.simplex_slack(edge_idx).expect("edge must have slack"));

                if !self.node_distance_within_limits(node_idx, min, max)
                    && (candidate_slack.is_none() || Some(slack) < candidate_slack)
                {
                    candidate_idx = Some(edge_idx);
                    // println!("   selected: {edge_idx}");
                } else {
                    // println!(
                    //     "   node not withing distance limits {min}-{max}: {}",
                    //     self.node_to_string(node_idx)
                    // )
                }
            } else if node_tree_dist_max < search_node_tree_dist_max {
                candidate_idx =
                    self.select_edge_for_simplex(disposition, node_idx, min, max, candidate_idx);
            } else {
                // println!("   select_edge_for_simplex: rejected edge {edge_idx}");
            }
        }
        // println!("   select_edge_for_simplex: next phase");
        for edge_idx in search_node
            .get_edges(disposition.opposite())
            .iter()
            .cloned()
        {
            // println!(
            //     "   select_edge_for_simplex: checking edge {}: {}",
            //     disposition.opposite(),
            //     self.edge_to_string(edge_idx)
            // );
            let edge = self.get_edge(edge_idx);
            let node_idx = match disposition.opposite() {
                In => edge.src_node,
                Out => edge.dst_node,
            };
            let node_tree_dist_max = self
                .get_node(node_idx)
                .tree_dist_max()
                .expect("Candidate node must have a subtree");

            if node_tree_dist_max < search_node_tree_dist_max {
                candidate_idx =
                    self.select_edge_for_simplex(disposition, node_idx, min, max, candidate_idx);
            } else {
                // println!("   select_edge_for_simplex: rejected edge {edge_idx}: {node_tree_dist_max} < {search_node_tree_dist_max}");
            }
        }

        candidate_idx
    }

    /// Documentation from paper:
    /// * The edges are exchanged, updating the tree and cut values.
    fn exchange_edges_in_spanning_tree(
        &mut self,
        neg_cut_edge_idx: usize,
        non_tree_edge_idx: usize,
    ) {
        self.get_edge(neg_cut_edge_idx).set_in_spanning_tree(false);
        self.get_edge(non_tree_edge_idx).set_in_spanning_tree(true);
    }

    /// Balance nodes either top to bottom or left to right
    fn balance_for_simplex(&mut self, target: SimplexNodeTarget) {
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

    /// Balance nodes left-right so that nodes in different ranks are balanced against each other.
    /// 
    /// So:  A
    ///      | \
    ///      B  C
    ///
    /// Becomes:   A
    ///           /  \
    ///          B    C
    /// 
    /// * In this case, A is "balanced" or centered against B and C
    /// * balance_left_right() is only called using the "transformed" aux_graph that was modified with
    ///   network simplex for setting horizontal positions (target == XCoordinate).
    ///   
    /// Summary:
    /// * for each tree_edge with a cutvalue of zero:
    ///   * find a non_tree_edge with the lowest remaining cutvalue.
    ///     * if non_tree_edge has a slack > 1:
    ///       * select the head or tail node of to tree_edge whos has a descendent farther from the root (deepest descendent)
    ///         * Rerank the selected node and all tree descendent nodes by:
    ///           * if you selected the tail: -non_tree_edge_slack / 2
    ///           * if you selected the head: non_tree_edge_slack / 2
    /// 
    /// * Note that this only steps through all tree edges once.
    /// * However, the reranking of nodes can affect subsequent rerankings
    /// 
    /// How it works:
    /// * As of now, I don't fully understand how it works.
    /// * First, you are using a transformed graph:
    ///   * The actual edges are replaced with two edges and a virtual node:
    ///     * A -> B becomes: A <- V -> B
    ///     * One reason to do this might be that:
    ///       * Instead of having one arrow that only points from one node to another,
    ///         you have one arrow that points from V -> A, and one node that points from A -> V
    ///         * This has a significant affect on the cut value, of the edges between N1 and N2,
    ///           since the direction of the arrow to be cut is include in the calculation.
    ///           So for:  N1 <-e1- V -e2-> N2, cutvalue(e1) == cutvalue(e2) +/- 2
    ///   * Additionally, adjacent nodes in the same rank are connected by an edge
    ///     which is there to keep the nodes from moving relative to each other 
    ///     (the original paper says it is to "force the nodes to be sufficiently
    ///     separate from one another but does not affect the cost of layout").
    /// * By selecting a cutvalue of zero, you are only considering edges that 
    /// 
    /// From the paper: (page 20 section 4.2: Optimal node placement)
    /// * We can now consider the level assignment problem on G′, which can be solved using the network simplex method.
    ///   Any solution of the positioning problem on G corresponds to a solution of the level assignment problem on G′ with the same cost.
    ///   This is achieved by assigning each n e the value uv min (x_u, x_v), using the notation of ﬁgure 4-2 and where x_u  and x_v  are
    ///   the X coordinates assigned to u and v in G.  Conversely, any level assignment in G′ induces a valid positioning in G.
    ///   In addition, in an optimal level assignment, one of e u or e must have length 0, and the other has length | x_u − x_v |. This
    ///   means the cost of an original edge (u, v) in G equals the sum of the cost of the two edges e_u, e_v in G′ and, globally, the
    ///   two solutions have the same cost, Thus, optimality of G′ implies optimality for G and solving G′ gives us a solution for G.
    ///
    fn balance_left_right(&mut self) {
        // self.print_nodes("balance_left_right start");

        for (tree_edge_idx, tree_edge) in self.tree_edge_iter() {
            if tree_edge.cut_value == Some(0) {
                // print!("Looking at edge with cut of 0:"); self.print_edge(tree_edge_idx);

                if let Some((_non_tree_edge_idx, non_tree_edge_slack)) = self.enter_edge_for_simplex(tree_edge_idx)
                {
                    if non_tree_edge_slack > 1 {
                        // println!("  balance_left_right(): replace {tree_edge_idx} with {non_tree_edge_idx}, slack: {non_tree_edge_slack}\n    replace: {}\n      with: {}",
                        //         self.edge_to_string(tree_edge_idx),
                        //         self.edge_to_string(non_tree_edge_idx),
                        //     );

                        let src_node = self.get_node(tree_edge.src_node);
                        let dst_node = self.get_node(tree_edge.dst_node);

                        if let (Some(src_dist_max), Some(dst_dist_max)) =
                            (src_node.tree_dist_max(), dst_node.tree_dist_max())
                        {
                            let (rerank_node_idx, delta) = if src_dist_max < dst_dist_max {
                                (tree_edge.src_node, non_tree_edge_slack / 2)
                            } else {
                                (tree_edge.dst_node, -non_tree_edge_slack / 2)
                            };
                            // println!("  rerank by node {rerank_node_idx} by: {delta}");
                            self.rerank_by_tree(rerank_node_idx, delta);
                        } else {
                            panic!("Not all nodes in spanning tree!");
                        }
                    } else {
                        // println!(
                        //     "  balance_left_right(): skipping candidate replacement edge with slack <= 1: slack={non_tree_edge_slack}: {}",
                        //     self.edge_to_string(_non_tree_edge_idx)
                        // );
                    }
                } else {
                    println!("  No edge to enter!");
                }
            } else {
                // println!(
                //     "balance_left_right(): skipping candidate removal edge with cutval != Some(0): {}",
                //     self.edge_to_string(tree_edge_idx)
                // );
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
    fn test_leave_edge_for_simplex() {
        let (mut graph, _expected_cutvals) = Graph::configure_example_2_3_a_extended();
        graph.init_spanning_tree_and_cutvalues();

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

    /// enter_edge_for_simplex() is supposed to find the next edge with the minimum cut value.
    ///
    /// Given a specific example, we know which edges we expect it to return.
    ///
    /// Ignored because the new_enter_edge_for_simplex unit test needs to be re-imagined.  Looks like
    /// the min/max/ranges stuff is not being inited properly in the test.
    #[ignore]
    #[test]
    fn test_enter_edge_for_simplex() {
        // let (mut graph, _expected_cutvals) = Graph::configure_example_2_3_a_extended();
        // graph.init_cutvalues();
        let mut graph = Graph::example_graph_from_paper_2_3();
        // graph.make_asyclic();
        // graph.merge_edges();
        graph.set_feasible_tree_for_simplex(true);

        assert_eq!(
            graph.tree_node_count(),
            graph.node_count(),
            "all nodes should be tree nodes"
        );

        // The non-tree edges we expect to be picked by leave_edge_for_simplex().
        //
        // leave_edge_for_simplex() chooses the lowest negative cutvalue remaining
        // in the tree.
        //
        // Somewhat a guess based on the approach because several candidates can have
        // the same cut value, so this test is a bit brittle.
        // But we know they can only be tree edges.

        let (_, neg_edge1_idx) = graph.get_named_edge("a", "e");
        let (_, neg_edge2_idx) = graph.get_named_edge("a", "f");

        //let (_, neg_edge1_idx) = graph.get_named_edge("l", "h");
        //let (_, neg_edge2_idx) = graph.get_named_edge("g", "h");

        println!("Graph (ne1: {neg_edge1_idx}, ne2: {neg_edge2_idx}):\n{graph}");

        let neg_edge = graph.leave_edge_for_simplex(0).unwrap();
        println!("NEG EDGE1: {neg_edge}");

        let (selected_edge_idx, selected_slack) = graph.enter_edge_for_simplex(neg_edge).unwrap();
        graph.rerank_for_simplex(neg_edge, selected_slack);
        graph.adjust_cutvalues_and_exchange_for_simplex(neg_edge, selected_edge_idx);

        assert_eq!(selected_edge_idx, neg_edge1_idx);

        let neg_edge2 = graph.leave_edge_for_simplex(neg_edge + 1).unwrap();
        println!("NEG EDGE2: {neg_edge2}");

        // let new_edge = graph.get_edge(new_edge);
        // let src_node = graph.get_node(new_edge.src_node);
        // let dst_node = graph.get_node(new_edge.dst_node);
        // println!("{graph}");
        // println!("Next edge 1: {} -> {}", src_node, dst_node);

        let (new_edge2, _) = graph.enter_edge_for_simplex(neg_edge2).unwrap();
        assert_eq!(new_edge2, neg_edge2_idx);

        // let new_edge2 = graph.get_edge(new_edge2);
        // let src_node = graph.get_node(new_edge2.src_node);
        // let dst_node = graph.get_node(new_edge2.dst_node);
        // println!("Next edge 2: {} -> {}", src_node, dst_node);
    }

    /// Only tests in that network_simplex_ranking() does not crash or
    /// stack overflow.
    #[test]
    fn test_network_simplex_ranking() {
        let mut graph = Graph::example_graph_from_paper_2_3();

        graph.network_simplex_ranking(SimplexNodeTarget::VerticalRank);
    }
}
