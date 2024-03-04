//! Methods for the network simplex algorithm that assist with generating
//! and using the spanning tree and feasible tree.
//!
//! The feasible tree is a spanning tree built all the nodes and some of
//! the edges of a graph.  The feasible tree is used in the network simplex
//! algorithm to reduce computation time.
//!
//! From the paper (page 7: 2.3 Network Simplex):
//!
//! We begin with a few deﬁnitions and observations. A feasible ranking is
//! one satisfying the length constraints l(e) ≥ δ(e) for all e.  Given
//! any ranking, not necessarily feasible, the slack of an edge is the
//! difference of its length and its minimum length. Thus, a ranking is
//! feasible if the slack of every edge is non-negative.  An edge is tight
//! if its slack is zero.
//!
//! A spanning tree of a graph induces a ranking, or rather, a family of
//! equivalent rankings.  (Note that the spanning tree is on the underlying
//! unrooted undirected graph, and is not necessarily a directed tree.)
//! This ranking is generated by picking an initial node and assigning
//! it a rank. Then, for each node adjacent in the spanning tree to a
//! ranked node, assign it the rank of the adjacent node, incremented
//! or decremented by the minimum length of the connecting edge,
//! depending on whether it is the head or tail of the connecting edge.
//! This process is continued until all nodes are ranked. A spanning tree
//! is feasible if it induces a feasible ranking. By construction,
//! all edges in the feasible tree are tight.
//!
//! <https://en.wikipedia.org/wiki/Network_simplex_algorithm>

use super::{
    heap::{HeapIndex, MinHeap},
    sub_tree::SubTree,
    Graph,
};
use crate::graph::edge::{
    Edge,
    EdgeDisposition::{self, In, Out},
};

impl Graph {
    /// Initializes the spanning tree data
    /// In the graphviz 9.0 code, this function is: dfs_range_init()
    fn init_spanning_tree(&mut self, node_idx: usize) {
        self.set_tree_ranges(true, node_idx, None, 1);
    }

    /// In the graphviz 9.0 code, this function is: dfs_range_init()
    fn update_tree_ranges(&mut self, lca_node_idx: usize, min: usize) {
        let parent_edge_idx = self.get_node(lca_node_idx).spanning_tree_parent_edge_idx();

        self.set_tree_ranges(false, lca_node_idx, parent_edge_idx, min);
    }

    /// In the graphviz 9.0 code, there are two functions:
    /// * dfs_range_init()
    /// * dfs_range()
    /// They are identical code, expect for the exit condition.  set_tree_ranges() =
    /// does both
    fn set_tree_ranges(
        &mut self,
        initializing: bool,
        node_idx: usize,
        parent_edge_idx: Option<usize>,
        min: usize,
    ) -> usize {
        let node = self.get_node(node_idx);

        if !initializing {
            if let Some(tree_data) = node.spanning_tree() {
                if tree_data.edge_idx_to_parent() == parent_edge_idx
                    && tree_data.sub_tree_idx_min() == min
                {
                    return tree_data.sub_tree_idx_max() + 1;
                }
            } else {
                panic!("Setting tree ranges on a node that is not in the tree!");
            }
        }

        let mut max = min;
        for (edge_idx, node_idx) in self.non_parent_tree_nodes(node_idx) {
            max = max.max(self.set_tree_ranges(initializing, node_idx, Some(edge_idx), max));
        }

        self.get_node_mut(node_idx)
            .set_tree_data(parent_edge_idx, min, max);

        max + 1
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
    pub(super) fn set_feasible_tree_for_simplex(&mut self) {
        self.init_simplex_rank();

        // Nodecount is at least as big as necessary: there will very likely be
        // fewer subtrees than nodes, but not more.
        let mut min_heap = MinHeap::new(self.node_count());

        for node_idx in 0..self.node_count() {
            if !self.get_node(node_idx).in_spanning_tree() {
                let sub_tree = self.find_tight_subtree(node_idx);

                min_heap.insert_unordered_item(sub_tree);
            }
        }
        min_heap.order_heap();

        while min_heap.len() > 1 {
            let sub_tree = min_heap.pop().expect("can't be empty");
            let edge_idx = self
                .find_tightest_incident_edge(sub_tree)
                .expect("cant find inter tree edge");

            let modified_sub_tree_idx = self.merge_sub_trees(edge_idx);
            min_heap.reorder_item(modified_sub_tree_idx);
        }

        self.init_cutvalues();
    }

    /// Given and edge, merge the two subtrees pointed to by each edge's nodes.
    ///
    /// Expects that:
    /// * The edge currently is not in the spanning tree, but will be added to it
    ///   by this method.
    /// * Both nodes are in sub_tree's which are different
    /// * Only one of these nodes has a heap_index (is currently
    ///   in the heap)
    ///   
    /// The sub_tree which is still in the heap will be over-written with the merged
    /// sub_tre information.
    fn merge_sub_trees(&self, edge_idx: usize) -> usize {
        let edge = self.get_edge(edge_idx);
        let edge_slack = self
            .simplex_slack(edge_idx)
            .expect("could not calculate edge slack");

        assert!(!edge.in_spanning_tree());

        let src_tree_root = self
            .find_node_sub_tree_root(edge.src_node)
            .expect("could not find src tree root");
        let dst_tree_root = self
            .find_node_sub_tree_root(edge.dst_node)
            .expect("count not find dst tree root");

        if src_tree_root.heap_idx().is_none() {
            let delta = edge_slack;

            self.tree_adjust_rank(src_tree_root.start_node_idx(), None, delta)
        } else {
            let delta = -edge_slack;

            self.tree_adjust_rank(dst_tree_root.start_node_idx(), None, delta)
        }
        self.add_tree_edge(edge);

        self.sub_tree_union(src_tree_root, dst_tree_root)
    }

    /// Adjust the rank of all nodes in the spanning tree by delta.
    fn tree_adjust_rank(&self, node_idx: usize, from_idx: Option<usize>, delta: i32) {
        let node = self.get_node(node_idx);
        let rank = node.simplex_rank().expect("all nodes should be ranked") as i32;
        let new_rank = (rank + delta) as u32;

        // XXX Needs to be mutable!
        // node.set_simplex_rank(Some(new_rank));

        for (edge_idx, next_node_idx) in self.get_node_edges_and_adjacent_node(node_idx) {
            let edge = self.get_edge(edge_idx);

            if edge.in_spanning_tree() && from_idx != Some(next_node_idx) {
                self.tree_adjust_rank(next_node_idx, Some(node_idx), delta)
            }
        }
    }

    /// Return a sub_tree that is the union of the two given sub_trees.
    ///
    /// It will have:
    /// * Refer to the same node?
    /// * a size which is tree1.size + tree2.size
    /// * The same parent of the larger of tree1 or tree2.
    /// * It becomes the new parent of both tree1 and tree2.
    fn sub_tree_union(&self, sub_tree_root1: SubTree, sub_tree_root2: SubTree) -> usize {
        let heap_idx1 = sub_tree_root1.heap_idx();
        let heap_idx2 = sub_tree_root2.heap_idx();
        let size1 = sub_tree_root1.size();
        let size2 = sub_tree_root2.size();

        // At least one of the roots must be in the heap
        assert!(!heap_idx1.is_none() || !heap_idx2.is_none());

        // The merged tree will replace one of the roots
        // based on whether it is in the heap, and of both
        // are in the heap, the biggest one.
        let selection = if heap_idx1.is_none() {
            2
        } else if heap_idx2.is_none() || size1 > size2 {
            1
        } else {
            2
        };

        // The selected root is still None (it was a root)
        // But the unselected root's parent becomes the selected root.
        let union_tree = if selection == 1 {
            sub_tree_root2.set_parent(Some(sub_tree_root1.clone()));
            sub_tree_root1.set_parent(None); // But should already be none...

            sub_tree_root1
        } else {
            sub_tree_root1.set_parent(Some(sub_tree_root2.clone()));
            sub_tree_root2.set_parent(None); // But should already be none...

            sub_tree_root2
        };

        union_tree.set_size(size1 + size2);

        union_tree.heap_idx().unwrap()
    }

    // Return the tightest edge to another tree incident on the given tree.
    // 
    // In GraphVis code: inter_tree_edge()
    fn find_tightest_incident_edge(&self, sub_tree: SubTree) -> Option<usize> {
        self.tightest_incident_edge_search(sub_tree.start_node_idx(), None, None)
    }

    // Find the tightest edge to another tree incident on the given tree.
    // * search_node_idx is the node to search
    // * from_node_idx is the node that was previously searched (via a connected edge)
    // * best_edge_idx is the best edge found so far
    //
    // In GraphViz code: inter_tree_edge_search()
    fn tightest_incident_edge_search(
        &self,
        search_node_idx: usize,
        from_node_idx: Option<usize>,
        best_edge_idx: Option<usize>,
    ) -> Option<usize> {
        let mut best_edge_idx = best_edge_idx;
        let mut best_cur_slack =
            best_edge_idx.and_then(|best_edge_idx| self.simplex_slack(best_edge_idx));
        // We can't do better than slack of zero: that's the tightest edge possible
        if best_cur_slack == Some(0) {
            return best_edge_idx;
        }

        let search_sub_tree_root = self.find_node_sub_tree_root(search_node_idx);
        for (edge_idx, next_node_idx) in self.get_node_edges_and_adjacent_node(search_node_idx) {
            if self.get_edge(edge_idx).in_spanning_tree() {
                // Already in spanning tree...continue the search
                if Some(next_node_idx) == from_node_idx {
                    continue; // do not search back in the tree
                } else {
                    // search forward in tree
                    best_edge_idx = self.tightest_incident_edge_search(
                        next_node_idx,
                        Some(search_node_idx),
                        best_edge_idx,
                    );
                }
            } else if self.find_node_sub_tree_root(next_node_idx) != search_sub_tree_root {
                // A candidate edge...
                let edge_slack = self.simplex_slack(edge_idx);

                if edge_slack.is_some() && (best_cur_slack.is_none() || edge_slack < best_cur_slack)
                {
                    best_edge_idx = Some(edge_idx);
                    best_cur_slack = edge_slack;
                }
            }
        }

        best_edge_idx
    }

    /// Given a node_idx, find it's root sub_tree.
    fn find_node_sub_tree_root(&self, node_idx: usize) -> Option<SubTree> {
        self.get_node(node_idx).sub_tree().map(|sub_tree| sub_tree.find_root())
    }

    /// Finds a tight subtree starting from node node_idx.
    /// * All edges and nodes of the subtree are marked as tree edges.
    ///
    /// So given a node, finds all nodes connected to that node that increase in rank by one,
    /// and then does this recursively on all the new nodes.
    ///
    /// Starting with a root node, a "tight subtree" is a list of nodes whereby each included
    /// edge on that node points to a node that has rank = cur_rank+1 (thus some edges will
    /// not be included if they point to a node of lesser rank or rank > rank + 1).  All nodes
    /// in the subtree follow this rule recursively.
    ///
    /// It only makes sense to do this search after all nodes have a rank, because a "tight" edge
    /// is (typically) an edge that goes from a node of rank to a node of rank+1.  Finding a node
    /// without a rank is an error.
    ///
    /// Note that only edges with a slack of zero are considered, meaning that they are "tight"
    /// edges, which is why this called "tight_subtree_search".  A slack of zero between to
    /// edges e1 and e2 implies that:
    /// * The rank of e1 > e2
    /// * rank(e1) - rank(e2) = min_rank_diff, where min_rank_diff is typically one.
    ///   * so the rank of rank(e1) == rank(e2) - min_rank_diff
    ///   
    /// In GraphViz code: find_tight_subtree()
    fn find_tight_subtree(&self, node_idx: usize) -> SubTree {
        let tree = SubTree::new(node_idx);

        // Update the tree size to the combined size of all the nodes we found.
        tree.set_size(self.tight_subtree_search(node_idx, tree.clone()));

        tree
    }

    /// Finds a tight subtree starting from node_idx, and fill in the values of sub_tree.
    ///
    /// Return the size of the sub_tree.
    ///
    /// Side effect:
    /// * Edges are added to the subtree as the search progresses and are marked as tree edges.
    /// * Both nodes on each new tree edge are set as tree nodes.
    ///
    /// In GraphViz code: tight_subtree_search()
    fn tight_subtree_search(&self, node_idx: usize, sub_tree: SubTree) -> u32 {
        let mut subtree_size = 1;

        // set this node to be in the given sub_tree.
        self.get_node(node_idx).set_sub_tree(sub_tree.clone());

        for (edge_idx, adjacent_node_idx) in self.get_node_edges_and_adjacent_node(node_idx) {
            let edge = self.get_edge(edge_idx);

            if !edge.in_spanning_tree()
                && self.get_node(adjacent_node_idx).sub_tree().is_none()
                && self.simplex_slack(edge_idx) == Some(0)
            {
                self.add_tree_edge(edge);
                subtree_size += self.tight_subtree_search(adjacent_node_idx, sub_tree.clone())
            }
        }

        subtree_size
    }

    /// Set edge to be a tree node, as well as the nodes it is connected to.
    ///
    /// In GraphViz code: add_tree_edge()
    fn add_tree_edge(&self, edge: &Edge) {
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);

        edge.set_in_spanning_tree(true);
        src_node.set_empty_tree_node();
        dst_node.set_empty_tree_node();
    }

    pub(super) fn old_set_feasible_tree_for_simplex(&mut self) {
        // init_simplex_rank() will set tree_node=false for all nodes.
        self.init_simplex_rank();

        // Lets not assume the tree fields are clear to begin with
        for edge in self.edges.iter_mut() {
            edge.set_in_spanning_tree(false)
        }

        let mut tree_node_cnt = 0;
        for node in self.nodes.iter_mut() {
            if node.no_out_edges() {
                node.set_tree_root_node();
                tree_node_cnt += 1;
            } else {
                node.clear_tree_data();
            }
        }

        while tree_node_cnt < self.node_count() {
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

            // increase the rank of everyt treenode by the slack
            for node in self.nodes.iter_mut().filter(|node| node.in_spanning_tree()) {
                let new_rank = node.simplex_rank.expect("Node does not have rank") as i32 + delta;
                node.simplex_rank = Some(new_rank as u32);
            }

            let node_idx = self
                .get_incident_node(edge_idx)
                .expect("Edge is not incident");
            self.get_node_mut(node_idx).set_tree_root_node();
            self.get_edge_mut(edge_idx).set_in_spanning_tree(true);
            tree_node_cnt += 1;
        }
    }

    /// Re-rank the given node by adding delta to the rank, and all sub_nodes in the tree.
    fn rerank_by_tree(&mut self, node_idx: usize, delta: i32) {
        let node = self.get_node(node_idx);

        if let Some(cur_rank) = node.simplex_rank {
            let new_rank = (cur_rank as i32 + delta) as u32;
            self.get_node_mut(node_idx).set_simplex_rank(Some(new_rank));

            for (_edge_idx, node_idx) in self.non_parent_tree_nodes(node_idx) {
                self.rerank_by_tree(node_idx, delta)
            }
        }
    }

    /// Return a vector of (edge_idx, node_idx) which are tree members and do not point to the
    /// parent node of the given node.
    fn non_parent_tree_nodes(&self, node_idx: usize) -> Vec<(usize, usize)> {
        let mut nodes = self.directional_non_parent_tree_nodes(node_idx, Out);
        let mut in_nodes = self.directional_non_parent_tree_nodes(node_idx, In);

        nodes.append(&mut in_nodes);
        nodes
    }

    /// Return a vector of (edge_idx, node_idx) which are tree members and do not point to the
    /// parent node of the given node from either in_edges or out_edges depending on disposition.
    fn directional_non_parent_tree_nodes(
        &self,
        node_idx: usize,
        disposition: EdgeDisposition,
    ) -> Vec<(usize, usize)> {
        let node = self.get_node(node_idx);
        let edges = match disposition {
            In => &node.in_edges,
            Out => &node.out_edges,
        };

        edges
            .iter()
            .filter_map(|edge_idx| {
                let edge = self.get_edge(*edge_idx);
                let other_node_idx = match disposition {
                    In => edge.src_node,
                    Out => edge.dst_node,
                };

                if edge.in_spanning_tree()
                    && node.spanning_tree_parent_edge_idx() != Some(*edge_idx)
                {
                    Some((*edge_idx, other_node_idx))
                } else {
                    None
                }
            })
            .collect()
    }
}
