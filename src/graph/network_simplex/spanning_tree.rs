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

#[allow(unused)]
impl Graph {
    /// Initializes the spanning tree data.
    ///
    /// This includes:
    /// * Setting the edge_idx_to_parent value
    /// * setting min and max ranges
    ///
    /// Must be done before calling network simplex.
    //
    /// In the graphviz 9.0 code, this function is: dfs_range_init()
    /// It assumes that graph is fully connected.
    pub(super) fn init_spanning_tree(&mut self) {
        // let last_node = self.get_node(last_node_idx);
        // let start_node = if last_node.is_virtual() { last_node_idx } else {0};

        if self.node_count() != 0 {
            // XXX This is kind of crazy.  The order in which the tree is build
            //     effects how it is laid out, apparently.  In GraphViz, new virtual nodes
            //     are added at the beginning of the list.  In dot-rs, they are necessarily
            //     added at the end because we don't want the indexes to change.
            //     So we want to pick the last virtual node...
            let last_node_idx = self.node_count() - 1;

            self.set_tree_parents_and_ranges(true, last_node_idx, None, 1);
        }
    }

    /// In the graphviz 9.0 code, there are two functions:
    /// * dfs_range_init()
    /// * dfs_range()
    /// They are identical code, expect for the exit condition.  set_tree_parents_and_ranges()
    /// does either depending on the "initializing flag"
    pub(super) fn set_tree_parents_and_ranges(
        &mut self,
        initializing: bool,
        node_idx: usize,
        parent_edge_idx: Option<usize>,
        min: usize,
    ) -> usize {
        let node = self.get_node(node_idx);

        let par_str = if let Some(parent_edge_idx) = parent_edge_idx {
            self.edge_to_string(parent_edge_idx)
        } else {
            "None".to_string()
        };
        println!(
            "set_tree_parents_and_ranges('{}', {parent_edge_idx:?}, {min})\n  parent_edge:{par_str}",
            node.name
        );

        if !initializing {
            if let Some(tree_data) = node.spanning_tree() {
                if tree_data.edge_idx_to_parent() == parent_edge_idx
                    && tree_data.tree_dist_min() == Some(min)
                {
                    return tree_data.tree_dist_max().unwrap_or(0) + 1;
                }
            } else {
                panic!("Setting tree ranges on a node that is not in the tree!");
            }
        }

        let cur_max = node.tree_dist_max();
        self.get_node_mut(node_idx)
            .set_tree_data(parent_edge_idx, Some(min), cur_max);

        let mut max = min;

        // println!("Setting range for: {node_idx}: {:?}", self.non_parent_tree_nodes(node_idx));

        for (node_idx, edge_idx) in self.non_parent_tree_nodes(node_idx) {
            max = max.max(self.set_tree_parents_and_ranges(
                initializing,
                node_idx,
                Some(edge_idx),
                max,
            ));
        }

        self.get_node_mut(node_idx).set_tree_dist_max(Some(max));

        max + 1
    }

    ///
    /// GraphViz comment for invalidate_path():
    ///   Invalidate DFS attributes by walking up the tree from to_node till lca
    ///   (inclusively). Called when updating tree to improve pruning in dfs_range().
    ///   Assigns ND_low(n) = -1 for the affected nodes.
    ///
    pub(super) fn invalidate_path(&self, lca_node_idx: usize, to_node_idx: usize) {
        let mut to_node_idx = to_node_idx;

        println!(
            "Invalidate path for node {lca_node_idx}: {}",
            self.node_to_string(lca_node_idx)
        );
        println!(
            "  to note {to_node_idx}: {}",
            self.node_to_string(to_node_idx)
        );
        loop {
            let to_node = self.get_node(to_node_idx);

            if to_node.tree_dist_min().is_none() {
                break;
            }

            to_node.set_tree_dist_min(None);

            if let Some(parent_edge_idx) = to_node.spanning_tree_parent_edge_idx() {
                let lca_node = self.get_node(lca_node_idx);

                if to_node.tree_dist_max() >= lca_node.tree_dist_max() {
                    if to_node_idx != lca_node_idx {
                        panic!("invalidate_path: skipped over LCA");
                    }
                    break;
                }

                let parent_edge = self.get_edge(parent_edge_idx);
                let parent_src = self.get_node(parent_edge.src_node);
                let parent_dst = self.get_node(parent_edge.dst_node);

                to_node_idx = if parent_src.tree_dist_max() > parent_dst.tree_dist_max() {
                    parent_edge.src_node
                } else {
                    parent_edge.dst_node
                };
            } else {
                break;
            }
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
    pub(super) fn set_feasible_tree_for_simplex(&mut self, init_rank: bool) {
        if init_rank {
            self.init_simplex_rank();
        }

        // Nodecount is at least as big as necessary: there will very likely be
        // fewer subtrees than nodes, but not more.
        let mut min_heap = MinHeap::new(self.node_count());

        // Whether you are in the spanning tree or not, create as subtree
        // for all nodes.
        self.print_nodes("before find_tight_subtree() in set_feasible_tree()");

        // We use node_idx_virtual_first() to mirror the order that GraphViz uses.
        for node_idx in self.node_indexes_in_graphviz_order() {
            // Don't place this exclusion in a filter, as it can change from the preivous call.
            if !self.get_node(node_idx).has_sub_tree() {
                let sub_tree = self.find_tight_subtree(node_idx);
                min_heap.insert_unordered_item(sub_tree);
            }
        }

        min_heap.order_heap();

        self.print_nodes(&format!(
            "after find_tight_subtree(): heap_size:{} in set_feasible_tree",
            min_heap.len()
        ));

        while min_heap.len() > 1 {
            let sub_tree = min_heap.pop().expect("can't be empty");
            println!(
                "   finding edge for: {:?} with heap of {}",
                sub_tree,
                min_heap.len()
            );
            let edge_idx = self
                .find_tightest_incident_edge(sub_tree)
                .expect("cant find inter tree edge");

            println!("   Merging with edge: {edge_idx}");
            let modified_sub_tree_idx = self.merge_sub_trees(edge_idx);
            println!("   Reording: {modified_sub_tree_idx}");
            min_heap.reorder_item(modified_sub_tree_idx);
            println!("   reorder done");
        }
        self.init_spanning_tree_and_cutvalues();
        self.print_nodes(&format!(
            "after init_spanning_tree_and_cutvalues() in set_feasible_tree:{}",
            min_heap.len()
        ));
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

        println!(
            "Merging sub_trees for {edge_idx}: {}",
            self.edge_to_string(edge_idx)
        );

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
        let rank = node.simplex_rank().expect("all nodes should be ranked");
        let new_rank = rank + delta;

        node.set_simplex_rank(Some(new_rank));

        for (edge_idx, next_node_idx) in self.get_node_edges_and_adjacent_node(node_idx, true) {
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
        assert!(heap_idx1.is_some() || heap_idx2.is_some());

        // The merged tree will replace one of the roots
        // based on whether it is in the heap, and of both
        // are in the heap, the biggest one.
        let union_tree = if heap_idx2.is_none() || size1 >= size2 {
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

        println!("tigtest_incident_edge_search: {search_node_idx}");
        let search_sub_tree_root = self.find_node_sub_tree_root(search_node_idx);
        for (edge_idx, next_node_idx) in
            self.get_node_edges_and_adjacent_node(search_node_idx, true)
        {
            if self.get_edge(edge_idx).in_spanning_tree() {
                // Already in spanning tree...continue the search
                if Some(next_node_idx) == from_node_idx {
                    continue; // do not search back in the tree
                } else {
                    // search forward in tree
                    println!("search forward from {search_node_idx} to {next_node_idx}");
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
        println!("Finding root of: {node_idx}");
        self.get_node(node_idx)
            .sub_tree()
            .map(|sub_tree| sub_tree.find_root())
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

        println!(
            "find_tight_subtree() for node: {}",
            self.node_to_string(node_idx)
        );

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

        println!(
            "    tight_subtree_search() for {}",
            self.node_to_string(node_idx)
        );

        // set this node to be in the given sub_tree.
        self.get_node(node_idx).set_sub_tree(sub_tree.clone());

        for (edge_idx, adjacent_node_idx) in self.get_node_edges_and_adjacent_node(node_idx, false)
        {
            let edge = self.get_edge(edge_idx);
            // let adjacent_node = self.get_node(adjacent_node_idx);

            // Note that in the GraphViz code, they ignore nodes with sub_trees
            // instead of ignoring nodes in the spanning tree.  I believe this
            // is incorrect logic, and only works by luck because they have
            // overloaded ND_subtree.
            if !edge.in_spanning_tree()
                && !self.get_node(adjacent_node_idx).has_sub_tree()
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
        if !src_node.in_spanning_tree() {
            src_node.set_empty_tree_node();
        }
        if !dst_node.in_spanning_tree() {
            dst_node.set_empty_tree_node();
        }
    }

    /// Re-rank the given node by subtracting delta to the rank, and all descendent nodes in the tree.
    /// 
    /// * First reduces the rank of node_idx by delta
    /// * Then recursively reduces the rank of all tree descendents of node_idx
    ///   by delta.
    ///   
    /// More intuitively written, this function should add delta rather
    /// than subtract it, but I have left it as a subtract to keep it similar
    /// to what GraphVis does.
    pub(super) fn rerank_by_tree(&self, node_idx: usize, delta: i32) {
        let node = self.get_node(node_idx);

        if let Some(cur_rank) = node.simplex_rank() {
            let new_rank = cur_rank - delta;
            println!(
                "Reranking {} by {delta} from {cur_rank} to {new_rank}",
                node.name
            );
            node.set_simplex_rank(Some(new_rank));

            let children = self
                .non_parent_tree_nodes(node_idx)
                .iter()
                .map(|(node_idx, _)| self.get_node(*node_idx).name.clone())
                .collect::<Vec<String>>();
            println!("   {} reranking to: {:?}", node.name, children);

            for (node_idx, _) in self.non_parent_tree_nodes(node_idx) {
                self.rerank_by_tree(node_idx, delta)
            }
        } else {
            panic!("Rank not set for node {} in rerank.", node.name);
        }
    }

    /// Return a vector of (node_idx, edge_idx) which are tree members and do not point to the
    /// parent node of the given node.
    ///
    /// Only returns non-ignored nodes.
    fn non_parent_tree_nodes(&self, node_idx: usize) -> Vec<(usize, usize)> {
        let mut nodes = self.directional_non_parent_tree_nodes(node_idx, Out);
        let mut in_nodes = self.directional_non_parent_tree_nodes(node_idx, In);

        nodes.append(&mut in_nodes);
        nodes
    }

    /// Return a vector of (node_idx, edge_idx) which are tree members and do not point to the
    /// parent node of the given node from either in_edges or out_edges depending on disposition.
    ///
    /// Only returns non-ignored nodes.
    ///
    /// TODO: Rewrite this to call: node_tree_edges(node_idx, disposition)
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
        use itertools::Itertools;

        edges
            .iter()
            .filter_map(|edge_idx| {
                let edge = self.get_edge(*edge_idx);
                let other_node_idx = match disposition {
                    In => edge.src_node,
                    Out => edge.dst_node,
                };

                if !edge.ignored
                    && edge.in_spanning_tree()
                    && node.spanning_tree_parent_edge_idx() != Some(*edge_idx)
                {
                    Some((other_node_idx, *edge_idx))
                } else {
                    // if edge.ignored {
                    //     println!("Edge {edge_idx}: ignored ");
                    // }
                    // if !edge.in_spanning_tree() {
                    //     println!("Edge {edge_idx}: not in spanning tree");
                    // }
                    // if node.spanning_tree_parent_edge_idx() == Some(*edge_idx) {
                    //     println!("Edge {edge_idx}: points to parent");
                    // }
                    None
                }
            })
            .collect()
    }

    /// Return all of a node's tree edges that are of the given disposition.
    pub(super) fn node_tree_edges(
        &self,
        node_idx: usize,
        disposition: EdgeDisposition,
    ) -> Vec<&Edge> {
        let node = self.get_node(node_idx);
        let edges = match disposition {
            In => &node.in_edges,
            Out => &node.out_edges,
        };

        edges
            .iter()
            .filter_map(|edge_idx| {
                let edge = self.get_edge(*edge_idx);

                if !edge.ignored && edge.in_spanning_tree() {
                    Some(edge)
                } else {
                    None
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_directional_non_parent_tree_nodes() {
        let mut graph = Graph::from("digraph { a -> b; b -> a; c -> d; c -> a; };");
        let a_idx = graph.name_to_node_idx("a").unwrap();
        let b_idx = graph.name_to_node_idx("b").unwrap();
        let c_idx = graph.name_to_node_idx("c").unwrap();
        let _d_idx = graph.name_to_node_idx("d").unwrap();

        graph.make_asyclic();
        graph.merge_edges();
        graph.init_simplex_rank();
        graph.set_feasible_tree_for_simplex(true);

        let a_in_nodes = graph.directional_non_parent_tree_nodes(a_idx, In);
        let a_out_nodes = graph.directional_non_parent_tree_nodes(a_idx, Out);

        assert_eq!(a_out_nodes, [(b_idx, 0)]);
        assert_eq!(a_in_nodes, []);

        let c_in_nodes = graph.directional_non_parent_tree_nodes(c_idx, In);
        let c_out_nodes = graph.directional_non_parent_tree_nodes(c_idx, Out);

        // It does not include (a_idx, 3), because that is a parent node.
        assert_eq!(c_out_nodes, [(a_idx, 3)]);
        assert_eq!(c_in_nodes, []);
    }

    /// This test is here to ensure init_spanning_tree() does not go into an
    /// infinite loop because it adds edges that should be ignored (edges that
    /// point to a node already in the tree).
    #[test]
    fn test_init_spanning_tree_ingore_tree_nodes() {
        let mut graph = Graph::from("digraph { a -> b; a -> c; b -> d; c -> d; }");
        graph.make_asyclic();
        graph.merge_edges();
        graph.init_simplex_rank();

        // at end end calls: init_cutvals() -> graph.init_spanning_tree();
        graph.set_feasible_tree_for_simplex(true);
    }

    /// Not a useful test yet...XXX
    #[test]
    fn test_set_feasible_tree_for_simplex() {
        // let mut graph = Graph::from("digraph { a -> b; b -> a; c -> d; c -> a; }");
        let mut graph = Graph::example_graph_from_paper_2_3_extended();

        graph.make_asyclic();
        graph.merge_edges();
        graph.init_simplex_rank();
        graph.set_feasible_tree_for_simplex(true);
        // at end end calls: init_cutvals() -> graph.init_spanning_tree();

        println!("{graph}");
    }
}
