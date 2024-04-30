//! Methods for the network simplex algorithm that assist with generating
//! and using the spanning tree and feasible tree.
//!
//! The feasible tree is a spanning tree built all the nodes and some of
//! the edges of a graph.  The feasible tree is used in the network simplex
//! algorithm to reduce computation time.
//!
//! From the paper (page 7: 2.3 Network Simplex):
//!
//! We begin with a few definitions and observations. A feasible ranking is
//! one satisfying the length constraints l(e) ≥ δ(e) for all e.  Given
//! any ranking, not necessarily feasible, the slack of an edge is the
//! difference of its length and its minimum length. Thus, a ranking is
//! feasible if the slack of every edge is non-negative.  An edge is tight
//! if its slack is zero.
//!
//! A spanning tree of a graph induces a ranking, or rather, a family of
//! equivalent rankings.  (Note that the spanning tree is on the underlying
//! un-rooted undirected graph, and is not necessarily a directed tree.)
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
    /// Return true if maybe_ancestor_idx is a common ancestor of child_idx.
    ///
    /// This is true only if the max distance from the child node to the root is within
    /// the min/max distance of the ancestor node.
    ///
    /// GraphViz code: SEQ(ND_low(v), ND_lim(w), ND_lim(v))
    ///              : SEQ(maybe_ancestor_node_idx.min, child_idx.max, maybe_ancestor_node_idx.max)
    pub(super) fn is_common_ancestor(
        &self,
        maybe_ancestor_node_idx: usize,
        child_idx: usize,
    ) -> bool {
        let maybe_ancestor_node = self.get_node(maybe_ancestor_node_idx);
        let descendent_min_traversal = maybe_ancestor_node
            .tree_descendent_min_traversal_number()
            .expect("tree distance must be set");
        let traversal_number = maybe_ancestor_node
            .tree_traversal_number()
            .expect("tree distance must be set");

        self.node_in_tail_component(child_idx, descendent_min_traversal, traversal_number)
    }

    /// Return true if node_idx is in the tail component the node that holds min and max.
    ///
    /// * Consider a tree edge E = (child_node, parent_node)
    /// * A node N is in the tail component of E if min(child_node) <= max(N) <= max(child_node)
    /// * True if node_idx has a tree_dist_max such that: min <= tree_dist_max <= max
    ///
    /// GraphViz code: SEQ(ND_low(v), ND_lim(w), ND_lim(v))
    ///              : SEQ(min, node_idx, max)
    ///              
    /// From the paper: page 12: Section 2.4: Implementation details
    ///
    /// Another valuable optimization, similar to a technique described in [Ch], is to perform a postorder
    /// traversal of the tree, starting from some fixed root node v root, and labeling each node v with its
    /// postorder traversal number lim(v), the least number low(v) of any descendant in the search, and the
    /// edge parent(v) by which the node was reached (see figure 2-5).
    ///
    /// This provides an inexpensive way to test whether a node lies in the head or tail component of a tree edge,
    /// and thus whether a non-tree edge crosses between the two components.  For example, if e = (u, v) is a tree
    /// edge and v root is in the head component of the edge (i.e., lim(u) < lim(v)), then a node w is in the tail
    /// component of e if and only if low(u) ≤ lim(w) ≤ lim(u).  These numbers can also be used to update the
    /// tree efficiently during the network simplex iterations.  If f = (w, x) is the entering edge, the only edges
    /// whose cut values must be adjusted are those in the path connecting w and x in the tree.  This path is determined
    /// by following the parent edges back from w and x until the least common ancestor is reached, i.e., the first node
    /// l such that low(l) ≤ lim(w) , lim(x) ≤ lim(l).  Of course, these postorder parameters must also be adjusted
    /// when exchanging tree edges, but only for nodes below l.
    pub(super) fn node_in_tail_component(&self, node_idx: usize, descendent_min_traversal: usize, max_traversal_number: usize) -> bool {
        let node_traversal_number = self
            .get_node(node_idx)
            .tree_traversal_number()
            .expect("tree_dist_max must be set");

        descendent_min_traversal <= node_traversal_number && node_traversal_number <= max_traversal_number
    }

    /// Sets a feasible tree within the given graph by setting feasible_tree_member on tree member nodes.
    ///
    /// Documentation from the paper: pages 8-9
    /// * The while loop code below finds an edge to a non-tree node that is adjacent to the tree, and adjusts the ranks of
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
    /// Additional paper details: page 7
    /// * A feasible ranking is one satisfying the length constraints l(e) ≥ δ(e) for all e.
    ///   * Thus, a ranking where the all edge rankings are > min_length().  Thus no rank < 1
    ///   * l(e) = length(e) = rank(e1)-rank(e2) = rank_diff(e)
    ///     * length l(e) of e = (v,w) is defined as λ(w) − λ(v)
    ///     * λ(w) − λ(v) = rank(w) - rank(v)
    ///   * δ(e) = min_length(e) = 1 unless requested by user
    /// * Given any ranking, not necessarily feasible, the "slack" of an edge is the difference of its length and its
    ///   minimum length.
    ///   * QUESTION: Is "its minimum length" == MIN_EDGE_LENGTH or just the minium it can be in a tree?
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
            // Don't place this exclusion in a filter, as it can change from the previous call.
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
            //  .find_tightest_incident_edge(sub_tree);
            // let edge_idx = if let Some(edge_idx) = edge_idx {
            //     edge_idx
            // } else {
            //     break
            // };

            println!("   Merging with edge: {edge_idx}");
            let modified_sub_tree_idx = self.merge_sub_trees(edge_idx);
            println!("   Reordering: {modified_sub_tree_idx}");
            min_heap.reorder_item(modified_sub_tree_idx);
            println!("   reorder done");
        }
        self.init_spanning_tree_and_cutvalues();
        self.print_nodes(&format!(
            "after init_spanning_tree_and_cutvalues() in set_feasible_tree:{}",
            min_heap.len()
        ));
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
    ///   of the tree, starting from some fixed root node v root, and labeling each node v with its postorder traversal
    ///   number lim(v), the least number low(v) of any descendant in the search, and the edge parent(v) by which the
    ///   node was reached (see figure 2-5).
    ///   * This provides an inexpensive way to test whether a node lies in the head or tail component of a tree edge,
    ///     and thus whether a non-tree edge crosses between the two components.
    pub fn init_spanning_tree_and_cutvalues(&mut self) {
        if self.node_count() > 0 {
            self.set_tree_parents_and_ranges(true, 0, None, 1);
            self.set_cutvals_depth_first(0);
        }
    }

    /// Starting at the top of the tree, set the tree parent, dist.min & dst.max of all children.
    ///
    /// * If we are not initializing, we can stop when we get to a node that already
    ///   has it's parent and tree_dist_min already correctly set
    ///
    ///
    /// In the graphviz 9.0 code, there are two functions:
    /// * dfs_range_init()
    /// * dfs_range()
    /// They are identical code, expect for the exit condition.  set_tree_parents_and_ranges()
    /// does either depending on the "initializing flag"
    ///
    /// From the paper: Page 13, Section 2.4
    ///
    /// These numbers can also be used to update the tree efficiently during the network simplex
    /// iterations.  If f=(w, x) is the entering edge, the only edges whose cut values must be
    /// adjusted are those in the path connecting w and x in the tree.  This path is determined
    /// by following the parent edges back from w and x until the least common ancestor is
    /// reached, i.e., the first node l such that low(l) ≤ lim(w), lim(x) ≤ lim(l).  Of course,
    /// these postorder parameters must also be adjusted when exchanging tree edges, but only
    /// for nodes below l.
    pub(super) fn set_tree_parents_and_ranges(
        &mut self,
        initializing: bool,
        node_idx: usize,
        parent_edge_idx: Option<usize>,
        min_traversal_number: usize,
    ) -> usize {
        let node = self.get_node(node_idx);

        let par_str = if let Some(parent_edge_idx) = parent_edge_idx {
            self.edge_to_string(parent_edge_idx)
        } else {
            "None".to_string()
        };
        println!(
            "set_tree_parents_and_ranges('{}', {parent_edge_idx:?}, {min})\n  cur_node: {}\n  parent_edge:{par_str}",
            node.name,
            self.node_to_string(node_idx),
        );

        if !initializing {
            // Since we are not initializing, we can stop if:
            // * our tree_parent already set correctly.
            // * our tree_dst_min is already set correctly.
            //   * when the path is invalidated, tree_dst_min is set to: None
            if let Some(tree_data) = node.spanning_tree() {
                if tree_data.edge_idx_to_parent() == parent_edge_idx
                    && tree_data.descendent_min_traversal_number() == Some(min_traversal_number)
                {
                    return tree_data.traversal_number().unwrap_or(0) + 1;
                }
            } else {
                panic!("Setting tree ranges on a node that is not in the tree!");
            }
        }

        let cur_max_traversal_number = node.tree_traversal_number();
        self.get_node_mut(node_idx)
            .set_tree_data(parent_edge_idx, Some(min_traversal_number), cur_max_traversal_number);

        let mut max_traversal_number = min_traversal_number;
        for (node_idx, edge_idx) in self.non_parent_tree_nodes(node_idx) {
            max_traversal_number = max_traversal_number.max(self.set_tree_parents_and_ranges(
                initializing,
                node_idx,
                Some(edge_idx),
                max_traversal_number,
            ));
        }

        self.get_node_mut(node_idx)
            .set_tree_traversal_number(Some(max_traversal_number));

        if max_traversal_number > self.node_count() {
            panic!("set_tree_parents_and_ranges must be looping!")
        }
        max_traversal_number + 1
    }

    /// Set tree_dist_min=None for all nodes from from_node up to and including lca_node.
    /// * This will give us a path from where cut_values will need to be updated.
    /// * We will move progressively up the tree by moving through the tree_parent of from_node
    ///
    /// GraphViz comment for invalidate_path():
    ///   Invalidate DFS attributes by walking up the tree from to_node till lca
    ///   (inclusively). Called when updating tree to improve pruning in dfs_range().
    ///   Assigns ND_low(n) = -1 for the affected nodes.
    ///
    pub(super) fn invalidate_path_to_lca(&self, lca_node_idx: usize, from_node_idx: usize) {
        let lca_node = self.get_node(lca_node_idx);
        let mut from_node_idx = from_node_idx;

        println!(
            "Invalidate path from node: {}",
            self.node_to_string(from_node_idx)
        );
        println!("  to lca node: {}", self.node_to_string(lca_node_idx));
        loop {
            let from_node = self.get_node(from_node_idx);

            if from_node.tree_descendent_min_traversal_number().is_none() {
                break;
            }

            // We are "invalidating" the node buy setting the tree_descendent_min to None
            from_node.set_tree_descendent_min(None);

            if let Some(parent_edge_idx) = from_node.spanning_tree_parent_edge_idx() {
                if from_node.tree_traversal_number() >= lca_node.tree_traversal_number() {
                    if from_node_idx != lca_node_idx {
                        panic!("invalidate_path: skipped over LCA");
                    }
                    break;
                }

                let parent_edge = self.get_edge(parent_edge_idx);
                let parent_src = self.get_node(parent_edge.src_node);
                let parent_dst = self.get_node(parent_edge.dst_node);

                from_node_idx =
                    if parent_src.tree_traversal_number() > parent_dst.tree_traversal_number() {
                        parent_edge.src_node
                    } else {
                        parent_edge.dst_node
                    };
            } else {
                break;
            }
        }
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

        let search_sub_tree_root = self.find_node_sub_tree_root(search_node_idx);
        println!(
            "tightest_incident_edge_search: {search_node_idx} sub_tree:{search_sub_tree_root:?}"
        );
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
                } else {
                    println!("no edge candidates with slack: {edge_slack:?} < {best_cur_slack:?}");
                }
            } else {
                println!("no edge candidates: {search_node_idx} == {next_node_idx}");
            }
        }
        println!("  tightest_incident_edge_search result: best_edge_idx: {best_edge_idx:?}");

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
    /// * Then recursively reduces the rank of all tree descendants of node_idx
    ///   by delta.
    ///   
    /// More intuitively written, this function should add delta rather
    /// than subtract it, but I have left it as a subtract to keep it similar
    /// to what GraphVis does.
    pub(super) fn rerank_by_tree(&self, node_idx: usize, delta: i32) {
        let node = self.get_node(node_idx);

        if let Some(cur_rank) = node.simplex_rank() {
            let new_rank = cur_rank - delta;
            // println!(
            //     "Re-ranking {} by {delta} from {cur_rank} to {new_rank}",
            //     node.name
            // );
            node.set_simplex_rank(Some(new_rank));

            // let children = self
            //     .non_parent_tree_nodes(node_idx)
            //     .iter()
            //     .map(|(node_idx, _)| self.get_node(*node_idx).name.clone())
            //     .collect::<Vec<String>>();
            // println!("   {} re-ranking to: {:?}", node.name, children);

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
    pub(super) fn non_parent_tree_nodes(&self, node_idx: usize) -> Vec<(usize, usize)> {
        let mut nodes = self.directional_non_parent_tree_nodes(node_idx, Out);
        let mut in_nodes = self.directional_non_parent_tree_nodes(node_idx, In);

        nodes.append(&mut in_nodes);
        nodes
    }

    /// Return a vector of (node_idx, edge_idx) which are tree members and do not point to the
    /// parent node of the given node from either in_edges or out_edges depending on disposition.
    ///
    /// Only returns non-ignored nodes.
    fn directional_non_parent_tree_nodes(
        &self,
        node_idx: usize,
        disposition: EdgeDisposition,
    ) -> Vec<(usize, usize)> {
        let given_parent_idx = self.get_node(node_idx).spanning_tree_parent_edge_idx();

        self.directional_non_given_parent_tree_nodes(node_idx, given_parent_idx, disposition)
    }

    /// Return a vector of (node_idx, edge_idx) which are tree members and do not point to the
    /// given parent node.
    ///
    /// This allows for a depth first search starting at any arbitrary node and without a specific
    /// parent node set.
    ///
    /// Only returns non-ignored nodes.
    fn directional_non_given_parent_tree_nodes(
        &self,
        node_idx: usize,
        given_parent_idx: Option<usize>,
        disposition: EdgeDisposition,
    ) -> Vec<(usize, usize)> {
        self.node_tree_edges(node_idx, disposition)
            .iter()
            .filter_map(|edge_idx| {
                if given_parent_idx != Some(*edge_idx) {
                    let edge = self.get_edge(*edge_idx);
                    let other_node_idx = match disposition {
                        In => edge.src_node,
                        Out => edge.dst_node,
                    };

                    Some((other_node_idx, *edge_idx))
                } else {
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
    ) -> Vec<usize> {
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
                    Some(*edge_idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Print out the spanning tree of the graph
    #[allow(unused)]
    pub fn print_spanning_tree_in_dot(&self) {
        let root_nodes = self
            .nodes_iter()
            .enumerate()
            .filter_map(|(node_idx, node)| {
                if node.in_spanning_tree() && node.spanning_tree_parent_edge_idx().is_none() {
                    Some(node_idx)
                } else {
                    None
                }
            });

        let mut count = 0;
        for (index, root_idx) in root_nodes.enumerate() {
            println!("TREE_ROOT {index}: digraph {{");
            for node in self.nodes_iter() {
                println!(
                    r#"{} [label="{} ({}, {})"]; "#,
                    node.name,
                    node.name,
                    node.tree_descendent_min_traversal_number()
                        .unwrap_or_default(),
                    node.tree_traversal_number().unwrap_or_default()
                )
            }
            self.print_tree_helper(root_idx);
            println!("}}");

            count += 1;
        }

        if count == 0 {
            println!("NO ROOT nodes!");
        }
    }

    #[allow(unused)]
    pub fn print_tree_helper(&self, parent_idx: usize) {
        let parent_node = self.get_node(parent_idx);
        let child_nodes = self.non_parent_nodes(parent_idx);
        let child_names = child_nodes
            .iter()
            .map(|(node_idx, tree_edge)| {
                let label = if *tree_edge {
                    ""
                } else {
                    " [color=red, penwidth=.5]"
                };

                format!(
                    "{} -> {}{label}",
                    parent_node.name,
                    self.get_node(*node_idx).name.clone()
                )
            })
            .collect::<Vec<String>>();

        if !child_names.is_empty() {
            println!("    {};", child_names.join("; "));
        }

        for (child_node_idx, tree_edge) in child_nodes.iter() {
            if *tree_edge {
                self.print_tree_helper(*child_node_idx)
            }
        }
    }

    #[allow(unused)]
    // Returns all nodes connected to the given node which is not the parent node.
    //
    // Note that this goes through non-tree edges as well, which is generally
    // not what you want!
    fn non_parent_nodes(&self, node_idx: usize) -> Vec<(usize, bool)> {
        let node = self.get_node(node_idx);
        let parent_idx = self.get_node(node_idx).spanning_tree_parent_edge_idx();

        node.get_all_edges_with_disposition(true)
            .filter_map(|(edge_idx, disposition)| {
                if parent_idx != Some(*edge_idx) {
                    let edge = self.get_edge(*edge_idx);

                    if edge.ignored {
                        None
                    } else {
                        let tree_edge = edge.in_spanning_tree();
                        let other_node_idx = match disposition {
                            In => edge.src_node,
                            Out => edge.dst_node,
                        };

                        if tree_edge || disposition == Out {
                            Some((other_node_idx, tree_edge))
                        } else {
                            None
                        }
                    }
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

    impl Graph {}

    #[test]
    fn test_directional_non_parent_tree_nodes() {
        let mut graph = Graph::from("digraph { a -> b; b -> a; c -> d; c -> a; };");
        let a_idx = graph.name_to_node_idx("a").unwrap();
        let b_idx = graph.name_to_node_idx("b").unwrap();
        let c_idx = graph.name_to_node_idx("c").unwrap();
        let d_idx = graph.name_to_node_idx("d").unwrap();

        graph.make_acyclic();
        graph.merge_edges();
        graph.init_simplex_rank();
        graph.set_feasible_tree_for_simplex(true);

        let a_in_nodes = graph.directional_non_parent_tree_nodes(a_idx, In);
        let a_out_nodes = graph.directional_non_parent_tree_nodes(a_idx, Out);

        // "a" is parent, no all its edges return
        assert_eq!(a_out_nodes, [(b_idx, 0)]);
        assert_eq!(a_in_nodes, [(c_idx, 3)]);

        let c_in_nodes = graph.directional_non_parent_tree_nodes(c_idx, In);
        let c_out_nodes = graph.directional_non_parent_tree_nodes(c_idx, Out);

        // "a" is parent, so we don't see c -> a
        assert_eq!(c_out_nodes, [(d_idx, 2)]);
        assert_eq!(c_in_nodes, []);
    }

    /// This test is here to ensure init_spanning_tree() does not go into an
    /// infinite loop because it adds edges that should be ignored (edges that
    /// point to a node already in the tree).
    #[test]
    fn test_init_spanning_tree_ignore_tree_nodes() {
        let mut graph = Graph::from("digraph { a -> b; a -> c; b -> d; c -> d; }");
        graph.make_acyclic();
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

        graph.make_acyclic();
        graph.merge_edges();
        graph.init_simplex_rank();
        graph.set_feasible_tree_for_simplex(true);
        // at end end calls: init_cutvals() -> graph.init_spanning_tree();

        println!("{graph}");
    }
}
