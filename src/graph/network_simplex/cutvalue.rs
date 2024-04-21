//! Methods for the network simplex algorithm that assist with generating
//! cutvalues for edges.
//!
//! Cutvalues are a key concept in graphviz layout, effecting both vertical
//! and horizontal positioning.  Cutvalues are the value that the network simplex seeks
//! to maximize in it's network flow calculation.
//!
//! From page 9 of the paper: Section 2.3 Network simplex
//!
//! Given a feasible spanning tree, we can associate an integer cut value with each tree edge as
//! follows. If the tree edge is deleted ("cut"), the tree breaks into two connected components,
//! the tail component containing the tail node of the edge, and the head component containing
//! the head node. The cut value is deﬁned as the sum of the weights of all edges from the
//! tail component to the head component, including the tree edge, minus the sum of the weights
//! of all edges from the head component to the tail component.

use super::Graph;

impl Graph {
    /// Adjust cutvalues based on the changing edges, and update cut values of tree edges.
    ///
    /// In SpanningTree Data, we have renamed the following fields from the paper to make it more understandable:
    /// * node.tree_dist_max(): lim(n)
    /// * node.tree_dist_min(): low(n)
    ///
    /// From the paper: page 12: Section 2.4: Implementation details
    ///
    /// Another valuable optimization, similar to a technique described in [Ch], is to perform a postorder
    /// traversal of the tree, starting from some fixed root node v root, and labeling each node v with its
    /// postorder traversal number lim(v), the least number low(v) of any descendant in the search, and the
    /// edge parent(v) by which the node was reached (see figure 2-5).
    ///
    /// This provides an inexpensive way to test whether a node lies in the head or tail component of a tree edge,
    /// and thus whether a non-tree edge crosses between the two components.  For example, if e = (u ,v) is a tree
    /// edge and v root is in the head component of the edge (i.e., lim(u) < lim(v)), then a node w is in the tail
    /// component of e if and only if low(u) ≤ lim(w) ≤ lim(u).  These numbers can also be used to update the
    /// tree efficiently during the network simplex iterations.  If f = (w ,x) is the entering edge, the only edges
    /// whose cut values must be adjusted are those in the path connecting w and x in the tree.  This path is determined
    /// by following the parent edges back from w and x until the least common ancestor is reached, i.e., the first node
    /// l such that low (l) ≤ lim(w) , lim(x) ≤ lim (l).  Of course, these postorder parameters must also be adjusted
    /// when exchanging tree edges, but only for nodes below l.
    pub(super) fn adjust_cutvalues_and_exchange_for_simplex(
        &mut self,
        neg_cut_edge_idx: usize,
        selected_edge_idx: usize,
    ) {
        let cutvalue = self
            .get_edge(neg_cut_edge_idx)
            .cut_value
            .expect("Selected edge must have cut value");
        let selected_edge = self.get_edge(selected_edge_idx);
        let sel_src_node_idx = selected_edge.src_node;
        let sel_dst_node_idx = selected_edge.dst_node;

        let lca_idx =
            self.adjust_cutvalues_to_lca(sel_src_node_idx, sel_dst_node_idx, cutvalue, true);
        let lca_idx2 =
            self.adjust_cutvalues_to_lca(sel_dst_node_idx, sel_src_node_idx, cutvalue, false);

        assert_eq!(lca_idx, lca_idx2, "Least common ancestor must match");

        let lca = self.get_node(lca_idx);
        let lca_min = lca
            .tree_dist_min()
            .expect("lca does not have a sub_tree_idx_min");

        self.invalidate_path_to_lca(lca_idx, sel_dst_node_idx);
        self.invalidate_path_to_lca(lca_idx, sel_src_node_idx);

        self.get_edge_mut(selected_edge_idx).cut_value = Some(-cutvalue);
        self.get_edge_mut(neg_cut_edge_idx).cut_value = None;

        // self.print_spanning_tree_in_dot();

        self.exchange_edges_in_spanning_tree(neg_cut_edge_idx, selected_edge_idx);

        // println!(
        //     "LCA of {} and {} is: {}",
        //     self.get_node(sel_src_node_idx).name,
        //     self.get_node(sel_dst_node_idx).name,
        //     self.get_node(lca_idx).name
        // );
        let lca = self.get_node(lca_idx);
        let lca_parent_edge_idx = lca.spanning_tree_parent_edge_idx();
        self.set_tree_parents_and_ranges(false, lca_idx, lca_parent_edge_idx, lca_min);
    }

    /// Adjust cutvalues by the given amount from node_idx1 up to the least common ancestor of nodes node_idx1 and node_idx2,
    /// and return the least common ancestor of node_idx1 and node_idx2.
    ///
    /// "down" is a signal as to which direction to change the cutvalue.  If down, the cutvalue should be increased if the next
    /// parent is the src_node.
    ///
    /// This is an efficient way of updating only the needed cutvalues during network simplex
    /// without having to recalculate them all, which can be a large percentage of node layout
    /// calculations.
    ///
    /// * Find the common ancestor by selecting a node (node1), and loop until we move past
    ///   the common ancestor with node2.
    fn adjust_cutvalues_to_lca(
        &mut self,
        node1_idx: usize,
        node2_idx: usize,
        cutvalue: i32,
        down: bool,
    ) -> usize {
        let mut maybe_lca_idx = node1_idx;

        while !self.is_common_ancestor(maybe_lca_idx, node2_idx) {
            let not_lca = self.get_node_mut(maybe_lca_idx);
            let parent_edge_idx = not_lca
                .spanning_tree_parent_edge_idx()
                .expect("Must have a common ancestor");
            let (parent_src_dist_max, parent_dst_dist_max) =
                self.edge_tree_dist_max(parent_edge_idx);
            let parent_edge = self.get_edge_mut(parent_edge_idx);
            let cur_cutvalue = parent_edge
                .cut_value
                .expect("cutvalue not set for parent edge");
            let is_down = if maybe_lca_idx == parent_edge.src_node {
                down
            } else {
                !down
            };

            parent_edge.cut_value = if is_down {
                Some(cur_cutvalue + cutvalue)
            } else {
                Some(cur_cutvalue - cutvalue)
            };

            maybe_lca_idx = if parent_src_dist_max > parent_dst_dist_max {
                parent_edge.src_node
            } else {
                parent_edge.dst_node
            };
        }

        maybe_lca_idx
    }

    /// Return the sub_tree_idx_max() for both the src and dst nodes of an edge.
    ///
    /// Panics if either node is not in the spanning tree, or does not have a sub_tree_idx_max value set.
    fn edge_tree_dist_max(&self, parent_edge_idx: usize) -> (usize, usize) {
        let parent_edge = self.get_edge(parent_edge_idx);
        let parent_src_dist_max = self
            .get_node(parent_edge.src_node)
            .tree_dist_max()
            .expect("tree_dist_max not set");
        let parent_dst_dist_max = self
            .get_node(parent_edge.dst_node)
            .tree_dist_max()
            .expect("tree_dist_max not set");

        (parent_src_dist_max, parent_dst_dist_max)
    }

    /// Set the cutvalues of all edges in the tree via a depth first search.
    ///
    /// * Start a depth first search on all nodes of this node not pointed to the parent.
    /// * After this is complete, set the cutvalue of this edge.
    ///
    /// GraphViz: dfs_cutval()
    pub(super) fn set_cutvals_depth_first(&mut self, node_idx: usize) {
        for (other_idx, _edge_idx) in self.non_parent_tree_nodes(node_idx) {
            self.set_cutvals_depth_first(other_idx)
        }
        if let Some(parent_edge_idx) = self.get_node(node_idx).spanning_tree_parent_edge_idx() {
            self.set_cutval(parent_edge_idx)
        }
    }

    /// Set the cut value of edge_idx by summing cutvalue components from each edge.
    ///
    /// * Assumes that cut values of edges on one side edge_idx have already been set
    ///   which will be true if called using a depth first search.
    ///
    /// GraphViz: x_cutval()
    fn set_cutval(&mut self, edge_idx: usize) {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let parent_edge_idx = src_node.spanning_tree_parent_edge_idx();
        let (edge_points_to_searched, searched_node_idx) = if parent_edge_idx == Some(edge_idx) {
            (true, edge.src_node)
        } else {
            (false, edge.dst_node)
        };

        let search_edges = self.get_node(searched_node_idx).get_all_edges();
        let sum = search_edges
            .map(|edge_idx| {
                self.calc_cutvalue_component(*edge_idx, searched_node_idx, edge_points_to_searched)
            })
            .sum();

        self.get_edge_mut(edge_idx).cut_value = Some(sum);
    }

    /// Compute the component of a cutvalue for another edge.
    /// Pass in:
    /// * edge_idx: The edge who will be contributing a cutvalue component
    /// * searched_node_idx: The node connected to edge_idx which has already been searched
    /// * edge_points_to_searched: True if this edge points to a node that has already been
    ///   searched (and thus is not the node is not tree parent of this edge)
    ///
    /// * Components from all edges of a node can be summed together to calculate a
    ///   cutvalue without looking through all the other nodes and edges.
    /// * This only works in the context of a depth first search, where the cutvalues
    ///   of nodes farther away from the root have already been calculated.
    /// * This works because the cutvalues of tree leaves can be locally calcualted
    ///   and the result of the cutvalues can be built from there.
    ///
    /// GraphViz: x_val()
    ///
    /// From paper: page 11, section 2.4:
    ///
    /// To reduce this cost (of caculating cutvalues), we note that the cut values can be
    /// computed using information local to an edge if the search is ordered from the leaves
    /// of the feasible tree inward.  It is trivial to compute the cut value of a tree edge
    /// with one of its endpoints a leaf in the tree, since either the head or the tail
    /// component consists of a single node.  Now, assuming the cut values are known for all
    /// the edges incident on a given node except one, the cut value of the remaining edge is
    /// the sum of the known cut values plus a term dependent only on the edges incident to
    /// the given node.
    ///
    /// We illustrate this computation in ﬁgure 2-4 in the case where two tree edges, with
    /// known cut values, join a third, with the shown orientations. The other cases are handled
    /// similarly.  We assume the cut values of (u, w) and (v, w) are known.  The edges labeled
    /// with capital letters represent the set of all non-tree edges with the given direction
    /// and whose heads and tails belong to the components shown. The cut values of (u, w) and
    /// (v ,w) are given by
    ///
    /// c(u, w) = ω(u, w) + A + C + F − B − E − D
    /// and
    /// c(v, w) = ω(v, w) + L + I + D − K − J − C
    ///
    /// respectively.  The cut value of (w, x) is then
    ///
    /// c(w, x) = ω(w, x) + G − H + A − B + L − K
    ///         = ω(w, x) + G − H + (c(u, w) − ω(u, w) − C − F + E + D) + (c (v, w) − ω(v, w) − I − D + J + C)
    ///         = ω(w, x) + G − H + c(u, w) − ω(u, w) + c(v ,w) − ω(v, w) − F + E − I + J
    ///
    /// an expression involving only local edge information and the known cut values.  By thus
    /// computing cut values incrementally, we can ensure that every edge is examined only twice.
    /// This greatly reduces the time spent computing initial cut values.
    fn calc_cutvalue_component(
        &self,
        edge_idx: usize,
        searched_node_idx: usize,
        edge_points_to_searched: bool,
    ) -> i32 {
        let edge = self.get_edge(edge_idx);
        let src_node_searched = edge.src_node == searched_node_idx;
        let dst_node_searched = edge.dst_node == searched_node_idx;
        let child_node_searched = if edge_points_to_searched {
            dst_node_searched
        } else {
            src_node_searched
        };

        let unsearched_node_idx = if src_node_searched {
            edge.dst_node
        } else {
            edge.src_node
        };

        let search_node_is_ancestor =
            self.is_common_ancestor(searched_node_idx, unsearched_node_idx);
        let negate_component = (search_node_is_ancestor && !child_node_searched)
            || (!search_node_is_ancestor && child_node_searched);

        let edge_weight = edge.weight as i32;
        let cutvalue_component = if search_node_is_ancestor {
            let cur_cutvalue = if edge.in_spanning_tree() {
                edge.cut_value.unwrap_or_default()
            } else {
                0
            };

            cur_cutvalue - edge_weight
        } else {
            edge_weight
        };

        if negate_component {
            -cutvalue_component
        } else {
            cutvalue_component
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    impl Graph {
        fn assert_expected_cutvals(&self, expected_cutvals: Vec<(&str, &str, i32)>) {
            for (src_name, dst_name, expected_cut_val) in expected_cutvals {
                let (edge, _) = self.get_named_edge(src_name, dst_name);

                assert_eq!(
                    Some(expected_cut_val),
                    edge.cut_value,
                    "unexpected cut_value for edge {src_name}->{dst_name}"
                );
                if Some(expected_cut_val) != edge.cut_value {
                    println!(
                        "unexpected cut_value for edge {src_name}->{dst_name}: {} vs {:?}",
                        expected_cut_val, edge.cut_value
                    );
                }
            }
        }
    }

    #[test]
    fn test_init_cut_values_2_3_a() {
        let (mut graph, expected_cutvals) = Graph::configure_example_2_3_a();

        graph.init_spanning_tree_and_cutvalues();
        graph.assert_expected_cutvals(expected_cutvals);
    }

    #[test]
    fn test_init_cut_values_2_3_b() {
        let (mut graph, expected_cutvals) = Graph::configure_example_2_3_b();

        graph.init_spanning_tree_and_cutvalues();
        graph.assert_expected_cutvals(expected_cutvals);
    }

    #[test]
    fn test_init_cut_values_2_3_extended() {
        let (mut graph, expected_cutvals) = Graph::configure_example_2_3_a_extended();

        graph.init_spanning_tree_and_cutvalues();
        graph.assert_expected_cutvals(expected_cutvals);
    }
}
