//! Methods for the network simplex algorithm that assist with generating
//! cutvalues for edges.

use super::Graph;

impl Graph {
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
        let negate_component = (search_node_is_ancestor && !child_node_searched) || (!search_node_is_ancestor && child_node_searched);

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
    use crate::graph::network_simplex::SimplexNodeTarget::XCoordinate;
    
    impl Graph {
        fn assert_expected_cutvals(&self, expected_cutvals: Vec<(&str, &str, i32)>) {
            for (src_name, dst_name, expected_cut_val) in expected_cutvals {
                let (edge, _) = self.get_named_edge(src_name, dst_name);

                assert_eq!(Some(expected_cut_val), edge.cut_value, "unexpected cut_value for edge {src_name}->{dst_name}");
                if Some(expected_cut_val) != edge.cut_value {
                    println!("unexpected cut_value for edge {src_name}->{dst_name}: {} vs {:?}", expected_cut_val, edge.cut_value);
                }
            }
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

        graph.rank_nodes_vertically();
        println!("{graph}");
    }

    /// This test sets a feasible_tree to match what GraphViz chooses to see if
    /// the choice of feasible tree is responsible for how it is graphed (and it appears to be)
    ///
    /// Ignored because for this to work, the feasible tree must not be reset during the XCoordinate
    /// run of network simplex.
    #[ignore]
    #[test]
    fn test_advanced_cutvalues() {
        let mut graph = Graph::from("digraph { a -> b; a -> c; b -> d; c -> d; }");
        graph.rank_nodes_vertically();
        graph.set_horizontal_ordering();
        graph.set_y_coordinates();

        graph.skip_tree_init = true;

        let mut aux_graph = graph.create_positioning_aux_graph();
        aux_graph.make_asyclic();
        let v7_idx = 7;
        let v6_idx = 6;
        let v5_idx = 5;
        let v4_idx = 4;
        let a_idx = 0;
        let b_idx = 1;
        let c_idx = 2;
        let d_idx = 3;

        aux_graph
            .get_node_mut(d_idx)
            .set_tree_data(Some(v7_idx), None, None);
        aux_graph
            .get_node_mut(v6_idx)
            .set_tree_data(Some(d_idx), None, None);
        aux_graph
            .get_node_mut(b_idx)
            .set_tree_data(Some(v6_idx), None, None);
        aux_graph
            .get_node_mut(c_idx)
            .set_tree_data(Some(b_idx), None, None);
        aux_graph
            .get_node_mut(v4_idx)
            .set_tree_data(Some(b_idx), None, None);
        aux_graph
            .get_node_mut(a_idx)
            .set_tree_data(Some(v4_idx), None, None);
        aux_graph
            .get_node_mut(v5_idx)
            .set_tree_data(Some(a_idx), None, None);

        for (src_name, dst_name) in [
            ("v7", "d"),
            ("v6", "d"),
            ("v6", "b"),
            ("b", "c"),
            ("v4", "b"),
            ("v4", "a"),
            ("v5", "a"),
        ] {
            let (_, edge_idx) = aux_graph.get_named_edge(src_name, dst_name);
            aux_graph.get_edge_mut(edge_idx).set_in_spanning_tree(true);
        }
        println!("--------START HERE------");
        println!("{aux_graph}");

        aux_graph.init_spanning_tree_and_cutvalues();
        aux_graph.network_simplex_ranking(XCoordinate);

        println!("AUX: {aux_graph}");
        graph.set_x_coordinates_from_aux(&aux_graph);
        println!("GRAPH{graph}");

        let svg = crate::svg::SVG::new(graph, false);
        svg.write_to_file("foo");
    }

    /// This test sets a feasible_tree to match what GraphViz chooses to see if
    /// the choice of feasible tree is responsible for how it is graphed (and it appears to be)
    ///
    /// Ignored because for this to work, the feasible tree must not be reset during the XCoordinate
    /// run of network simplex.
    #[ignore]
    #[test]
    fn test_cutvalues_b_and_c_to_a() {
        let mut graph = Graph::from("digraph { b -> a; c -> a; }");
        graph.rank_nodes_vertically();
        graph.set_horizontal_ordering();
        graph.set_y_coordinates();

        let mut aux_graph = graph.create_positioning_aux_graph();
        aux_graph.make_asyclic();

        graph.skip_tree_init = true;
        let v4_idx = 4;
        let v3_idx = 3;
        let b_idx = 0;
        let a_idx = 1;
        let c_idx = 2;

        for (child_idx, parent_idx) in [
            (a_idx, v4_idx),
            (v3_idx, a_idx),
            (v3_idx, b_idx),
            (b_idx, c_idx),
        ] {
            aux_graph
                .get_node_mut(child_idx)
                .set_tree_data(Some(parent_idx), None, None);
        }

        for (src_name, dst_name) in [("v4", "a"), ("v3", "a"), ("v3", "b"), ("b", "c")] {
            let (_, edge_idx) = aux_graph.get_named_edge(src_name, dst_name);
            aux_graph.get_edge_mut(edge_idx).set_in_spanning_tree(true);
        }
        aux_graph.init_spanning_tree_and_cutvalues();
        println!("--------START HERE------");
        println!("{aux_graph}");

        aux_graph.network_simplex_ranking(XCoordinate);

        println!("AUX: {aux_graph}");
        graph.set_x_coordinates_from_aux(&aux_graph);
        println!("GRAPH: {graph}");

        let svg = crate::svg::SVG::new(graph, false);
        svg.write_to_file("foo");
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