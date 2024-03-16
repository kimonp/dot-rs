//! As structure which assists ordering graph nodes within previously calculated ranks.

use itertools::Itertools;

use crate::graph::crossing_lines::count_crosses;
use crate::graph::rank_orderings::AdjacentRank::{Above, Below};
use std::collections::{BTreeSet, HashMap};
use std::fmt::Display;
use std::mem::swap;
use std::{cell::RefCell, collections::BTreeMap};

use super::crossing_lines::Line;

/// The usize in the BTree is the node_idx also used in the "nodes" HashMap<usize, <>>
type RankOrder = RefCell<BTreeSet<usize>>;

#[derive(Debug, Clone)]
pub struct RankOrderings {
    /// Ordered list of ranks in the graph, with a set of all node positons at that rank.
    ranks: BTreeMap<i32, RankOrder>,
    /// Map of node_idx to node positions.  Node postions are shared refs into positions in "ranks".
    nodes: RefCell<HashMap<usize, RefCell<NodePosition>>>,
}

impl Display for RankOrderings {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        let max_rank = self.ranks.len();

        for (rank_idx, rank) in self.ranks.iter() {
            let crosses = self.crossing_count_to_next_rank(&rank.borrow());

            if *rank_idx < max_rank as i32 - 1 {
                let _ = writeln!(fmt, "Rank {rank_idx}: crosses below: {crosses}");
            } else {
                let _ = writeln!(fmt, "Rank {rank_idx}:");
            }

            for node_idx in rank.borrow().iter().sorted_by(|a, b| {
                let nodes = self.nodes.borrow();
                {
                    let a_pos = nodes.get(a).unwrap().borrow().position;
                    let b_pos = nodes.get(b).unwrap().borrow().position;

                    a_pos.cmp(&b_pos)
                }
            }) {
                if let Some(node) = self.nodes.borrow().get(node_idx) {
                    let _ = writeln!(fmt, "  {node_idx}: {}", node.borrow());
                } else {
                    let _ = writeln!(fmt, "  {node_idx}: BAD INDEX");
                }
            }
        }
        Ok(())
    }
}

/// NodePosition is the structure with the RankOderings struct that keeps
/// all the critical information about a node's position with ranks so that
/// calculations can be done effeciently.
#[derive(Debug, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub struct NodePosition {
    /// node_idx into the parent graph
    node_idx: usize,
    /// List "adjacent" NodePositions in the rank above this node
    above_adjacents: Vec<usize>,
    /// List "adjacent" NodePositions in the rank below this node
    below_adjacents: Vec<usize>,
    /// Position of this node in the rank.
    position: usize,
    /// Used to adjust the position of this rank.
    median: Option<usize>,
}

impl NodePosition {
    fn new(node_idx: usize, position: usize) -> Self {
        NodePosition {
            node_idx,
            above_adjacents: vec![],
            below_adjacents: vec![],
            position,
            median: None,
        }
    }

    pub fn position(&self) -> usize {
        self.position
    }
}

impl Display for NodePosition {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(
            fmt,
            "idx:{} pos:{} med:{:?} above:{:?} bellow:{:?}",
            self.node_idx, self.position, self.median, self.above_adjacents, self.below_adjacents
        )
    }
}

#[derive(Debug, Clone, Copy)]
pub enum AdjacentRank {
    Above,
    Below,
}

impl RankOrderings {
    pub fn new() -> RankOrderings {
        RankOrderings {
            ranks: BTreeMap::new(),
            nodes: RefCell::new(HashMap::new()),
        }
    }

    /// Add the given node index to the given rank, creating a new rank if necessary.
    pub fn add_node_idx_to_rank(&mut self, rank: i32, node_idx: usize) {
        if self.get_rank_mut(rank).is_none() {
            self.ranks.insert(rank, RefCell::new(BTreeSet::new()));
        };
        self.add_node_idx_to_existing_rank(rank, node_idx);
    }

    pub fn add_node_idx_to_existing_rank(&self, rank: i32, node_idx: usize) {
        let rank_set = if let Some(rank_set) = self.get_rank(rank) {
            rank_set
        } else {
            panic!("Rank {rank} does not exist");
        };

        let position = rank_set.borrow().len();
        rank_set.borrow_mut().insert(node_idx);
        self.nodes.borrow_mut().insert(
            node_idx,
            RefCell::new(NodePosition::new(node_idx, position)),
        );
    }

    pub fn set_adjacent_nodes(&self, node_idx: usize, above: &[usize], below: &[usize]) {
        if let Some(node_pos) = self.nodes.borrow().get(&node_idx) {
            node_pos.borrow_mut().above_adjacents = above.to_vec();
            node_pos.borrow_mut().below_adjacents = below.to_vec();
        } else {
            panic!("node_idx {node_idx} not known");
        }
    }

    /// Iterate through each rank.
    pub fn iter(&self) -> std::collections::btree_map::Iter<'_, i32, RankOrder> {
        self.ranks.iter()
    }

    pub fn nodes(&self) -> &RefCell<HashMap<usize, RefCell<NodePosition>>> {
        &self.nodes
    }

    fn get_rank_mut(&mut self, rank: i32) -> Option<&mut RankOrder> {
        self.ranks.get_mut(&rank)
    }

    fn get_rank(&self, rank: i32) -> Option<&RankOrder> {
        self.ranks.get(&rank)
    }
    
    // /// Return the maximum rank currently in RankOrder.
    // pub fn max_rank(&self) -> Option<i32> {
    //     self.ranks.last_key_value().map(|(rank, _)| *rank)
    // }

    /// Documentation from paper: page 14
    /// * wmedian re-orders the nodes within each rank based on the weighted median heuristic.
    /// * Depending on the parity of the current iteration number, the ranks are traversed from
    ///   top to bottom or from bottom to top.
    /// * To simplify the presentation, ﬁgure the code below only shows one direction in detail
    ///
    /// * In the forward traversal of the ranks, the main loop starts at rank 1 and ends at the maximum rank.
    /// * At each rank a vertex is assigned a median based on the adjacent vertices on the previous rank.
    /// * Then, the vertices in the rank are sorted by their medians.
    /// * An important consideration is what to do with vertices that have no adjacent vertices on the previous rank.
    ///   * In our implementation such vertices are left ﬁxed in their current positions with non-ﬁxed vertices
    ///     sorted into the remaining positions
    ///
    /// wmedian(order, iter) {
    ///     if iter mod 2 == 0 {
    ///         for r in 0..=MAX_RANK {
    ///             for v in order(r) {
    ///                 median[v] = median_value(v, r-1)
    ///             }
    ///             sort(order[r], median);
    ///         }
    ///     } else ...
    /// }
    pub fn weighted_median(&self, index: usize) {
        if index % 2 == 0 {
            for (_rank, rank_order) in self.iter() {
                self.adjust_rank(rank_order, Below);
            }
        } else {
            for (_rank, rank_order) in self.iter().rev() {
                self.adjust_rank(rank_order, Above);
            }
        }
    }

    /// Adjust the rank_order of rank from the adjacent rank specified.
    ///
    /// * Get a median value for each node in the given rank by looking at nodes in
    ///   the specified adjacent rank.
    /// * Sort the nodes in the given rank by the collected medians
    ///
    /// Documentation from paper: page 14
    ///             for v in order(r) {
    ///                 median[v] = median_value(v, r-1)
    ///             }
    ///             sort(order[r], median);
    fn adjust_rank(&self, rank_order: &RankOrder, which_rank: AdjacentRank) {
        // Collect all the node_positions that should be re-ordered, and those that stay where they are
        let mut ordering = vec![];
        let mut static_pos = vec![];

        let nodes = self.nodes.borrow();
        for node_idx in rank_order.borrow().iter() {
            if let Some(node_pos) = nodes.get(node_idx) {
                if self.set_median_value(node_pos, which_rank) {
                    ordering.push(node_pos);
                } else {
                    static_pos.push(node_pos)
                }
            } else {
                panic!("node_idx {node_idx} not found in order");
            }
        }

        // Sort the elements that have a new median value by that median value
        ordering.sort_by(|a, b| a.borrow().median.cmp(&b.borrow().median));

        // Static items (indicated by medium == NONE) stay in their original positions.
        // Place them in order so that they can be inserted into their original places, one by one
        static_pos.sort_by(|a, b| a.borrow().position.cmp(&b.borrow().position));
        for node_pos in static_pos {
            let orig_pos = node_pos.borrow().position;
            ordering.insert(orig_pos, node_pos);
        }

        // Set the values of all the newly calculated positions
        for (new_pos, node_pos) in ordering.iter().enumerate() {
            (*node_pos).borrow_mut().position = new_pos;
        }
    }

    /// Set a median value for the given node_position by looking at the adjacent rank.
    /// * If adjacent nodes == 0: return false and set median = None
    ///   * None means that the node is to be left in their current position when sorted
    /// * If adjacent nodes == 1: set median = Some(adjacent_nodes[0])
    /// * If adjacent nodes == 2: set median = Some(Median of those two node positions)
    /// * If adjacent nodes == 3: ...EXPLAIN: TODO...
    ///
    /// Documentation from paper: page 15
    /// * The median value of a vertex is deﬁned as the median position of the adjacent
    ///   vertices if that is uniquely deﬁned.
    ///   * Otherwise, it is interpolated between the two median positions using a measure
    ///     of tightness.
    ///   * Generally, the weighted median is biased toward the side where vertices are
    ///     more closely packed.
    /// * The adj_position function returns an ordered array of the present positions of the
    ///   nodes adjacent to v in the given adjacent rank.
    /// * Nodes with no adjacent vertices are given a median value of -1.  This is used within
    ///   the sort function to indicate that these nodes should be left in their current positions
    ///
    ///  fn median_value(v,adj_rank) {
    ///      P = adj_position(v,adj_rank);
    ///      m = |P|/2;
    ///      if |P| == 0 {
    ///          return -1.0;
    ///      } else if |P| mod 2 == 1 {
    ///          return P[m];
    ///      } else if |P| == 2 {
    ///          return (P[0] + P[1])/2;
    ///      } else {
    ///          left = P[m-1] - P[0];
    ///          right = P[|P|-1] - P[m];
    ///          return (P[m-1]*right + P[m]*left)/(left+right);
    ///      }
    ///  }
    fn set_median_value(
        &self,
        node_position: &RefCell<NodePosition>,
        which_rank: AdjacentRank,
    ) -> bool {
        let adjacent_nodes = self.adjacent_position(node_position, which_rank);
        let adj_count = adjacent_nodes.len();
        let med = adj_count / 2;

        let median = if adj_count == 0 {
            None
        } else if adj_count % 2 == 1 {
            Some(adjacent_nodes[med])
        } else if adj_count == 2 {
            Some((adjacent_nodes[0] + adjacent_nodes[1]) / 2)
        } else {
            let left = adjacent_nodes[med - 1] - adjacent_nodes[0];
            let right = adjacent_nodes[adj_count - 1] - adjacent_nodes[med];

            Some(adjacent_nodes[med - 1] * right + adjacent_nodes[med] * left / (left + right))
        };

        node_position.borrow_mut().median = median;

        median.is_some()
    }

    /// Return an array of the positions of all nodes adjacent to node_position in the given adjacent rank.
    fn adjacent_position(
        &self,
        node_position: &RefCell<NodePosition>,
        which_rank: AdjacentRank,
    ) -> Vec<usize> {
        let above_rank = &node_position.borrow().above_adjacents;
        let below_rank = &node_position.borrow().below_adjacents;
        let adj_rank = match which_rank {
            Above => above_rank,
            Below => below_rank,
        };

        adj_rank
            .iter()
            .map(|node_idx| {
                if let Some(node_pos) = self.nodes.borrow().get(node_idx) {
                    node_pos.borrow().position
                } else {
                    panic!("{node_idx} not in nodes");
                }
            })
            .sorted()
            .collect::<Vec<usize>>()
    }

    /// Return the number of edges that cross from rank to rank.
    ///
    pub fn crossing_count(&self) -> u32 {
        let mut crossing_count = 0;

        let mut prev_rank: Option<&RefCell<BTreeSet<usize>>> = None;
        for (_rank, rank_order) in self.iter() {
            if let Some(prev_rank) = prev_rank {
                crossing_count += self.crossing_count_to_next_rank(&prev_rank.borrow());
            }
            prev_rank = Some(rank_order);
        }

        crossing_count
    }

    fn crossing_count_to_next_rank(&self, rank: &BTreeSet<usize>) -> u32 {
        let mut lines = vec![];
        let nodes = self.nodes.borrow();

        for node_idx in rank.iter() {
            let node_pos = nodes.get(node_idx).unwrap();

            for adj_idx in node_pos.borrow().below_adjacents.iter() {
                let adj_pos = nodes.get(adj_idx).unwrap();
                let line = Line::new_down(
                    node_pos.borrow().position as u32,
                    adj_pos.borrow().position as u32,
                );
                lines.push(line);
            }
        }
        count_crosses(lines)
    }

    /// Documentation from paper: Page 16
    /// * This is the main loop that iterates as long as the number of edge crossings can be reduced
    ///   by transpositions.
    ///   * TODO: As in the loop in the ordering function, an adaptive strategy could be applied
    ///     here to terminate the loop once the improvement is a sufﬁciently small fraction of
    ///     the number of crossings.
    /// * Each adjacent pair of vertices is examined.
    ///   * Their order is switched if this reduces the number of crossings.
    ///   * The function crossing(v,w) simply counts the number of edge crossings
    ///     if v appears to the left of w in their rank.
    ///
    /// procedure transpose(rank)
    ///     improved = True;
    ///     while improved do
    ///         improved = False;
    ///         for r = 0 to Max_rank do
    ///             for i = 0 to |rank[r]|-2 do
    ///                 v = rank[r][i];
    ///                 w = rank[r][i+1];
    ///                 if crossing(v,w) > crossing(w,v) then
    ///                     improved = True;
    ///                     exchange(rank[r][i],rank[r][i+1]);
    ///                 endif
    ///             end
    ///         end
    ///     end
    /// end
    pub fn transpose(&self) {
        let mut improved = true;

        while improved {
            improved = false;

            for (_rank, rank_set) in self.ranks.iter() {
                let mut rank_position = self.rank_order_to_vec(rank_set);
                let position_count = rank_position.len();
                // println!("--- transpose for rank {_rank} ({position_count} positions)");

                if position_count > 1 {
                    let rank_set = rank_set.borrow();

                    for position in 0..position_count - 1 {
                        let v = rank_position[position];
                        let w = rank_position[position + 1];

                        if self.exchange_if_crosses_decrease(&rank_set, v, w) {
                            // println!("{_rank}: exchanged {position} with {}", position + 1);
                            rank_position.swap(position, position + 1);
                            improved = true
                        } else {
                            //  println!("{_rank}: no exchange for {position}");
                        }
                    }
                }
            }
        }
    }

    /// Exchange positions of the given node indexes within rank if this reduces cross count with the next rank.
    ///
    /// Return true if the row was exchanged (thus reducing the cross count).
    fn exchange_if_crosses_decrease(
        &self,
        rank: &BTreeSet<usize>,
        node_idx_a: usize,
        node_idx_b: usize,
    ) -> bool {
        let cur_value = self.crossing_count_to_next_rank(rank);
        self.exchange_positions(node_idx_a, node_idx_b);
        let new_value = self.crossing_count_to_next_rank(rank);

        if new_value < cur_value {
            true
        } else {
            self.exchange_positions(node_idx_b, node_idx_a);

            false
        }
    }

    /// Exchange positions of node_idx_a and node_idx_b.
    ///
    /// Both nodes should be in the same rank, but this is not enforced.
    fn exchange_positions(&self, node_idx_a: usize, node_idx_b: usize) {
        let nodes = self.nodes.borrow();
        let node_pos_a = nodes.get(&node_idx_a);
        let node_pos_b = nodes.get(&node_idx_b);

        match (node_pos_a, node_pos_b) {
            (Some(node_pos1), Some(node_pos2)) => {
                swap(
                    &mut node_pos1.borrow_mut().position,
                    &mut node_pos2.borrow_mut().position,
                );
            }
            _ => panic!("Unable to get node positions {node_idx_a} and/or {node_idx_b}"),
        }
    }

    /// Return a vector of node_idx from the given rank ordered by position within the rank.
    fn rank_order_to_vec(&self, rank_order: &RankOrder) -> Vec<usize> {
        let rank_order = rank_order.borrow();
        let mut order = vec![0; rank_order.len()];

        for node_pos_idx in rank_order.iter() {
            if let Some(node_position) = self.nodes.borrow().get(node_pos_idx) {
                let position = node_position.borrow().position;

                order[position] = *node_pos_idx;
            } else {
                panic!("Cant find node_pos {node_pos_idx}");
            }
        }
        order
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::graph::{network_simplex::SimplexNodeTarget::VerticalRank, Graph};

    #[test]
    fn test_adjacent_position() {
        let mut graph = Graph::example_graph_from_paper_2_3();
        graph.rank_nodes_vertically();
        let order = graph.init_horizontal_order();

        let node_f = graph.name_to_node_idx("f").unwrap();
        let node_g = graph.name_to_node_idx("g").unwrap();

        let nodes = order.nodes.borrow();
        let node_f_pos = nodes.get(&node_f).unwrap();
        let node_g_pos = nodes.get(&node_g).unwrap();

        let above_f = order.adjacent_position(node_f_pos, Above);
        let below_f = order.adjacent_position(node_f_pos, Below);
        let above_g = order.adjacent_position(node_g_pos, Above);
        let below_g = order.adjacent_position(node_g_pos, Below);

        assert_eq!(above_f, vec![0]);
        assert_eq!(below_f, vec![0]);
        assert_eq!(above_g, vec![0, 1]);
        assert_eq!(below_g, vec![1]);

        // order.weighted_median(1);
        graph.set_horizontal_ordering();
    }

    /// Fixture that generates two ranks, with crossing to oppoite lower ranks, e.g:
    /// [0, 1, 2]
    ///  \  |  /
    ///     |
    ///  /  |  \
    /// [3, 4, 5]
    fn double_rank_orderings(max: usize) -> RankOrderings {
        let mut order = RankOrderings::new();
        let rank1 = 0;
        let rank2 = 1;

        for i in 0..max {
            order.add_node_idx_to_rank(rank1, i);
            order.set_adjacent_nodes(i, &[], &[(2 * max - 1) - i]);
        }

        for i in max..max * 2 {
            order.add_node_idx_to_rank(rank2, i);
            order.set_adjacent_nodes(i, &[(max - 1) - (i - max)], &[]);
        }

        order
    }

    /// Test that transpose() can unravel crosses for two ranks fully crossed.
    #[test]
    fn test_transpose() {
        let max = 5;
        let ordering = double_rank_orderings(max);

        assert_eq!(ordering.crossing_count(), 10);
        ordering.transpose();
        assert_eq!(ordering.crossing_count(), 0);
    }

    /// Test that exchange_if_crosses_decrease() exchanges only when appropriate.
    ///
    /// * Start with an ordering with many crosses
    /// * exchange it in a way that reduces those crosses in a known way
    /// * Ensure the exchange took place.
    /// * Run the same exchange again (it which case it would increase crosses)
    /// * Ensure the exchange did not take place.
    #[test]
    fn test_exchange_if_crosses_decrease() {
        let max = 3;
        let ordering = double_rank_orderings(max);

        assert_eq!(ordering.crossing_count(), 3);
        let rank = ordering.ranks.get(&0).unwrap().borrow();
        ordering.exchange_if_crosses_decrease(&rank, 0, max - 1);
        assert_eq!(ordering.crossing_count(), 0);
        ordering.exchange_if_crosses_decrease(&rank, 0, max - 1);
        assert_eq!(ordering.crossing_count(), 0);
    }

    // Fixture that generates a ordering with a single rank with max positions in that rank.
    fn single_rank_orderings(max: usize) -> (RankOrderings, i32) {
        let mut order = RankOrderings::new();
        let rank_idx = 0;

        for i in 0..max {
            order.add_node_idx_to_rank(rank_idx, i);
        }
        (order, rank_idx)
    }

    #[test]
    fn test_exchange_positions() {
        let (order, rank_idx) = single_rank_orderings(2);
        let rank_order = &order.ranks[&rank_idx];
        let vec = order.rank_order_to_vec(rank_order);
        let (node_idx_a, node_idx_b) = (0, 1);

        assert_eq!(vec[node_idx_a], node_idx_a, "setup not reflected");
        assert_eq!(vec[node_idx_b], node_idx_b, "setup not reflected");

        order.exchange_positions(node_idx_a, node_idx_b);
        let vec = order.rank_order_to_vec(rank_order);

        assert_eq!(vec[node_idx_a], node_idx_b, "exchange not reflected");
        assert_eq!(vec[node_idx_b], node_idx_a, "exchange not reflected");
    }

    /// test that rank_order_to_vec():
    /// * returns a accurate count
    /// * returns an accucate ordering
    /// * If you swap two positions, a new ordering reflects that
    #[test]
    fn test_rank_order_to_vec() {
        let max = 5;
        let (order, rank_idx) = single_rank_orderings(max);
        let rank_order = &order.ranks[&rank_idx];
        let vec = order.rank_order_to_vec(rank_order);

        assert_eq!(vec.len(), max, "incorrect rank size");
        for (index, pos) in vec.iter().enumerate() {
            assert_eq!(index, *pos, "out of position");
        }

        let node_idx_a = 1;
        let node_idx_b = max - 1;
        order.exchange_positions(node_idx_a, node_idx_b);

        let vec = order.rank_order_to_vec(rank_order);

        assert_eq!(vec[node_idx_a], node_idx_b, "exchange not reflected");
        assert_eq!(vec[node_idx_b], node_idx_a, "exchange not reflected");
    }

    // Test that the 2_3 example from the paper has zero crosses after ordering.
    #[test]
    fn test_order_example_from_paper_2_3() {
        let mut graph = Graph::example_graph_from_paper_2_3();

        graph.init_simplex_rank();
        graph.assign_simplex_rank(VerticalRank);
        graph.rank_nodes_vertically();
        let order = graph.init_horizontal_order();

        assert_eq!(order.crossing_count(), 1);
        println!("ORDER: {order}");
        println!("Nodes: {:?}", order.nodes().borrow().keys().sorted());
        println!("Node Zero: {:?}", order.nodes().borrow().get(&0));

        let order = graph.set_horizontal_ordering();
        assert_eq!(order.crossing_count(), 0);
    }
}
