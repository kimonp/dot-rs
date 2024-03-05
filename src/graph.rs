//! Implement a graph that can be drawn using the algorithm described in the 1993 paper:
//! "A Technique for Drawing Directed Graphs" by Gansner, Koutsofios, North and Vo
//!
//! This paper is referred to as simply "the paper" below.

mod crossing_lines;
pub mod dot_parser;
mod edge;
mod network_simplex;
mod node;
mod rank_orderings;

use rank_orderings::RankOrderings;
use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    mem::replace,
};

use self::{
    edge::{Edge, EdgeDisposition},
    network_simplex::SimplexNodeTarget::{VerticalRank, XCoordinate},
    node::{Node, NodeType},
    rank_orderings::AdjacentRank,
};

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

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
            edges: vec![],
        }
    }

    /// Lays out the nodes in the graph as described in the graphviz dot paper:
    ///
    /// "A Technique for Drawing Directed Graphs"
    pub fn layout_nodes(&mut self) {
        self.rank_nodes_vertically();
        self.set_horizontal_ordering();
        self.set_horizontal_coordinates();
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

    /// Return the max position in any rank, and the max rank.
    fn max_positions(&self) -> (u32, u32) {
        let mut max_pos = 0;
        let mut max_rank = 0;

        for node in self.nodes.iter() {
            if let Some(pos) = node.horizontal_position {
                max_pos = max_pos.max(pos);
            }
            if let Some(rank) = node.vertical_rank {
                max_rank = max_rank.max(rank);
            }
        }
        (max_pos as u32, max_rank)
    }

    fn get_vertical_adjacent_nodes(&self, node_idx: usize) -> (Vec<usize>, Vec<usize>) {
        let mut above_nodes = vec![];
        let mut below_nodes = vec![];

        for (edge_idx, direction) in self.get_vertical_adjacent_edges(node_idx) {
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

    fn get_vertical_adjacent_edges(
        &self,
        node_idx: usize,
    ) -> impl Iterator<Item = (usize, AdjacentRank)> + '_ {
        let node = self.get_node(node_idx);
        let node_rank = node.vertical_rank;

        node.get_all_edges().cloned().filter_map(move |edge_idx| {
            if let Some(other_node_idx) = self.get_connected_node(node_idx, edge_idx) {
                let other_node = self.get_node(other_node_idx);

                if let (Some(n1), Some(n2)) = (node_rank, other_node.vertical_rank) {
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

    /// Add a new node identified by name, and return the node's index in the graph.
    pub fn add_node(&mut self, name: &str) -> usize {
        let new_node = Node::new(name);
        let idx = self.nodes.len();
        self.nodes.push(new_node);

        idx
    }

    /// Add a new node (marked as virtual=true)
    ///
    /// Virtual nodes are (typically) added after ranking.  Once
    /// ranking has been done, all nodes must be ranked, so
    /// a rank must be passed in for the new virtual node.
    pub fn add_virtual_node(&mut self, rank: u32, node_type: NodeType) -> usize {
        if !node_type.is_virtual() {
            panic!("Cannot add read node as a virtual node");
        }

        let idx = self.add_node("V");
        let v_node = self.get_node_mut(idx);

        v_node.node_type = node_type;
        v_node.simplex_rank = Some(rank);
        v_node.vertical_rank = Some(rank);

        idx
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
    pub fn rank_nodes_vertically(&mut self) {
        self.make_asyclic();
        self.merge_edges();
        self.network_simplex_ranking(VerticalRank);
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
            if !node.in_spanning_tree() {
                return Some(node_idx);
            }
        }
        None
    }

    /// Return a queue of nodes that don't have incoming edges (source nodes).
    fn get_source_nodes(&self) -> VecDeque<usize> {
        let mut queue = VecDeque::new();

        for (node_idx, _node) in self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_i, n)| n.no_in_edges())
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
            self.get_node_mut(node_idx).set_empty_tree_node();

            let node = self.get_node(node_idx);
            let mut edges_to_reverse = Vec::new();
            for edge_idx in node.out_edges.iter().cloned() {
                let edge = self.get_edge(edge_idx);
                let dst_node = self.get_node(edge.dst_node);

                if !dst_node.in_spanning_tree() {
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
    fn set_horizontal_ordering(&mut self) -> RankOrderings {
        const MAX_ITERATIONS: usize = 24;
        let order = self.init_horizontal_order();
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
        }
        // println!(
        //     "-- Final order (crosses: {}): --\n{best}",
        //     best.crossing_count()
        // );

        self.set_node_positions(&best);

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
    fn init_horizontal_order(&mut self) -> RankOrderings {
        let order = self.get_initial_horizontal_orderings();

        self.fill_vertical_rank_gaps(&order);
        self.set_adjacent_nodes_in_vertical_ranks(&order);

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
    fn fill_vertical_rank_gaps(&mut self, order: &RankOrderings) {
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

            let virt_node_idx = self.add_virtual_node(new_rank, NodeType::RankFiller);

            let old_edge = self.get_edge_mut(cur_edge_idx);
            let orig_dst = replace(&mut old_edge.dst_node, virt_node_idx);

            cur_edge_idx = self.add_edge(virt_node_idx, orig_dst);
            order.add_node_idx_to_existing_rank(new_rank, virt_node_idx);

            remaining_slack += if reverse_edge { 1 } else { -1 };
        }
    }

    /// Return an initial horizontal ordering of each of the graph ranks.
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
    fn get_initial_horizontal_orderings(&mut self) -> RankOrderings {
        let mut rank_order = RankOrderings::new();
        let mut dfs_queue = self.get_min_vertical_rank_nodes();
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
                if let Some(rank) = node.vertical_rank {
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
    fn set_adjacent_nodes_in_vertical_ranks(&self, rank_order: &RankOrderings) {
        for (node_idx, _node_position) in rank_order.nodes().borrow().iter() {
            let (above_adj, below_adj) = self.get_vertical_adjacent_nodes(*node_idx);

            rank_order.set_adjacent_nodes(*node_idx, &above_adj, &below_adj);
        }
    }

    /// Return a VecDequeue of nodes which have minimum rank.
    ///
    /// * Assumes that the graph has been ranked
    fn get_min_vertical_rank_nodes(&self) -> VecDeque<usize> {
        let mut min_rank_nodes = VecDeque::new();
        let min_rank = self
            .nodes
            .iter()
            .min()
            .and_then(|min_node| min_node.vertical_rank);

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.vertical_rank == min_rank {
                min_rank_nodes.push_back(node_idx);
            }
        }
        min_rank_nodes
    }

    // /// Return a hash map of rank -> vec<node_idx> as well as the minimum rank
    // fn get_rank_map(&self) -> (Option<u32>, HashMap<u32, Vec<usize>>) {
    //     let mut ranks: HashMap<u32, Vec<usize>> = HashMap::new();
    //     let mut min_rank = None;

    //     for (node_idx, node) in self.nodes.iter().enumerate() {
    //         if let Some(rank) = node.simplex_rank {
    //             if let Some(level) = ranks.get_mut(&rank) {
    //                 level.push(node_idx);
    //             } else {
    //                 ranks.insert(rank, vec![node_idx]);
    //             }

    //             min_rank = if let Some(min_rank) = min_rank {
    //                 Some(u32::min(min_rank, rank))
    //             } else {
    //                 Some(rank)
    //             };
    //         }
    //     }

    //     (min_rank, ranks)
    // }

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
    fn set_horizontal_coordinates(&mut self) {
        self.init_node_coordinates();
        let mut aux_graph = self.create_positioning_aux_graph();

        aux_graph.network_simplex_ranking(XCoordinate);

        // Technically, once we find the firs XCoordCalc node, we can exit, since they are all added
        // after other nodes.  At least, that's true now...
        for (node_idx, node) in aux_graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_idx, node)| node.node_type != NodeType::XCoordCalc)
        {
            let x = node.coordinates.unwrap().x();

            self.get_node_mut(node_idx).coordinates.unwrap().set_x(x);
        }
    }

    /// Initialize the x and y coordinates of all nodes.
    fn init_node_coordinates(&mut self) {
        for (node_idx, node) in self.nodes.iter_mut().enumerate() {
            let min_x = node.min_separation_x();
            let min_y = node.min_separation_y();
            let new_x = if let Some(position) = node.horizontal_position {
                position as u32 * min_x
            } else {
                panic!("Node {node_idx}: position not set");
            };
            let new_y = if let Some(rank) = node.vertical_rank {
                rank * min_y
            } else {
                panic!("Node {node_idx}: rank not set");
            };

            node.set_coordinates(new_x, new_y);
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

        // Add all the existing edges to the current graph without edges.
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
                aux_graph.add_virtual_node(src_node.vertical_rank.unwrap(), NodeType::XCoordCalc);
            let omega = Edge::edge_omega_value(src_node.is_virtual(), dst_node.is_virtual());
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

            // TODO: deal with unwraps();
            let new_node = aux_graph.get_node_mut(new_node_idx);
            let src_x = src_node.coordinates.unwrap().x();
            let src_y = src_node.coordinates.unwrap().y();
            let dst_x = dst_node.coordinates.unwrap().x();

            // ...assigning each n_e the value min(x_u, x_v), using the notation of ﬁgure 4-2
            // and where x_u and x_v are the X coordinates assigned to u and v in G.
            new_node.set_coordinates(src_x.min(dst_x), src_y);
        }

        // Add edges between adjacent nodes in a rank, as per figure 4-2 in the paper.
        // TODO

        aux_graph
    }

    #[allow(unused)]
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
        let omega = Edge::edge_omega_value(src_node.is_virtual(), dst_node.is_virtual());
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

    /// Return an SVG representation of graph.
    /// * self.layout_nodes() must be called first.
    /// * debug=true will display additional debugging information on the svg
    ///
    /// For now, this is just quick and dirty to be able to see the graph
    /// in svg form.
    ///
    /// Later, the dot parser of graph should be improved, and the output
    /// of get_svg() should be generalized and take the parser directives
    /// into account.
    pub fn get_svg(&self, debug: bool) -> String {
        const DEFAULT_X: usize = 1;
        const DEFAULT_Y: u32 = 1;
        let (max_pos, max_rank) = self.max_positions();
        let max_pos = max_pos as f64;
        let max_rank = max_rank as f64;

        let mult = 110.0;
        let (x_scale, y_scale) = if max_pos > max_rank {
            (1.0, max_pos / max_rank)
        } else {
            (max_rank / max_pos, 1.0)
        };
        let x_scale = x_scale * mult;
        let y_scale = y_scale * mult;

        let width = max_pos * x_scale;
        let height = max_rank * y_scale;
        let px_size = 1_f64 / y_scale * 1.0;

        let vb_width = max_pos + 0.5 + 0.5;
        let vb_height = max_rank + 0.5 + 0.5;

        // To set the coordinates within the svg We set the viewbox's height and width
        let mut svg = vec![
            format!(
                r#"<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"
                    viewBox="-0.5 -0.5 {vb_width} {vb_height}"
                    preserveAspectRatio="xMinYMin meet"
                    >"#
            ),
            format!(
                r#"<defs>
                    <marker
                        id="arrow-head"
                        orient="auto"
                        viewBox="0 0 10 10"
                        refX="10" refY="5"
                        markerWidth="10" markerHeight="10"
                    >
                        <path d="M0,0 L10,5 L0,10 Z" fill="black"/>
                    </marker>
                </defs>"#
            ),
        ];

        let real_radius = 0.2;
        let virtual_radius = if debug { real_radius } else { 0.0 };

        for edge in self.edges.iter() {
            let reversed = edge.reversed;
            let (src_node_idx, dst_node_idx) = if reversed {
                (edge.dst_node, edge.src_node)
            } else {
                (edge.src_node, edge.dst_node)
            };
            let src_node = self.get_node(src_node_idx);
            let src_x = src_node.horizontal_position.unwrap_or(DEFAULT_X) as f64;
            let src_y = src_node.vertical_rank.unwrap_or(DEFAULT_Y) as f64;

            let dst_node = self.get_node(dst_node_idx);
            let mut dst_x = dst_node.horizontal_position.unwrap_or(DEFAULT_X + 1) as f64;
            let mut dst_y = dst_node.vertical_rank.unwrap_or(DEFAULT_Y + 1) as f64;

            let slope = (src_y - dst_y) / (src_x - dst_x); // # TODO
            let theta = slope.atan();
            let show_dst = !dst_node.is_virtual() || debug;
            let node_radius = if show_dst {
                // TODO: This is just an approximation if the node
                //       is shows as an ellipse with x_radius=1.5*y_radius
                let node_radius_x = real_radius * 1.5 * 0.8;

                if debug {
                    node_radius_x
                } else {
                    real_radius
                }
            } else {
                virtual_radius
            };

            let y_offset = theta.sin() * node_radius;
            let x_offset = theta.cos() * node_radius;

            if (slope < 0.0 && !reversed) || (slope > 0.0 && reversed) {
                dst_x += x_offset;
                dst_y += y_offset;
            } else {
                dst_x -= x_offset;
                dst_y -= y_offset;
            }

            let dashed = if !debug || edge.in_spanning_tree() {
                ""
            } else {
                r#"stroke-dasharray="0.015""#
            };
            let marker = if show_dst {
                r#"marker-end="url(#arrow-head)""#
            } else {
                ""
            };

            svg.push(format!(
                r#"<path {marker} d="M{src_x} {src_y} L{dst_x} {dst_y}" stroke="black" {dashed} stroke-width="{px_size}px"/>"#
            ));

            if debug {
                let font_size = 0.1;
                let font_style = format!("font-size:{font_size}; text-anchor: left");
                let label_x = (src_x + dst_x) / 2.0 + (font_size / 3.0);
                let label_y = (src_y + dst_y) / 2.0 + (font_size / 3.0);
                let edge_label = if let Some(cut_value) = edge.cut_value {
                    format!("{cut_value}")
                } else {
                    "Null".to_string()
                };

                svg.push(format!(
                    r#"<text x="{label_x}" y="{label_y}" style="{font_style}">{edge_label}</text>"#
                ));
            }
        }

        for node in self.nodes.iter() {
            let x = node.horizontal_position.unwrap_or(DEFAULT_X) as f64;
            let y = node.vertical_rank.unwrap_or(DEFAULT_Y) as f64;
            let name = if debug {
                format!(
                    "{}: {:?}",
                    node.name,
                    node.coordinates
                        .unwrap_or(node::Point::new(DEFAULT_Y, DEFAULT_Y))
                        .x()
                )
            } else {
                node.name.to_string()
            };
            let font_size = if debug { 0.17 } else { 0.2 };
            let font_style = format!("font-size:{font_size}; text-anchor: middle");
            let label_x = x;
            let label_y = y + (font_size / 3.0);

            let show_node = !node.is_virtual() || debug;
            let node_radius = if show_node {
                real_radius
            } else {
                virtual_radius
            };
            let fill = if node.is_virtual() {
                "lightgray"
            } else {
                "skyblue"
            };
            let style = format!("fill: {fill}; stroke: black; stroke-width: {px_size}px;");

            if debug {
                let node_radius_x = node_radius * 1.5;

                svg.push(format!(
                    r#"<ellipse cx="{x}" cy="{y}" ry="{node_radius}" rx="{node_radius_x}" style="{style}"/>"#
                ));
            } else {
                svg.push(format!(
                    r#"<circle cx="{x}" cy="{y}" r="{node_radius}" style="{style}"/>"#
                ));
            }

            // svg.push(format!(
            //     r#"<svg viewBox="{rect_x} {rect_y} {rect_width} {rect_height}">"#
            // ));
            // svg.push(format!(
            //     r#"<rect x="0" y="0" height="1" width="1" rx=".001" style="{rect_style}"/>"#
            // ));
            if show_node {
                svg.push(format!(
                    r#"<text x="{label_x}" y="{label_y}" style="{font_style}">{name}</text>"#
                ));
            }
            // svg.push("</svg>".to_string());

            // if true {
            //     break;
            // }
        }

        svg.push("</svg>".to_string());

        svg.join("\n")
    }

    /// Write out the graph as is to the given file name (with an svg suffix).
    #[allow(unused)]
    fn write_svg_file(&self, name: &str, debug: bool) {
        use std::fs::File;
        use std::io::prelude::*;

        let svg = self.get_svg(debug);
        let mut file = File::create(format!("{name}.svg")).unwrap();
        file.write_all(svg.as_bytes()).unwrap();
    }

    /// Given a node index, return an iterator to a tuple with each edge_idx, and the node_idx of
    /// the node on the other side of the edge.
    ///
    /// Only returns non-ignored edges.
    fn get_node_edges_and_adjacent_node(
        &self,
        node_idx: usize,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let node = self.get_node(node_idx);

        node.get_all_edges_with_disposition()
            .filter_map(|(edge_idx, disposition)| {
                let edge_idx = *edge_idx;
                let edge = self.get_edge(edge_idx);

                if edge.ignored {
                    None
                } else {
                    let adjacent_node_idx = match disposition {
                        EdgeDisposition::In => edge.src_node,
                        EdgeDisposition::Out => edge.dst_node,
                    };
                    Some((edge_idx, adjacent_node_idx))
                }
            })
    }
}

impl Display for Graph {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        for (edge_id, edge) in self.edges.iter().enumerate() {
            let reversed = if edge.reversed { "reversed!" } else { "" };
            let in_tree = if edge.in_spanning_tree() {
                "TREE MEMBER"
            } else {
                "NOT IN TREE"
            };
            let src = &self.nodes[edge.src_node];
            let dst = &self.nodes[edge.dst_node];

            let line = if let Some(val) = edge.cut_value {
                format!(" {:2} ", val)
            } else if edge.in_spanning_tree() {
                "----".to_string()
            } else {
                " - -".to_string()
            };
            let src_rank = if let Some(rank) = src.simplex_rank {
                format!("r:{:2}", rank)
            } else {
                "r: -".to_string()
            };
            let dst_rank = if let Some(rank) = dst.simplex_rank {
                format!("r:{:2}", rank)
            } else {
                "r: -".to_string()
            };

            let _ = writeln!(
                fmt,
                "{src}({src_rank}) -{line}> {dst}({dst_rank}) eid:{edge_id} {in_tree} {reversed}",
            );
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
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
        fn configure_node(&mut self, name: &str, vertical_rank: u32) {
            let node_idx = self.name_to_node_idx(name).unwrap();
            let node = self.get_node_mut(node_idx);

            node.vertical_rank = Some(vertical_rank);
            node.simplex_rank = Some(vertical_rank);
            node.set_tree_root_node();
        }

        /// Get the edge that has src_node == src_name, dst_node == dst_name.
        ///
        /// Expensive for large data sets: O(e*n)
        pub fn get_named_edge(&self, src_name: &str, dst_name: &str) -> (&Edge, usize) {
            for (edge_idx, edge) in self.edges.iter().enumerate() {
                let src_node = self.get_node(edge.src_node);
                let dst_node = self.get_node(edge.dst_node);

                if src_node.name == src_name && dst_node.name == dst_name {
                    return (edge, edge_idx);
                }
            }
            panic!("Could not find requested edge: {src_name} -> {dst_name}");
        }

        pub fn example_graph_from_paper_2_3() -> Graph {
            Graph::from(
                "digraph {
                    a -> b; a -> e; a -> f;
                    e -> g; f -> g; b -> c;
                    c -> d; d -> h;
                    g -> h;
                }",
            )
        }

        pub fn example_graph_from_paper_2_3_extended() -> Graph {
            Graph::from(
                "digraph {
                    a -> b; a -> e; a -> f;
                    e -> g; f -> g; b -> c;
                    c -> d; d -> h;
                    g -> h;
                    a -> i; a -> j; a -> k;
                    i -> l; j -> l; k -> l;
                    l -> h;
                }",
            )
        }

        // Set the ranks /ngiven in example 2-3 (a)
        pub fn configure_example_2_3_a() -> (Graph, Vec<(&'static str, &'static str, i32)>) {
            let mut graph = Graph::example_graph_from_paper_2_3();
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
                    edge.set_in_spanning_tree(true);
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
        pub fn configure_example_2_3_b() -> (Graph, Vec<(&'static str, &'static str, i32)>) {
            let mut graph = Graph::example_graph_from_paper_2_3();
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
                edge.set_in_spanning_tree(!(edge.src_node == g_idx || edge.dst_node == f_idx));
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

        /// The two cut value examples given in the paper are too simple for
        /// more complex testing.  For example, there is only one negative cut value,
        /// thus network simplex never kicks in.
        ///
        /// This extended example adds a second cut value from l -> h which is -2, so
        /// there are two different cut values.
        pub fn configure_example_2_3_a_extended() -> (Graph, Vec<(&'static str, &'static str, i32)>)
        {
            let mut graph = Graph::example_graph_from_paper_2_3_extended();
            graph.configure_node("a", 0);
            graph.configure_node("b", 1);
            graph.configure_node("c", 2);
            graph.configure_node("d", 3);
            graph.configure_node("h", 4);

            graph.configure_node("e", 1);
            graph.configure_node("f", 1);
            graph.configure_node("g", 2);

            graph.configure_node("i", 1);
            graph.configure_node("j", 1);
            graph.configure_node("k", 1);
            graph.configure_node("l", 2);

            // Set feasible edges given in example 2-3 (a)
            let e_idx = graph.name_to_node_idx("e").unwrap();
            let f_idx = graph.name_to_node_idx("f").unwrap();
            let i_idx = graph.name_to_node_idx("i").unwrap();
            let j_idx = graph.name_to_node_idx("j").unwrap();
            let k_idx = graph.name_to_node_idx("k").unwrap();
            for edge in graph.edges.iter_mut() {
                if edge.dst_node != e_idx
                    && edge.dst_node != f_idx
                    && edge.dst_node != i_idx
                    && edge.dst_node != j_idx
                    && edge.dst_node != k_idx
                {
                    edge.set_in_spanning_tree(true);
                }
            }

            // cutvalues expected in example 2-3 (a)
            (
                graph,
                vec![
                    ("a", "b", 6),
                    ("b", "c", 6),
                    ("c", "d", 6),
                    ("d", "h", 6),
                    ("e", "g", 0),
                    ("f", "g", 0),
                    ("g", "h", -1),
                    ("l", "h", -2),
                ],
            )
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

        graph.rank_nodes_vertically();
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
        let mut graph = Graph::from("digraph { a -> b; b -> a; c -> d; c -> a; }");
        let node_c_idx = graph.name_to_node_idx("c").unwrap();

        graph.rank_nodes_vertically();

        let min_rank = graph.get_min_vertical_rank_nodes();
        let min_rank = min_rank.iter().cloned().collect::<Vec<usize>>();

        assert_eq!(min_rank, vec![node_c_idx], "min node should be 'c'");
    }

    #[test]
    fn test_get_initial_ordering() {
        let mut graph = Graph::from("digraph { a -> b; b -> c; b -> d; c -> e; d -> e }");

        graph.rank_nodes_vertically();

        println!("{graph}");
        let order = graph.get_initial_horizontal_orderings();

        println!("{order:?}");
    }

    #[test]
    fn test_rank() {
        let mut graph = Graph::example_graph_from_paper_2_3();

        graph.rank_nodes_vertically();

        println!("{graph}");
    }

    #[test]
    fn test_fill_rank_gaps() {
        use crate::graph::edge::MIN_EDGE_LENGTH;

        let (mut graph, _expected_cutvals) = Graph::configure_example_2_3_a();
        graph.init_cutvalues();
        let order = graph.get_initial_horizontal_orderings();

        println!("{graph}");
        graph.fill_vertical_rank_gaps(&order);
        println!("{graph}");

        for (edge_idx, _edge) in graph.edges.iter().enumerate() {
            if let Some(len) = graph.simplex_edge_length(edge_idx) {
                assert!(len.abs() <= MIN_EDGE_LENGTH as i32)
            }
        }
    }

    #[test]
    fn test_draw_graph() {
        let mut graph = Graph::example_graph_from_paper_2_3();

        println!("{graph}");
        graph.layout_nodes();
        println!("{graph}");
    }
}
