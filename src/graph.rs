//! Implement a graph that can be drawn using the algorithm described in the 1993 paper:
//! "A Technique for Drawing Directed Graphs" by Gansner, Koutsofios, North and Vo
//!
//! This paper is referred to as simply "the paper" below.

mod crossing_lines;
pub mod dot_parser;
mod edge;
mod network_simplex;
pub mod node;
mod rank_orderings;
pub mod snapshot;

use rank_orderings::RankOrderings;
use std::{
    cell::RefCell,
    collections::{HashMap, HashSet, VecDeque},
    fmt::Display,
    mem::replace,
};

use crate::svg::{SvgStyle, SVG};

use self::{
    edge::{
        Edge,
        EdgeDisposition::{self, In, Out},
        MIN_EDGE_LENGTH, MIN_EDGE_WEIGHT,
    },
    network_simplex::SimplexNodeTarget::{VerticalRank, XCoordinate},
    node::{Node, NodeType, Point, Rect, NODE_MIN_SEP_X, NODE_START_HEIGHT},
    rank_orderings::AdjacentRank, snapshot::Snapshots,
};

/// Simplest possible representation of a graph until more is needed.
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
#[derive(Debug, Clone)]
pub struct Graph {
    /// All nodes in the graph.
    nodes: Vec<Node>,
    /// All edges in the graph.
    edges: Vec<Edge>,
    /// Ordering of nodes by vertical rank.
    rank_orderings: Option<RankOrderings>,
    /// Separation of nodes horizontally in pixels, assuming 72 pixels per inch.
    horizontal_node_separation: u32,
    /// First node_idx in the graph that is virtual.  All subsequent nodes must be virtual too.
    first_virtual_idx: Option<usize>,
    /// Snapshots that are taken during the layout process for debugging.
    /// Snapshots are grouped into lists so the can be scrolled through
    /// somewhat hierarchically.
    svg_debug_snapshots: RefCell<Snapshots>,
    /// Must be true to enable snapshots.  False by default for performance reasons.
    snapshots_enabled: bool,
    /// Attributes for nodes.  For example, label="my_node_label"
    node_attr: HashMap<String, HashMap<String, String>>,
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
            rank_orderings: None,
            horizontal_node_separation: NODE_MIN_SEP_X as u32,
            first_virtual_idx: None,
            svg_debug_snapshots: RefCell::new(Snapshots::new()),
            snapshots_enabled: false,
            node_attr: HashMap::new(),
        }
    }

    /// Lays out the nodes in the graph as described in the graphviz dot paper:
    ///
    /// "A Technique for Drawing Directed Graphs"
    pub fn layout_nodes(&mut self) {
        self.rank_nodes_vertically();
        self.set_horizontal_ordering();
        self.set_coordinates();
        // self.make_splines()
    }

    /// Saves a snapshot of the current state of the graph to SVG.
    ///  * Used for debugging layout, specifically during min_cross
    ///  * To be used during vertical layout, the ranks would have to be
    ///    copied over from the aux_graph or other accommodation made.
    ///
    /// Can only be called after:
    ///  * nodes have been ranked vertically
    ///  * rank_orderings has been set in self.
    #[allow(unused)]
    pub(super) fn take_svg_snapshot(&self, title: &str, rank_orderings: Option<&RankOrderings>) {
        if self.snapshots_enabled {
            let mut snapshot_graph = self.clone();
            if let Some(rank_orderings) = rank_orderings {
                snapshot_graph.set_coordinates_from_rank_orderings(rank_orderings);
            }

            let svg = SVG::new(snapshot_graph, SvgStyle::MinCross);

            self.svg_debug_snapshots
                .borrow_mut()
                .add(title, &svg.to_string());
        }
    }

    #[allow(unused)]
    pub(super) fn new_snapshot_group(&self, title: &str) {
        self.svg_debug_snapshots.borrow_mut().new_group(title);
    }

    pub fn get_debug_svg_snapshots(&self) -> Snapshots {
        self.svg_debug_snapshots.borrow().clone()
    }

    /// Enables debug snapshots.  On complex graphs, this
    /// can slow things down quite a bit.
    pub(super) fn enable_snapshots(&mut self) {
        self.snapshots_enabled = true;
    }

    /// Gives a HashMap of node names and attributes, assign them.
    pub fn set_node_attr(&mut self, attr: HashMap<String, HashMap<String, String>>) {
        for (name, attrs) in attr.iter() {
            self.node_attr.insert(name.clone(), attrs.clone());
        }
    }

    // fn get_node_name_map(&self) -> HashMap<String, usize> {
    //     let mut name_map = HashMap::new();

    //     for (node_idx, node) in self.nodes.iter().enumerate() {
    //         name_map.insert(node.name.clone(), node_idx);
    //     }
    //     name_map
    // }

    /// Return the label of a node, if there is any.
    pub fn get_node_label(&self, node_name: &str) -> Option<&String> {
        self.node_attr
            .get(node_name)
            .and_then(|attr| attr.get("label"))
    }

    fn set_coordinates_from_rank_orderings(&mut self, rank_orderings: &RankOrderings) {
        for (rank, _) in rank_orderings.iter() {
            for (position, node_idx) in rank_orderings
                .rank_to_positions(*rank)
                .expect("rank does not exit")
                .iter()
                .cloned()
                .enumerate()
            {
                let position = position as i32;
                let node = self.get_node_mut(node_idx);

                node.set_coordinates(position * NODE_MIN_SEP_X, rank * node.min_separation_y())
            }
        }
    }

    pub fn horizontal_node_separation(&self) -> u32 {
        self.horizontal_node_separation
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

    /// Return the top left and bottom right coordinates in the graph.
    pub fn graph_rect(&self) -> Rect {
        let mut min = Point::new(i32::MAX, i32::MAX);
        let mut max = Point::new(i32::MIN, i32::MIN);

        if self.node_count() == 0 {
            panic!("No nodes in graph");
        }

        for node in self.nodes.iter() {
            if let Some(coords) = node.coordinates {
                max.set_x(max.x().max(coords.x()));
                max.set_y(max.y().max(coords.y()));

                min.set_y(min.x().min(coords.x()));
                min.set_x(min.x().min(coords.x()));
            } else {
                panic!("All nodes must have coordinates");
            }
        }
        Rect::new(min, max)
    }

    // Return the number of vertical ranks.
    fn num_ranks(&self) -> Option<usize> {
        self.rank_orderings.as_ref().map(|order| order.num_ranks())
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
    pub fn add_virtual_node(&mut self, rank: i32, node_type: NodeType) -> usize {
        if !node_type.is_virtual() {
            panic!("Cannot add read node as a virtual node");
        }

        let name = format!("v{}", self.node_count());
        let idx = self.add_node(&name);

        if self.first_virtual_idx.is_none() {
            self.first_virtual_idx = Some(idx);
        }

        let v_node = self.get_node_mut(idx);

        v_node.node_type = node_type;
        v_node.set_simplex_rank(Some(rank));
        v_node.vertical_rank = Some(rank);

        idx
    }

    /// Add a new edge between two nodes, and return the edge's index in the graph.
    ///
    /// Weight is set to the minimum edge weight.
    pub fn add_edge(&mut self, src_node: usize, dst_node: usize) -> usize {
        self.add_edge_with_details(src_node, dst_node, MIN_EDGE_LENGTH, MIN_EDGE_WEIGHT)
    }

    /// Add a new edge between two nodes with a given weight, and return the edge's index in the graph.
    pub fn add_edge_with_details(
        &mut self,
        src_node: usize,
        dst_node: usize,
        min_len: i32,
        weight: u32,
    ) -> usize {
        let new_edge = Edge::new_with_details(src_node, dst_node, min_len, weight);
        let idx = self.edges.len();
        self.edges.push(new_edge);

        self.nodes[src_node].add_edge(idx, Out);
        self.nodes[dst_node].add_edge(idx, In);

        // println!("Added edge: {}", self.edge_to_string(idx));

        idx
    }

    /// Rank nodes in the tree using the network simplex algorithm.
    pub fn rank_nodes_vertically(&mut self) {
        self.make_acyclic();
        self.merge_edges();
        self.network_simplex_ranking(VerticalRank);
    }

    /// Make the graph acyclic by reversing edges to previously visited nodes.
    ///
    /// From Paper section 2.1: Making the graph acyclic (page 6)
    ///
    /// * A useful procedure for breaking cycles is based on depth-first search.
    /// * Edges are searched in the "natural order" of the graph input, starting
    ///   from some source or sink nodes if any exist.
    /// * Depth-first search partitions edges into two sets: tree edges and non-tree
    ///   edges [AHU].
    /// * The tree defines a partial order on nodes.
    /// * Given this partial order, the non-tree edges
    ///   further partition into three sets: cross edges, forward edges, and back edges.
    /// * Cross edges connect unrelated nodes in the partial order.
    ///   * Forward edges connect a node to some of its descendants.
    ///   * Back edges connect a descendant to some of its ancestors.
    /// * It is clear that adding forward and cross edges to the partial order does not
    ///   create cycles.
    /// * Because reversing back edges makes them into forward edges, all cycles are
    ///   broken by this procedure.
    fn make_acyclic(&mut self) {
        // self.print_nodes("before make_acyclic()");
        self.ignore_node_loops();

        let mut visited = vec![false; self.node_count()];
        let mut rec_stack = vec![false; self.node_count()];

        // Start with the source nodes
        for node_idx in self.get_source_nodes().iter().cloned() {
            self.make_acyclic_worker(node_idx, &mut visited, &mut rec_stack)
        }

        // In case we didn't visit anybody (e.g. there might be no source nodes
        // as the graph starts as potentially cyclic)
        for node_idx in 0..self.node_count() {
            if !visited[node_idx] {
                self.make_acyclic_worker(node_idx, &mut visited, &mut rec_stack)
            }
        }

        // self.print_nodes("after make_acyclic()");
    }

    /// On the given node
    /// * Mark the node as visited, and mark it on the recursive stack
    /// * Do a depth first search of it's outbound edges
    ///   * If an edge has not yet been visited, visit it and its children recursively
    ///   * If an edge is currently on the stack, reverse the edge
    ///     (because this would cause a cycle)
    /// * Remove the node from the recursive stack.
    fn make_acyclic_worker(
        &mut self,
        node_idx: usize,
        visited: &mut [bool],
        rec_stack: &mut [bool],
    ) {
        if !visited[node_idx] {
            visited[node_idx] = true;
            rec_stack[node_idx] = true;

            let out_edges = self.get_node(node_idx).get_edges(Out).clone();
            for edge_idx in out_edges {
                let dst_node_idx = self.get_edge(edge_idx).dst_node;

                if !visited[dst_node_idx] {
                    self.make_acyclic_worker(dst_node_idx, visited, rec_stack);
                } else if rec_stack[dst_node_idx] {
                    self.reverse_edge(edge_idx);
                }
            }
        }

        rec_stack[node_idx] = false;
    }

    /// Ignore individual edges that loop to and from the same node.
    fn ignore_node_loops(&mut self) {
        self.edges
            .iter_mut()
            .filter(|edge| edge.src_node == edge.dst_node)
            .for_each(|edge| edge.ignored = true);
    }

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

    fn get_sink_nodes(&self) -> VecDeque<usize> {
        let mut queue = VecDeque::new();

        for (node_idx, _node) in self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_i, n)| n.no_out_edges())
        {
            queue.push_back(node_idx);
        }
        queue
    }

    fn reverse_edge(&mut self, edge_idx_to_reverse: usize) {
        let (src_node_idx, dst_node_idx) = {
            let edge = self.get_edge(edge_idx_to_reverse);

            (edge.src_node, edge.dst_node)
        };

        // Swap the references in src and dst nodes
        // println!(
        //     "SWAPPING from {} to {}",
        //     self.get_node(src_node_idx).name,
        //     self.get_node(dst_node_idx).name
        // );
        self.get_node_mut(src_node_idx)
            .swap_edge_in_list(edge_idx_to_reverse, Out);
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

    /// Run thr horizontal ordering algorithm in two different configurations and choose the best
    /// * The best being the one with the fewest line crosses between ranks.
    ///
    /// From the Paper: page 17 section 3: Vertex Ordering Within Ranks
    ///
    /// One final point is that it is generally worth the extra cost to run the vertex ordering algorithm twice:
    /// once for an initial order determined by starting with vertices of minimal rank and searching out-edges,
    /// and the second time by starting with vertices of maximal rank and searching in-edges.  This allows one
    /// to pick the better of two different solutions.
    ///
    fn set_horizontal_ordering(&mut self) -> &RankOrderings {
        self.set_horizontal_ordering_with_direction(true);
        // self.set_horizontal_ordering_with_direction(false);

        self.rank_orderings.as_ref().unwrap()
    }

    /// Set the rank_orderings field of graph and set the horizontal position of each node.
    ///
    /// TODO: Note that the horizontal position of each node is not used.  It is taken from the
    ///       rank_orderings again later.  This should be cleaned up once debugging is complete.
    ///
    /// GraphViz: essentially the function: dot_mincross(), though this does more.
    ///
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
    fn set_horizontal_ordering_with_direction(&mut self, min: bool) -> &RankOrderings {
        const MAX_ITERATIONS: usize = 24;
        let order = self.init_horizontal_order(min);
        let mut best = order.clone();

        self.new_snapshot_group("initial");
        self.take_svg_snapshot(
            &format!("initial ordering: cc={}", order.crossing_count()),
            Some(&order),
        );

        if best.crossing_count() != 0 {
            for i in 0..MAX_ITERATIONS {
                let cur_crossing_count = order.crossing_count();
                let best_crossing_count = best.crossing_count();

                self.new_snapshot_group(&format!("ordering pass: {i} cur_count={cur_crossing_count} best_count={best_crossing_count}"));
                println!("Ordering pass {i}: cross count: {cur_crossing_count}");
                let forward = i % 2 == 0;
                let exchange_if_equal = i % 3 < 2;

                order.weighted_median(forward, exchange_if_equal, self);
                // println!("weighed_median: {}", order.crossing_count());
                // self.take_svg_snapshot("after weighted median", Some(&order));

                self.new_snapshot_group(&format!("ordering pass: {i}, transpose"));
                order.transpose(exchange_if_equal, Some(self));
                // println!(
                //     "  After transpose ({exchange_if_equal}): {}",
                //     order.crossing_count()
                // );
                // self.take_svg_snapshot("after transpose", Some(&order));

                let new_crossing_count = order.crossing_count();
                if new_crossing_count < best.crossing_count() {
                    best = order.clone();

                    if new_crossing_count == 0 {
                        break;
                    }
                }
            }
        } else {
            println!("No reason to reduce crosses starting with zero...");
        }
        println!(
            "-- Final order (crosses: {}): --\n{best}",
            best.crossing_count()
        );
        self.new_snapshot_group("final");
        self.take_svg_snapshot(
            &format!("final ordering: cc={}", best.crossing_count()),
            Some(&best),
        );

        if self.should_update_ordering(&best) {
            self.set_node_positions(&best);

            self.rank_orderings = Some(best);
        }
        self.print_nodes("After set_horizontal_ordering (dot_mincross())");

        self.rank_orderings.as_ref().unwrap()
    }

    /// Return true if the given new_ordering has fewer crosses than the current.
    fn should_update_ordering(&self, new_ordering: &RankOrderings) -> bool {
        if let Some(cur_count) = self.cur_crossing_count() {
            cur_count > new_ordering.crossing_count()
        } else {
            true
        }
    }

    /// If a rank_orderings has been set, return it's crossing count, otherwise None.
    fn cur_crossing_count(&self) -> Option<u32> {
        self.rank_orderings
            .as_ref()
            .map(|ordering| ordering.crossing_count())
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
    fn init_horizontal_order(&mut self, min: bool) -> RankOrderings {
        let order = self.get_initial_horizontal_orderings(min);

        self.fill_vertical_rank_gaps(&order);
        self.set_adjacent_nodes_in_vertical_ranks(&order);

        order
    }

    /// Edges between nodes more than one rank apart are replaced by chains of virtual nodes.
    ///
    /// After running, no edge spans more than one rank.
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

    // Replace the given edge with a chain virtual nodes and edges.
    //
    // * rank is the rank where the edge starts
    // * slack is the extra distance between ranks.  When slack is zero, they are only
    //   one rank apart.  If slack is negative, the destination node is up the ranks
    //   instead of down the ranks.
    fn replace_edge_with_virtual_chain(
        &mut self,
        edge_idx: usize,
        rank: i32,
        slack: i32,
        order: &RankOrderings,
    ) {
        let mut remaining_slack = slack;
        let up_edge = slack < 0;

        let mut cur_edge_idx = edge_idx;
        let mut new_rank = rank;
        while remaining_slack != 0 {
            new_rank = if up_edge { new_rank - 1 } else { new_rank + 1 };

            let virt_node_idx = self.add_virtual_node(new_rank, NodeType::RankFiller);

            let old_edge = self.get_edge_mut(cur_edge_idx);
            let old_edge_reversed_for_acyclic = old_edge.reversed;
            let orig_dst = replace(&mut old_edge.dst_node, virt_node_idx);

            self.get_node_mut(virt_node_idx).add_edge(cur_edge_idx, In);

            // The old edge needs to be removed from the original dst_node.
            self.get_node_mut(orig_dst).remove_edge(cur_edge_idx, In);

            cur_edge_idx = self.add_edge(virt_node_idx, orig_dst);
            // If the edge was reversed to keep the graph acyclic, all the edges we add
            // should be reversed in this way as well.
            if old_edge_reversed_for_acyclic {
                self.get_edge_mut(cur_edge_idx).reversed = true;
            }

            order.add_node_idx_to_existing_rank(new_rank, virt_node_idx);

            remaining_slack += if up_edge { 1 } else { -1 };
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
    ///  * Do a breadth first search by following edges that point to nodes that
    ///    have not yet been assigned an ordering
    ///    * Pull a node off of the queue.  If it has not yet been assigned:
    ///      * Add it to the rank_order BTreeMap under it's given rank
    ///      * Mark it assigned
    ///      * push the all the unassigned nodes the node's edges point to on the back of the queue
    /// * Continue until the queue in empty and return the rank order.
    ///
    /// NOTE: I initially implemented this as a depth first search.  But a depth first search
    ///       sometimes reverses the initial ranking implied by the user's input string.
    ///       Thus could cause incorrect horizontal coordinates in the assign coordinate phase.
    ///       I am unclear what causes this side effect, but switching to a breadth first search
    ///       resolved it.  Hopefully I can figure out why!
    ///
    /// Documentation from paper: page 14
    /// init_order initially orders the nodes in each rank.
    /// * This may be done by a depth-first or breadth-first search starting with vertices of minimum rank.
    ///   * Vertices are assigned positions in their ranks in left-to-right order as the search progresses.
    ///     * This strategy ensures that the initial ordering of a tree has no crossings.
    ///     * This is important because such crossings are obvious, easily- avoided "mistakes."
    fn get_initial_horizontal_orderings(&mut self, min: bool) -> RankOrderings {
        let mut rank_order = RankOrderings::new();
        let mut dfs_queue = self.get_vertical_rank_nodes(min);
        let mut assigned = HashSet::new();

        // println!("Starting Queue (min={min}): {dfs_queue:?}");

        while let Some(node_idx) = dfs_queue.pop_front() {
            let node = self.get_node(node_idx);
            let node_iter = if min {
                node.out_edges.iter()
            } else {
                node.in_edges.iter()
            };
            let unassigned_nodes = node_iter
                .cloned()
                .filter_map(|edge_idx| {
                    let edge = self.get_edge(edge_idx);
                    let other_node = if min { edge.dst_node } else { edge.src_node };

                    if edge.ignored {
                        None
                    } else if assigned.get(&other_node).is_none() {
                        Some(other_node)
                    } else {
                        None
                    }
                })
                .collect::<Vec<usize>>();

            // println!("Ordering node: {node_idx}");
            if assigned.get(&node_idx).is_none() {
                let rank = node
                    .vertical_rank
                    .expect("All nodes must have a vertical rank");
                rank_order.add_node_idx_to_rank(rank, node_idx);
                assigned.insert(node_idx);
                // println!("  Assigned node: {node_idx}");
            }

            for node_idx in unassigned_nodes {
                dfs_queue.push_back(node_idx);
            }
        }
        rank_order
    }

    /// The graph is responsible for setting adjacent nodes in the rank_order once all nodes have been added to it.
    fn set_adjacent_nodes_in_vertical_ranks(&self, rank_order: &RankOrderings) {
        for (node_idx, _node_position) in rank_order.nodes().borrow().iter() {
            let (above_adj, below_adj) = self.get_vertical_adjacent_nodes(*node_idx);

            rank_order.set_adjacent_nodes(*node_idx, &above_adj, &below_adj);
        }
    }

    /// Return a VecDequeue of nodes which have minimum or maximum rank.
    ///
    /// * Assumes that the graph has been ranked
    /// * Since this will be used to populate ranks, we must
    ///   include all source/sink nodes (nodes with no in/out edges) otherwise
    ///   they will not be ranked.  The initial vertical ranking might
    ///   not rank all source nodes as rank zero, so we need to check
    ///   that they are all included.
    ///   * We use source nodes if we are starting with min rank, otherwise
    ///     we use sink nodes.
    fn get_vertical_rank_nodes(&self, min: bool) -> VecDeque<usize> {
        let mut edge_rank_nodes = VecDeque::new();
        let node_iter = self.nodes.iter();
        let edge_rank = if min {
            node_iter.min().and_then(|min_node| min_node.vertical_rank)
        } else {
            node_iter.max().and_then(|max_node| max_node.vertical_rank)
        };
        let mut starting_nodes: HashSet<usize> = if min {
            HashSet::from_iter(self.get_source_nodes().iter().cloned())
        } else {
            HashSet::from_iter(self.get_sink_nodes().iter().cloned())
        };

        for (node_idx, node) in self.nodes.iter().enumerate() {
            if node.vertical_rank == edge_rank {
                edge_rank_nodes.push_back(node_idx);
                starting_nodes.remove(&node_idx);
            }
        }

        // Include any source nodes that have not yet been included.
        for node_idx in starting_nodes {
            edge_rank_nodes.push_back(node_idx);
        }

        edge_rank_nodes
    }

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
    /// * The method involves constructing an auxiliary graph as illustrated in figure 4-2.
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
    ///      * This edge forces the nodes to be sufficiently separated but does not affect the cost of the layout.
    /// * We can now consider the level assignment problem on G′, which can be solved using the network
    ///   simplex method.
    ///   * Any solution of the positioning problem on G corresponds to a solution of the level assignment
    ///     problem on G′ with the same cost. This is achieved by assigning each new_v_node the value
    ///     min (x_u, x_v), using the notation of figure 4-2 and where x_u and x_v are the X coordinates assigned
    ///     to u and v in G.
    ///   * Conversely, any level assignment in G′ induces a valid positioning in G. In addition, in an optimal
    ///     level assignment, one of e_u or e_v must have length 0, and the other has length |x_u − x_v|. This
    ///     means the cost of an original edge (u ,v) in G equals the sum of the cost of the two edges e_u, e_v
    ///     in G′ and, globally, the two solutions have the same cost.
    ///   * Thus, optimality of G′ implies optimality for G and solving G′ gives us a solution for G.
    fn set_coordinates(&mut self) {
        self.set_y_coordinates();

        let mut aux_graph = self.create_positioning_aux_graph();

        // This is necessary as part of initializing tree nodes.
        // XXX This should be more obvious in the name, because otherwise set_feasible_ranking can crash.
        aux_graph.make_acyclic();

        aux_graph.network_simplex_ranking(XCoordinate);
        self.set_x_coordinates_from_aux(&aux_graph);

        // self.print_nodes("after set_coordinates in network_simplex_ranking (dot_position())");
    }

    fn set_x_coordinates_from_aux(&mut self, aux_graph: &Graph) {
        // Technically, once we find the first XCoordCalc node, we can exit, since they are all added
        // after other nodes.  At least, that's true now...
        for (node_idx, node) in aux_graph
            .nodes
            .iter()
            .enumerate()
            .filter(|(_idx, node)| node.node_type != NodeType::XCoordCalc)
        {
            let x = node.coordinates.unwrap().x();

            self.get_node_mut(node_idx).assign_x_coord(x);
        }
    }

    /// Set the y coordinates of all nodes.
    ///
    /// TODO: this currently set's x and y coordinates, but the x coordinate
    ///       will be overwritten later.  This should be cleaned up.
    fn set_y_coordinates(&mut self) {
        // We set the y coordinate in reverse rank to match what GraphViz does.
        let _num_ranks = self.num_ranks().unwrap();
        for (node_idx, node) in self.nodes.iter_mut().enumerate() {
            let min_y = node.min_separation_y();
            let new_y = if let Some(rank) = node.vertical_rank {
                // XXX
                // GraphViz code does this backwards, and later flips it back...
                // but no need for us to do that foolishness.
                // ((_num_ranks as i32 - 1) - rank) * min_y + NODE_START_HEIGHT
                rank * min_y + NODE_START_HEIGHT
            } else {
                panic!("Node {node_idx}: rank not set");
            };

            node.assign_y_coord(new_y);
        }
        // self.print_nodes("after set_y_coordinates()");
    }

    /// Documentation from paper: 4.2 Optimal Node Placement page 20
    /// * The method involves constructing an auxiliary graph as illustrated in figure 4-2.
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
    ///      * This edge forces the nodes to be sufficiently separated but does not affect the cost of the layout.
    ///      
    ///  QUESTIONS: ☑ ☐ ☒
    ///  * For the new graph G':
    ///    * Are virtual nodes copied too? (almost certainly yes)
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
    ///   * This is achieved by assigning each n_e the value min (x_u, x_v), using the notation of figure 4-2
    ///     and where x_u and x_v are the X coordinates assigned to u and v in G.
    ///   * Conversely, any level assignment in G′ induces a valid positioning in G.
    ///   * In addition, in an optimal level assignment, one of e_u or e must have length 0, and the other
    ///     has length |x_u − x_v|. This means the cost of an original edge (u, v) in G equals the sum
    ///     of the cost of the two edges e_y, e_v in G′ and, globally, the two solutions have the same cost.
    ///   * Thus, optimality of G′ implies optimality for G and solving G ′ gives us a solution for G.
    fn create_positioning_aux_graph(&self) -> Graph {
        let mut aux_graph = Graph::new();

        // self.print_nodes("before create_positioning_aux_graph() (create_aux_edges)");

        // Add all the existing nodes from the current graph without edges.
        for node in self.nodes.iter() {
            let mut new_node = node.clone();
            new_node.clear_edges();
            // let new_node = Node::new(node.name());

            aux_graph.nodes.push(new_node)
        }

        // aux_graph.print_nodes("before set_left_right_constraints() (make_LR_constraints)");
        self.set_left_right_constraints(&mut aux_graph);
        // aux_graph.print_nodes("after set_left_right_constraints()");

        self.add_virtual_nodes_for_horizontal_positioning(&mut aux_graph);
        // aux_graph.print_nodes("after add_virtual_nodes_for_horizontal...() (make_edge_pairs)");

        // We need to build a feasible tree from scratch
        for node in aux_graph.nodes.iter_mut() {
            node.clear_tree_data();
        }

        aux_graph
    }

    /// Add the new node and edges between the new node, as per figure 4-2 in paper
    ///
    /// Note that we are adding 2 edges and one node for every edge, but not the original edge itself.
    fn add_virtual_nodes_for_horizontal_positioning(&self, aux_graph: &mut Graph) {
        for edge in self.edges.iter() {
            let src_node_idx = edge.src_node;
            let dst_node_idx = edge.dst_node;
            // Must use aux_graph because it is already different from self
            let src_node = aux_graph.get_node(src_node_idx).clone();
            let dst_node = aux_graph.get_node(dst_node_idx).clone();

            // As well, must use simplex_rank() instead of vertical_rank, because it may have been
            // changed.
            let src_rank = src_node.simplex_rank().unwrap();
            let dst_rank = dst_node.simplex_rank().unwrap();
            let new_rank = (src_rank - 1).min(dst_rank - 1);

            let new_node_idx = aux_graph.add_virtual_node(new_rank, NodeType::XCoordCalc);
            //let new_node = aux_graph.get_node(new_node_idx);
            //println!("Adding virtual node {} with rank {:?}: {src_rank} from {} and {dst_rank} from {}", new_node.name, new_node.simplex_rank(), src_node.name, dst_node.name);
            //println!("COMPARE Self\n{self}To Aux:\n{aux_graph}");

            // The paper described setting the new weight edge.weight*omega, but the GraphViz code does
            // not seem to do this, and just passes on the weight of the existing edge:
            //
            // let omega = Edge::edge_omega_value(src_node.is_virtual(), dst_node.is_virtual());
            // let new_weight = edge.weight * omega;
            let new_weight = edge.weight;

            aux_graph.add_edge_with_details(
                new_node_idx,
                src_node_idx,
                MIN_EDGE_LENGTH,
                new_weight,
            );
            aux_graph.add_edge_with_details(
                new_node_idx,
                dst_node_idx,
                MIN_EDGE_LENGTH,
                new_weight,
            );

            // GraphViz code lets these coords just be zero
            let new_node = aux_graph.get_node_mut(new_node_idx);
            new_node.set_coordinates(0, 0);
            // The paper seems to say something different...
            // // TODO: deal with unwraps();
            // let src_x = src_node.coordinates.unwrap().x();
            // let src_y = src_node.coordinates.unwrap().y();
            // let dst_x = dst_node.coordinates.unwrap().x();

            // // ...assigning each new node n_e the value min(x_u, x_v), using the notation of figure 4-2
            // // and where x_u and x_v are the X coordinates assigned to u and v in G.
            // new_node.set_coordinates(src_x.min(dst_x), src_y);
        }
    }

    /// Add constraints to nodes in preparation for ranking nodes horizontally.
    /// * Set ranks to be based on horizontal_node_separation: usually 72 pixes per inch.
    /// * Add edges between adjacent nodes in a rank, as per figure 4-2 in the paper.
    ///
    /// This allows the nodes to be properly spaced during network simplex ranking.
    ///
    /// Note: "flat" edges are edges that connect two nodes on the same rank.
    ///
    /// In GraphViz code: make_LR_constraints() (which does more as well)
    fn set_left_right_constraints(&self, aux_graph: &mut Graph) {
        let node_sep = self.horizontal_node_separation; // Same as GraphViz because 72 dpi is standard.

        // make edges to separate nodes on the same rank
        //
        // But also, critically, switch the simplex rank on AuxGraph to be the x coordinate:
        // * Start with the position
        // * Multiply by node separation
        //
        if let Some(rank_orderings) = self.rank_orderings.as_ref() {
            for rank in 0..rank_orderings.num_ranks() as i32 {
                let mut prev_rank = None;
                let mut prev_node_idx = None;
                // To do this correctly, we must step through the nodes in each
                // rank ordered by their positions which we calculated previously.
                let rank_by_position = rank_orderings
                    .rank_to_positions(rank)
                    .expect("all rank have position");

                for node_idx in rank_by_position.iter().cloned() {
                    let new_rank: i32 = if let Some(prev_rank) = prev_rank {
                        prev_rank + node_sep as i32
                    } else {
                        0
                    };
                    if let Some(prev_node_idx) = prev_node_idx {
                        aux_graph.add_edge_with_details(
                            prev_node_idx,
                            node_idx,
                            node_sep as i32,
                            0,
                        );
                    }

                    let node = self.get_node(node_idx);
                    assert_eq!(Some(rank), node.simplex_rank());
                    // println!(
                    //     "rank{rank}: Setting rank for node {}: pos:{:?} to: {new_rank}",
                    //     node.name, node.horizontal_position
                    // );

                    aux_graph
                        .get_node(node_idx)
                        .set_simplex_rank(Some(new_rank));

                    prev_node_idx = Some(node_idx);
                    prev_rank = Some(new_rank);
                }
            }
        }
    }

    #[allow(unused)]
    fn make_splines(&mut self) {
        todo!();
    }

    /// Given a node index, return an iterator to a tuple with each edge_idx, and the node_idx of
    /// the node on the other side of the edge.
    ///
    /// Only returns non-ignored edges.
    fn get_node_edges_and_adjacent_node(
        &self,
        node_idx: usize,
        out_first: bool,
    ) -> impl Iterator<Item = (usize, usize)> + '_ {
        let node = self.get_node(node_idx);

        node.get_all_edges_with_disposition(out_first)
            .filter_map(|(edge_idx, disposition)| {
                let edge_idx = *edge_idx;
                let edge = self.get_edge(edge_idx);

                if edge.ignored {
                    None
                } else {
                    let adjacent_node_idx = match disposition {
                        In => edge.src_node,
                        Out => edge.dst_node,
                    };
                    Some((edge_idx, adjacent_node_idx))
                }
            })
    }

    pub fn edges_iter(&self) -> impl Iterator<Item = &Edge> {
        self.edges.iter()
    }

    pub fn nodes_iter(&self) -> impl Iterator<Item = &Node> {
        self.nodes.iter()
    }

    /// Return a vector of node indexes, starting with the virtual nodes
    /// listed in reverse order they were added, followed by the non-virtual
    /// nodes in the order they were added.
    ///
    /// This mirrors the order of nodes in GraphViz (which is most useful for display)
    ///
    /// If we are relying on this function for something other than display, it will mean
    /// we are brittle because it can break if we add nodes in a different order that GraphViz.
    fn node_indexes_in_graphviz_order(&self) -> Vec<usize> {
        if let Some(first_virtual_idx) = self.first_virtual_idx {
            let mut v1 = (first_virtual_idx..self.node_count())
                .rev()
                .collect::<Vec<usize>>();
            let mut v2 = (0..first_virtual_idx).collect::<Vec<usize>>();

            v1.append(&mut v2);

            v1
        } else {
            (0..self.node_count()).collect::<Vec<usize>>()
        }
    }

    /// An iterator that returns only tree edges (not all graph edges).
    fn tree_edge_iter(&self) -> impl Iterator<Item = (usize, &Edge)> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| !edge.ignored && edge.in_spanning_tree())
    }

    #[allow(dead_code)]
    fn not_tree_edge_iter(&self) -> impl Iterator<Item = (usize, &Edge)> {
        self.edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| !edge.ignored && !edge.in_spanning_tree())
    }

    /// An iterator that returns only nodes that are "real" (not virtual nodes added
    /// for layout)
    #[allow(dead_code)]
    fn real_nodes_iter(&self) -> impl Iterator<Item = (usize, &Node)> {
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| node.node_type == NodeType::Real)
    }

    #[allow(dead_code)]
    fn print_edge(&self, edge_idx: usize) {
        println!("{}", self.edge_to_string(edge_idx));
    }

    fn edge_to_string(&self, edge_idx: usize) -> String {
        let edge = self.get_edge(edge_idx);
        let src_node = self.get_node(edge.src_node);
        let dst_node = self.get_node(edge.dst_node);
        let in_tree = if edge.in_spanning_tree() {
            ""
        } else {
            "NOT IN TREE"
        };
        let ignored = if edge.ignored { "IGNORED" } else { "" };
        let cut_value = if let Some(cut_value) = edge.cut_value {
            format!("{cut_value}")
        } else {
            "None".to_string()
        };
        let slack = if let Some(slack) = self.simplex_slack(edge_idx) {
            format!("{slack}")
        } else {
            "None".to_string()
        };

        format!(
            "{ignored}{: >3} ->{: >3}: ({edge_idx}) cut_value: {cut_value} slack: {slack} weight: {} min_len: {} {in_tree}",
            src_node.name,
            dst_node.name,
            edge.weight,
            edge.min_len()
        )
    }

    fn node_to_string(&self, node_idx: usize) -> String {
        let node = self.get_node(node_idx);
        let coords = node
            .coordinates
            .map(|coords| format!("({},{})", coords.x(), coords.y()))
            .unwrap_or("None".to_string());
        let tree_node = if let Some(tree_data) = node.spanning_tree() {
            let parent = if let Some(parent_idx) = node.spanning_tree_parent_edge_idx() {
                if parent_idx < self.edges.len() {
                    let edge = self.get_edge(parent_idx);
                    let parent_node = if edge.src_node == node_idx {
                        self.get_node(edge.dst_node)
                    } else {
                        self.get_node(edge.src_node)
                    };
                    parent_node.name()
                } else {
                    "+"
                }
            } else {
                "-"
            };
            let sub_tree = if let Some(sub_tree) = &node.sub_tree() {
                sub_tree.to_string()
            } else {
                "NO SUB_TREE".to_string()
            };

            format!("TreeParent:{parent} {sub_tree} {tree_data}")
        } else {
            "NOT IN TREE".to_string()
        };
        format!(
            "{node_idx}: {}: type={:?} rank={:?} x_pos:{:?} coord:{coords} {tree_node}",
            node.name,
            node.node_type,
            node.simplex_rank(),
            node.horizontal_position,
        )
    }

    pub fn print_nodes(&self, title: &str) {
        println!("-{title}: NODES----------------");
        println!("{self}----------------------");
    }
}

impl Display for Graph {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        for node_idx in self.node_indexes_in_graphviz_order() {
            let node = self.get_node(node_idx);
            let _ = writeln!(fmt, "{}", self.node_to_string(node_idx));
            for (edge_idx, disp) in node.get_all_edges_with_disposition(false) {
                if !self.get_edge(*edge_idx).ignored {
                    let _ = writeln!(fmt, "    {disp: <8}: {}", self.edge_to_string(*edge_idx));
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    use crate::dot_examples::dot_example_graph;

    use super::*;
    use std::ops::RangeInclusive;
    use tests::edge::MIN_EDGE_WEIGHT;

    /// Additional test only functions for Graph to make graph construction testing easier.
    impl Graph {
        /// Add multiple nodes with names given from a range of characters.
        ///
        /// Returns a HashMap of the created node names and node indexes to be
        /// used with add_edges().
        ///
        /// * Nodes must be named after a single character.
        /// * The range is inclusive only of the left side.  So 'a'..'d' includes: a, b, c but NOT d.
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
        fn configure_node(&mut self, name: &str, vertical_rank: i32) {
            let node_idx = self.name_to_node_idx(name).unwrap();
            let node = self.get_node_mut(node_idx);

            node.vertical_rank = Some(vertical_rank);
            node.set_simplex_rank(Some(vertical_rank));
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
            dot_example_graph("tse_paper/example_2_3")
        }

        pub fn example_graph_from_paper_2_3_extended() -> Graph {
            dot_example_graph("tse_paper/example_2_3_extended")
        }

        // Set the ranks given in example 2-3 (a)
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

    /// Test that two simple cyclic graphs are both made acyclic.
    #[test]
    fn test_make_acyclic() {
        let mut graph = Graph::new();

        let node_map = graph.add_nodes('a'..='d');
        let edges = vec![("a", "b"), ("b", "a"), ("c", "d"), ("d", "c")];
        graph.add_edges(&edges, &node_map);

        graph.make_acyclic();

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

        let min_rank = graph.get_vertical_rank_nodes(true);
        let min_rank = min_rank.iter().cloned().collect::<Vec<usize>>();

        assert_eq!(min_rank, vec![node_c_idx], "min node should be 'c'");
    }

    #[test]
    fn test_get_initial_ordering() {
        let mut graph = Graph::from("digraph { a -> b; b -> c; b -> d; c -> e; d -> e }");

        graph.rank_nodes_vertically();

        println!("{graph}");
        let order = graph.get_initial_horizontal_orderings(true);

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
        graph.init_spanning_tree_and_cutvalues();
        let order = graph.get_initial_horizontal_orderings(true);

        println!("{graph}");
        graph.fill_vertical_rank_gaps(&order);
        println!("{graph}");

        for (edge_idx, _edge) in graph.edges.iter().enumerate() {
            if let Some(len) = graph.simplex_edge_length(edge_idx) {
                assert!(len.abs() <= MIN_EDGE_LENGTH)
            }
        }
    }
}
