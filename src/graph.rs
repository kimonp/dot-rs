use std::{collections::HashSet, fmt::Display};

#[derive(Debug)]
pub struct Graph {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
}

impl Graph {
    pub fn new() -> Self {
        Graph {
            nodes: vec![],
            edges: vec![],
        }
    }

    pub fn get_node(&self, node_idx: usize) -> &Node {
        &self.nodes[node_idx]
    }

    pub fn get_node_mut(&mut self, node_idx: usize) -> &mut Node {
        &mut self.nodes[node_idx]
    }

    pub fn get_edge(&self, edge_idx: usize) -> &Edge {
        &self.edges[edge_idx]
    }

    /// Add a new node identified by name, and return the node's index in the graph.
    pub fn add_node(&mut self, name: &str) -> usize {
        let new_node = Node::new(name);
        let idx = self.nodes.len();
        self.nodes.push(new_node);

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

    #[allow(unused)]
    pub fn draw_graph(&mut self) {
        self.rank();
        self.ordering();
        self.position();
        self.make_splines()
    }

    /// Rank nodes in the graph using the network simplex algorithm described in [TSE93].
    pub fn rank(&mut self) {
        self.feasible_tree();
        // while let Some(e) = self.leave_edge() {
        //     let f = self.enter_edge(e);
        //     self.exchange(e, f);
        // }
        // self.normalize();
        // self.balance();
    }

    fn feasible_tree(&mut self) {
        self.init_rank();
        // while self.tight_tree() < |V| {
        //     let e = self.minimal_non_tree_edge();
        //     let mut delta = slack(e);
        //     if foo = e.head {
        //         delta = -delta;
        //     }
        //     for node in tree {
        //         node.rank r.rank + delta;
        //     }
        // }
        // self.init_cutvalues();
        // todo!()
    }

    /// Documentation from the paper:
    /// * Nodes with no unscanned in-edges are placed in a queue.
    /// * As nodes are taken off the queue, they are assigned the least rank
    ///   that satisfies their in-edges, and their out-edges are marked as scanned.
    /// * In the simplist case, where minLength() == 1 for all edges, this corresponds
    ///   to viewing the graph as a poset (partially ordered set) and assigning the
    ///   the minimal elements to rank 0.  These nodes are removed from the poset and the
    ///   new set of minimal elements are assigned rank 1, etc.
    /// 
    //// TODO: Don't we have to remove redundant edges and ensure the graph is
    ///        not circular to befor we even start this?
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
    fn init_rank(&mut self) {
        let mut nodes_to_rank = Vec::new();
        let mut scanned_edges = HashSet::new();

        // Initialize the queue with all nodes with no incoming edges (since no edges
        // are scanned yet)
        for (index, node) in self.nodes.iter_mut().enumerate() {
            node.set_rank(None);

            if node.no_in_edges() {
                nodes_to_rank.push(index);
            }
        }

        let mut cur_rank = 0;
        while !nodes_to_rank.is_empty() {
            let mut next_nodes_to_rank = Vec::new();
            while let Some(node_idx) = nodes_to_rank.pop() {
                let node = self.get_node_mut(node_idx);

                if node.rank.is_none() {
                    node.set_rank(Some(cur_rank));

                    let out_edges = self.get_node(node_idx).out_edges.clone();
                    for edge_idx in out_edges {
                        scanned_edges.insert(edge_idx);

                        let edge = self.get_edge(edge_idx);
                        let dst_node = self.get_node(edge.dst_node);
                        if dst_node.no_unscanned_in_edges(&scanned_edges) {
                            next_nodes_to_rank.push(edge.dst_node);
                        }
                    }
                }
            }

            cur_rank += 1;
            next_nodes_to_rank.iter().for_each(|idx| nodes_to_rank.push(*idx));
        }
    }

    fn leave_edge(&self) -> Option<usize> {
        todo!()
    }

    fn enter_edge(&self, edge: usize) -> usize {
        todo!()
    }

    fn exchange(&self, e1: usize, e2: usize) -> usize {
        todo!()
    }

    fn normalize(&mut self) {
        todo!();
    }

    fn balance(&mut self) {
        todo!();
    }

    fn ordering(&mut self) {
        todo!();
    }

    fn position(&mut self) {
        todo!();
    }

    fn make_splines(&mut self) {
        todo!();
    }
}

impl Display for Graph {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        for edge in &self.edges {
            let src = &self.nodes[edge.src_node];
            let dst = &self.nodes[edge.dst_node];

            let _ = writeln!(fmt, "{src} -> {dst}");
        }
        Ok(())
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Node {
    name: String,
    rank: Option<u32>,
    in_edges: Vec<usize>,
    out_edges: Vec<usize>,
}

enum EdgeDisposition {
    In,
    Out,
}

impl Node {
    pub fn new(name: &str) -> Self {
        Node {
            name: name.to_string(),
            rank: None,
            in_edges: vec![],
            out_edges: vec![],
        }
    }

    /// Add either an in our out edge to the node.
    fn add_edge(&mut self, edge: usize, disposition: EdgeDisposition) {
        match disposition {
            EdgeDisposition::In => &self.in_edges.push(edge),
            EdgeDisposition::Out => &self.out_edges.push(edge),
        };
    }

    /// Return the list of In our Out edges.
    fn get_edges(&self, disposition: EdgeDisposition) -> &Vec<usize> {
        match disposition {
            EdgeDisposition::In => &self.in_edges,
            EdgeDisposition::Out => &self.out_edges,
        }
    }

    // Return true if none of the incoming edges to node are in the set scanned_edges.
    // * If any incoming edges are not scanned, return false
    // * If there are no incoming edges, return true
    fn no_unscanned_in_edges(&self, scanned_edges: &HashSet<usize>) -> bool {
        for edge_idx in self.get_edges(EdgeDisposition::In) {
           if !scanned_edges.contains(edge_idx) {
                return false;
           }
        }
        true
    }

    fn no_in_edges(&self) -> bool {
        self.get_edges(EdgeDisposition::In).is_empty()
    }

    fn set_rank(&mut self, rank: Option<u32>) {
        self.rank = rank;
    }
}

impl Display for Node {
    fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> std::result::Result<(), std::fmt::Error> {
        write!(fmt, "{:?}: {}", self.rank, &self.name)
    }
}

#[derive(Debug, Eq, PartialEq)]
pub struct Edge {
    src_node: usize,
    dst_node: usize,
    cut_value: Option<u32>,
}

impl Edge {
    pub fn new(src_node: usize, dst_node: usize) -> Self {
        Edge {
            src_node,
            dst_node,
            cut_value: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_rank() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");
        let d_idx = graph.add_node("D");

        graph.add_edge(a_idx, b_idx);
        graph.add_edge(a_idx, c_idx);
        graph.add_edge(b_idx, d_idx);
        graph.add_edge(c_idx, d_idx);

        graph.rank();

        println!("{graph}");

        assert_eq!(graph.nodes[a_idx].rank, Some(0));
        assert_eq!(graph.nodes[b_idx].rank, Some(1));
        assert_eq!(graph.nodes[c_idx].rank, Some(1));
        assert_eq!(graph.nodes[d_idx].rank, Some(2));
    }

    #[test]
    fn test_rank_scaning() {
        let mut graph = Graph::new();
        let a_idx = graph.add_node("A");
        let b_idx = graph.add_node("B");
        let c_idx = graph.add_node("C");

        graph.add_edge(a_idx, b_idx);
        graph.add_edge(a_idx, c_idx);
        graph.add_edge(b_idx, c_idx);

        graph.rank();
        println!("{graph}");

        assert_eq!(graph.nodes[a_idx].rank, Some(0));
        assert_eq!(graph.nodes[b_idx].rank, Some(1));
        assert_eq!(graph.nodes[c_idx].rank, Some(2));
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

        graph.rank();
        println!("{graph}");
    }
}
