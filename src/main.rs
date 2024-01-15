mod graph;

use graph::Graph;

fn main() {
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
}
