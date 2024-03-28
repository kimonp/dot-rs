//! Test all the examples in DOT_EXAMPLES.

use dot_rs::dot_examples::dot_example_graph;
use dot_rs::graph::Graph;
use dot_rs::svg::SVG;
use rstest::rstest;

#[rstest(
    graph,
    case::a_to_b_and_c(dot_example_graph("a_to_b_and_c")),
    case::b_and_c_to_a(dot_example_graph("b_and_c_to_a")),
    case::a_to_c_and_b_to_a(dot_example_graph("a_to_c_and_b_to_a")),
    case::two_cyclic(dot_example_graph("two_acyclic")),
    case::three_cyclic(dot_example_graph("three_acyclic")),
    case::complex_cyclic(dot_example_graph("complex_acyclic")),
    case::flux_capacitor(dot_example_graph("flux_capacitor")),
    case::t1_2_1(dot_example_graph("t1_2_1")),
    case::a_to_4_nodes(dot_example_graph("a_to_4_nodes")),
    case::simple_scramble(dot_example_graph("simple_scramble")),
    case::reverse_scramble(dot_example_graph("reverse_scramble")),
    case::tse_paper_example_2_3(dot_example_graph("tse_paper_example_2_3")),
    case::tse_paper_example_2_3_scrambled(dot_example_graph(
        "tse_paper_example_2_3_scrambled"
    )),
    case::tse_paper_example_2_3_extended(dot_example_graph("tse_paper_example_2_3_extended")),
    case::tse_paper_example_2_3_simplified(dot_example_graph(
        "tse_paper_example_2_3_simplified"
    )),
    case::in_spread(dot_example_graph("in_spread"))
)]
fn dot_to_svg(mut graph: Graph) {
    graph.layout_nodes();

    let svg = SVG::new(graph, false);
    svg.write_to_file("test_dot_to_svg");
}
