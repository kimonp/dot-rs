//! Examples of graphs to be used for testing and display.

use crate::graph::Graph;

pub const DOT_EXAMPLES: [(&str, &str); 16] = [
    ("a_to_b_and_c",        "digraph {a -> b; a -> c;}"),
    ("b_and_c_to_a",        "digraph {b -> a; c -> a;}"),
    ("a_to_c_and_b_to_a",   "digraph {a -> c; b -> a;}"),
    ("two_acyclic",         "digraph {a -> b; b -> a;}"),
    ("three_acyclic",       "digraph {a -> b; b -> c; c -> a; }"),
    ("complex_acyclic",     "digraph { a -> c; b -> d; c -> e; e -> d; d -> c; }"),
    ("flux_capacitor",      "digraph {a -> c; b -> c; c -> d}"),
    ("t1_2_1",              "digraph { a -> b; a -> c; b -> d; c -> d; }"),
    ("a_to_4_nodes",        "digraph {a -> b; a -> c; a -> d; a -> e;}"),
    ( "simple_scramble",    "digraph { a -> b; a -> c; c -> d; b -> e; }"),
    ( "reverse_scramble",   "digraph { a -> b; a -> c; b -> e; c -> d; }"),
    (
        "tse_paper_example_2_3",
        "digraph {
            a -> b; a -> e; a -> f;
            e -> g; f -> g; b -> c;
            c -> d; d -> h;
            g -> h;
        }",
    ),
    (
        "tse_paper_example_2_3_scrambled",
        "digraph {
            a -> e; a -> f; a -> b;
            e -> g; f -> g; b -> c;
            c -> d; d -> h;
            g -> h;
        }",
    ),
    (
        "tse_paper_example_2_3_extended",
        "digraph {
            a -> b; a -> e; a -> f;
            e -> g; f -> g; b -> c;
            c -> d; d -> h;
            g -> h;
            a -> i; a -> j; a -> k;
            i -> l; j -> l; k -> l;
            l -> h;
        }",
    ),
    (
        "tse_paper_example_2_3_simplified",
        "digraph {
            a -> b; a -> e; a -> f;
            e -> g; f -> g; b -> c;
            c -> d; d -> h;
        }",
    ),
    (
        "in_spread",
        "digraph {
            a -> b; a -> c; a -> d;
            a -> i; a -> j; a -> k;
            i -> l; j -> l; k -> l;
            l -> h
        }",
    ),
];

pub fn dot_example_graph(title: &str) -> Graph {
    for (dot_title, dot) in DOT_EXAMPLES {
        if title == dot_title {
            return Graph::from(dot)
        }
    }
    panic!("Could not find requested example: {title}")
}