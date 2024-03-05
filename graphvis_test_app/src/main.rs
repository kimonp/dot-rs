// import the prelude to get access to the `rsx!` macro and the `Scope` and `Element` types
use dioxus::prelude::*;
use dioxus_desktop::tao::dpi::LogicalPosition;
use dioxus_desktop::{Config, PhysicalSize, WindowBuilder};
use dot_rs::graph::Graph;
use std::io::Write;
use std::process::{Command, Stdio};
use std::str;

fn main() {
    // launch the dioxus app in a webview
    // dioxus_desktop::launch(App);

    let size = PhysicalSize::new(2000, 1000);
    let position = LogicalPosition::new(10, 10);
    let window = WindowBuilder::new()
        .with_title("GraphViz Comparison")
        .with_inner_size(size)
        .with_position(position);

    dioxus_desktop::launch_with_props(App, (), Config::new().with_window(window))
}

// fn petgraph_test() {
//     use petgraph_evcxr::draw_graph;
//     let mut tree: petgraph::Graph<&str, &str, petgraph::Directed> = petgraph::Graph::new();
//     let tree_item1 = tree.add_node("a");
//     let tree_item2 = tree.add_node("b");
//     let tree_item3 = tree.add_node("c");
//     let tree_item4 = tree.add_node("d");
//     let tree_item5 = tree.add_node("e");
//     tree.add_edge(tree_item1, tree_item2, "");
//     tree.add_edge(tree_item1, tree_item3, "");
//     tree.add_edge(tree_item2, tree_item4, "");
//     tree.add_edge(tree_item2, tree_item5, "");
//     let foo = draw_graph(&tree);
// }

/// Calls dot via a system command and returns the svg as a string.
///
/// Note: graphviz (and dot) needs to be installed for this to work: https://graphviz.org/download/
///
/// Example command: echo 'digraph { a -> b; a -> c; b -> d; c -> d;}' | dot -Tsvg
fn dot_to_svg(graph: &str) -> String {
    let graph = graph.to_string();
    let mut dot_child = Command::new("dot")
        .arg("-Tsvg")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("Failed to start dot process");

    let mut stdin = dot_child.stdin.take().expect("Failed to open stdin");
    std::thread::spawn(move || {
        stdin
            .write_all(graph.as_bytes())
            .expect("Failed to write to stdin");
    });
    let output = dot_child.wait_with_output().expect("Failed to wait on dot");

    String::from_utf8(output.stdout).expect("Output of dot not UTF-8")
}

// define a component that renders a div with the text "Hello, world!"
#[component]
fn App(cx: Scope) -> Element {
    // let dot_str = "digraph {
    //     a -> e; a -> f; a -> b;
    //     e -> g; f -> g; b -> c;
    //     c -> d; d -> h;
    //     g -> h;
    // }";
    // let dot_str = "digraph {
    //     a -> c; b -> c
    // }";
    let dot_str = "digraph {
        a -> b; a -> e; a -> f;
        e -> g; f -> g; b -> c;
        c -> d; d -> h;
        g -> h;
        a -> i; a -> j; a -> k;
        i -> l; j -> l; k -> l;
        l -> h;
    }";

    let mut graph = Graph::from(dot_str);

    graph.layout_nodes();

    let dot_rs = graph.get_svg(false);
    let dot_rs_debug = graph.get_svg(true);

    let dot_svg = dot_to_svg(dot_str);

    cx.render(rsx! {
        div { display: "flex", flex_flow: "column nowrap",
            div { display: "flex", flex_flow: "row nowrap", width: "100%",
                div { display: "flex", flex: 1, justify_content: "center", "dot-rs" }
                div { display: "flex", flex: 1, justify_content: "center", "dot-rs debug" }
                div { display: "flex", flex: 1, justify_content: "center", "graphviz & dot" }
            }
            div { display: "flex", flex_flow: "row nowrap", width: "100%",
                div { display: "flex", flex: 1, justify_content: "center", dangerous_inner_html: "{dot_rs}" }
                div { display: "flex", flex: 1, justify_content: "center", dangerous_inner_html: "{dot_rs_debug}" }
                div { display: "flex", flex: 1, justify_content: "center", dangerous_inner_html: "{dot_svg}" }
            }
        }
    })
}
