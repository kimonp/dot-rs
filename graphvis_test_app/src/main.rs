// import the prelude to get access to the `rsx!` macro and the `Scope` and `Element` types
use dioxus::prelude::*;
use dioxus_desktop::tao::dpi::LogicalPosition;
use dioxus_desktop::{Config, PhysicalSize, WindowBuilder};
use dot_rs::graph::Graph;
use dot_rs::svg::SVG;
use std::io::Write;
use std::process::{Command, Stdio};
use std::str;

use dot_rs::dot_examples::DOT_EXAMPLES;

fn main() {
    // launch the dioxus app in a webview
    // dioxus_desktop::launch(App);

    let size = PhysicalSize::new(2000, 4000);
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
fn dot_to_svg(graph: &str, custom_dot: bool) -> String {
    let graph = graph.to_string();
    let dot_path = if custom_dot {
        "/Users/kimonp/code/graphviz-9.0.0/cmd/dot/"
    } else {
        ""
    };
    let dot_cmd = "dot";
    let dot_bin = format!("{dot_path}{dot_cmd}");
    let mut dot_child = Command::new(dot_bin)
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
    let rows = DOT_EXAMPLES
        .iter()
        .map(|(title, dot)| rsx! { DotSet { title: title.to_string(), dot: dot.to_string() } });

    cx.render(rsx! {rows})
}

#[component]
fn DotSet(cx: Scope, title: String, dot: String) -> Element {
    let mut graph = Graph::from(dot);

    graph.layout_nodes();

    let svg = SVG::new(graph.clone(), false);
    // let svg_debug = SVG::new(graph, false);
    let dot_rs = svg.to_string();
    //let dot_rs_debug = svg_debug.to_string();

    // let custom_dot_svg = dot_to_svg(dot, true);
    let std_dot_svg = dot_to_svg(dot, false);

    cx.render(rsx! {
        div { display: "flex", flex_flow: "column nowrap",
            div { display: "flex", flex_flow: "row nowrap", width: "100%", div { display: "flex", flex: 1, justify_content: "left", "{title}" } }
            div { display: "flex", flex_flow: "row nowrap", width: "100%",
                div { display: "flex", flex_flow: "row nowrap", width: "100%", div { display: "flex", flex: 1, justify_content: "left", pre { "{dot}" } } }
                div { display: "flex", flex_flow: "row nowrap", width: "100%", div { display: "flex", flex: 1, justify_content: "center", dangerous_inner_html: "{dot_rs}" } }
                div { display: "flex", flex_flow: "row nowrap", width: "100%", div { display: "flex", flex: 1, justify_content: "center", dangerous_inner_html: "{std_dot_svg}" } }
            }
        }
    })
}
