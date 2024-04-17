#![allow(non_snake_case)]

use dioxus::prelude::*;
use log::LevelFilter;

use std::io::Write;
use std::process::{Command, Stdio};
use std::str;

use dot_rs::api::dot_to_svg;
use dot_rs::dot_examples::DOT_EXAMPLES;

fn main() {
    // Init debug
    dioxus_logger::init(LevelFilter::Info).expect("failed to init logger");

    dioxus::launch(App);
}

#[component]
fn App() -> Element {
    // Build cool things ✌️
    let rows = DOT_EXAMPLES
        .iter()
        // .filter(|(title, _dot)| *title == "large_example")
        // .filter(|(title, _dot)| *title != "large_example")
        .map(|(title, dot)| rsx! { DotSet { title: title.to_string(), dot: dot.to_string() } });


    rsx! {
        link { rel: "stylesheet", href: "style.css" }
        // img { src: "header.svg", id: "header" }
        div { class: "dropdown",
            button { class: "dropbtn", "Select Function" }
            div { class: "dropdown-content",
                a { href: "#", "Large Example" }
                a { href: "#", "All Examples" }
                a { href: "#", "Animations" }
            }
        }
        {rows}
    }
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
fn system_call_dot_to_svg(graph: &str, custom_dot: bool) -> String {
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

fn layout_rs_to_svg(dot: &str) -> String {
    use gv::GraphBuilder;
    use layout::backends::svg::SVGWriter;
    use layout::gv;

    let mut parser = gv::DotParser::new(dot);
    let graph = parser.process().unwrap();
    let mut svg = SVGWriter::new();
    let mut gb = GraphBuilder::new();
    gb.visit_graph(&graph);
    let mut vg = gb.get();

    vg.do_it(false, false, false, &mut svg);
    svg.finalize()
}

#[component]
fn DotSet(title: String, dot: String) -> Element {
    let dot_rs_svg = dot_to_svg(&dot);
    let graphviz_dot_svg = system_call_dot_to_svg(&dot, false);
    let layout_svg = layout_rs_to_svg(&dot);

    rsx! {
        div { display: "flex", flex_flow: "column nowrap",
            div { display: "flex", flex_flow: "row nowrap", width: "100%",
                div { display: "flex", flex: 1, justify_content: "left", "{title}" }
            }
            div { display: "flex", flex_flow: "row nowrap", width: "100%",
                div { display: "flex", flex_flow: "row nowrap", width: "100%",
                    div { display: "flex", flex: 1, justify_content: "left", pre { "{dot}" } }
                }
                div { display: "flex", flex_flow: "row nowrap", width: "100%",
                    div {
                        display: "flex",
                        flex: 1,
                        justify_content: "center",
                        dangerous_inner_html: "{dot_rs_svg}"
                    }
                }
                div { display: "flex", flex_flow: "row nowrap", width: "100%",
                    div {
                        display: "flex",
                        flex: 1,
                        justify_content: "center",
                        dangerous_inner_html: "{graphviz_dot_svg}"
                    }
                }
                div { display: "flex", flex_flow: "row nowrap", width: "100%",
                    div {
                        display: "flex",
                        flex: 1,
                        justify_content: "center",
                        dangerous_inner_html: "{layout_svg}"
                    }
                }
            }
        }
    }
}
