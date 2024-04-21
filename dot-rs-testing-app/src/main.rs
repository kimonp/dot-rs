#![allow(non_snake_case)]

use dioxus::prelude::*;
use log::LevelFilter;

use std::borrow::Borrow;
use std::io::Write;
use std::process::{Command, Stdio};
use std::str;

use dot_rs::api::{dot_to_svg, dot_to_svg_debug_snapshots};
use dot_rs::dot_examples::{dot_example_str, DOT_EXAMPLES};

fn main() {
    // Init debug
    dioxus_logger::init(LevelFilter::Info).expect("failed to init logger");

    dioxus::launch(App);
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Selection {
    LargeExample,
    AllExamples,
    Animations,
}

use Selection::*;

#[component]
fn App() -> Element {
    let mut selection = use_signal(|| Animations);

    rsx! {
        link { rel: "stylesheet", href: "style.css" }
        div { class: "dropdown",
            button { class: "dropbtn", "Select Function" }
            div { class: "dropdown-content",
                div { onmouseup: move |_| selection.set(LargeExample), "Large Example" }
                div { onmouseup: move |_| selection.set(AllExamples), "All Examples" }
                div { onmouseup: move |_| selection.set(Animations), "Animations" }
            }
        }
        DotExamples { selection: *selection.read() }
    }
}

// fn set_selection(event: Event<MouseData>, mut signal: Signal<Selection>, selection: Selection) {
//     if event.
//     signal.set(selection);
// }

// fn click_select_menu(item: Selections) {
//     log::info!("Clicked!");
// }

#[component]
fn DotExamples(selection: Selection) -> Element {
    match selection {
        Animations => rsx! {
            AllAnimations {}
        },
        LargeExample => rsx! {
            LargeDotExample {}
        },
        AllExamples => rsx! {
            AllDotExamples {}
        },
    }
}

#[component]
fn AllAnimations() -> Element {
    let frame = use_signal(|| 0);
    let snapshots = use_signal(|| {
        let dot = dot_example_str("large_example");
        dot_to_svg_debug_snapshots(dot)
    });
    let max_frame = snapshots.read().total_count() - 1;
    let (step_back, step_forward) = snapshots.read().steps(*frame.read());
    let (group_title, title, svg) = snapshots.read().get(*frame.read() as usize).expect("invalid frame");

    rsx! {
        link { rel: "stylesheet", href: "style.css" }
        h1 { "Crossing Animations" }
        GraphSnapshots { svg }
        h4 { { format!("{group_title}: {title}")} }
        div { class: "slider-container",
            input {
                r#type: "range",
                min: "0",
                max: "{max_frame}",
                value: "{frame}",
                class: "slider",
                id: "animationFrame",
                onchange: move |event| set_frame_from_event(event, frame, max_frame)
            }
        }
        div {
            div { class: "button-tray",
                button {
                    class: "animation-btn",
                    onmouseup: move |_| set_frame(frame, step_back, max_frame),
                    "⏮"
                }
                button {
                    class: "animation-btn",
                    onmouseup: move |_| dec_frame(frame, 1),
                    "⏴"
                }
                button {
                    class: "animation-btn",
                    onmouseup: move |_| inc_frame(frame, 1, max_frame),
                    "⏵"
                }
                button {
                    class: "animation-btn",
                    onmouseup: move |_| set_frame(frame, step_forward, max_frame),
                    "⏭"
                }
            }
            h3 { "Frame {frame} of {max_frame}" }
        }
    }
}

fn set_frame_from_event(event: Event<FormData>, mut frame: Signal<usize>, max_frame: usize) {
    let value = event.value();
    let new_val = 0.max(max_frame.min(value.parse::<usize>().unwrap_or(0)));

    frame.set(new_val);
}

fn set_frame(mut frame: Signal<usize>, value: usize, max_frame: usize) {
    let new_val = 0.max(max_frame.min(value));

    frame.set(new_val);
}

fn inc_frame(mut frame: Signal<usize>, amount: usize, max_frame: usize) {
    let cur_val = *frame.read();
    let new_val = max_frame.min(cur_val + amount);

    frame.set(new_val);
}

fn dec_frame(mut frame: Signal<usize>, amount: usize) {
    let cur_val = *frame.read();
    let new_val = 0_isize.max(cur_val as isize - amount as isize) as usize;

    frame.set(new_val)
}

#[component]
fn GraphSnapshots(svg: String) -> Element {
    rsx! {
        div { display: "flex", flex_flow: "column nowrap",
            div {
                display: "flex",
                flex: 1,
                justify_content: "center",
                dangerous_inner_html: "{svg}"
            }
        }
    }
}

#[component]
fn AllDotExamples() -> Element {
    let rows = DOT_EXAMPLES.iter().map(|(title, dot)| {
        rsx! {
            DotSet { title: title.to_string(), dot: dot.to_string() }
        }
    });

    rsx! {
        div { "All examples" }
        {rows}
    }
}

#[component]
fn LargeDotExample() -> Element {
    let rows = DOT_EXAMPLES
        .iter()
        .filter(|(title, _dot)| *title == "large_example")
        .map(|(title, dot)| {
            rsx! {
                DotSet { title: title.to_string(), dot: dot.to_string() }
            }
        });

    rsx! {
        div { "Large example" }
        {rows}
    }
}

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
    // let dot_rs_svg = "DOT-RS HERE";
    // let graphviz_dot_svg = "GRAPHVIZ HERE";
    // let layout_svg = "LAYOUT HERE";
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
                    div { display: "flex", flex: 1, justify_content: "left",
                        pre { "{dot}" }
                    }
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
