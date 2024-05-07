//! Test all the examples in DOT_EXAMPLES.

use std::fs::File;
use std::io::Write;
use std::process::{Command, Stdio};

use dot_rs::api::dot_to_svg;
use dot_rs::dot_examples::{get_dot_example, get_svg_example_path, SvgGenerator};
use rstest::rstest;

/// Writes both the dot-rs and GraphViz version of the SVG to the examples directory.
fn write_svg_to_example_file(test_title: &str, svg: &str) {
    let dot_rs_svg_file_path = get_svg_example_path(test_title, SvgGenerator::DotRs);
    let mut file = File::create(dot_rs_svg_file_path).unwrap();
    file.write_all(svg.as_bytes()).unwrap();
}

fn write_graphviz_svg_example(test_title: &str, dot_str: &str) {
    let graphviz_svg_file_path = get_svg_example_path(test_title, SvgGenerator::GraphViz);

    system_call_dot_to_svg_file(dot_str, &graphviz_svg_file_path);
}

/// Calls dot via a system command and returns the svg as a string.
///
/// Note: graphviz (and dot) needs to be installed for this to work: https://graphviz.org/download/
///
/// Example command: echo 'digraph { a -> b; a -> c; b -> d; c -> d;}' | dot -Tsvg example.svg
fn system_call_dot_to_svg_file(dot_str: &str, svg_path: &str) {
    let graph = dot_str.to_string();
    let dot_bin = "dot";
    let result = Command::new(dot_bin)
        .arg("-Tsvg")
        .arg(&format!("-o{svg_path}"))
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn();
    
    if let Ok(mut dot_child) = result {
        let mut stdin = dot_child.stdin.take().expect("Failed to open stdin");
        std::thread::spawn(move || {
            stdin
                .write_all(graph.as_bytes())
                .expect("Failed to write to stdin");
        });
        let _status = dot_child.wait_with_output().expect("Failed to wait on dot");
    } else {
        // Don't expect this to work in CI since no dot is installed.
    }
}

#[rstest(
    test_title,
    case::a_to_b_and_c("basic/a_to_b_and_c"),
    case::b_and_c_to_a("basic/b_and_c_to_a"),
    case::a_to_c_and_b_to_a("basic/a_to_c_and_b_to_a"),
    case::two_cyclic("cyclic/two_cyclic"),
    case::three_cyclic("cyclic/three_cyclic"),
    case::complex_cyclic("cyclic/complex_cyclic"),
    case::crossing_issues("crossings/must_cross"),
    case::complex_crossing("crossings/complex_crossing"),
    case::flux_capacitor("basic/flux_capacitor"),
    case::t1_2_1("basic/t1_2_1"),
    case::t2_1_2("basic/t2_1_2"),
    case::a_to_4_nodes("basic/a_to_4_nodes"),
    case::simple_scramble("basic/simple_scramble"),
    case::reverse_scramble("basic/reverse_scramble"),
    case::tse_paper_example_2_3("tse_paper/example_2_3"),
    case::tse_paper_example_2_3_scrambled("tse_paper/example_2_3_scrambled"),
    case::tse_paper_example_2_3_extended("tse_paper/example_2_3_extended"),
    case::tse_paper_example_2_3_simplified("tse_paper/example_2_3_simplified"),
    case::in_spread("basic/in_spread"),
    case::large_example("layout/large_example"),
    case::go_imports("graphviz.org/go_package_imports"),
    // case::world("graphviz.org/world"),
    case::unix("graphviz.org/unix"),
    case::profile("graphviz.org/profile"),
)]
fn test_dot_to_svg(test_title: &str) {
    let dot = get_dot_example(test_title);
    let svg = dot_to_svg(&dot);

    write_svg_to_example_file(test_title, &svg);
    write_graphviz_svg_example(test_title, &dot)
}
