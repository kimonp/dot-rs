//! Test all the examples in DOT_EXAMPLES.

use std::fs::File;
use std::io::Write;

use dot_rs::api::dot_to_svg;
use dot_rs::dot_examples::get_dot_example;
use rstest::rstest;

pub fn write_svg_to_file(name: &str, svg: &str) {
    let mut file = File::create(format!("{name}.svg")).unwrap();

    file.write_all(svg.as_bytes()).unwrap();
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
    // case::profile("graphviz.org/profile"),
)]
fn test_dot_to_svg(test_title: &str) {
    let dot = get_dot_example(test_title);
    let svg = dot_to_svg(&dot);

    write_svg_to_file("test_dot_to_svg", &svg);
}