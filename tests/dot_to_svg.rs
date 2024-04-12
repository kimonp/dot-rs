//! Test all the examples in DOT_EXAMPLES.

use std::fs::File;
use std::io::Write;

use dot_rs::api::dot_to_svg;
use dot_rs::dot_examples::dot_example_str;
use rstest::rstest;

pub fn write_svg_to_file(name: &str, svg: &str) {
    let mut file = File::create(format!("{name}.svg")).unwrap();

    file.write_all(svg.as_bytes()).unwrap();
}

#[rstest(
    test_title,
    case::a_to_b_and_c("a_to_b_and_c"),
    case::b_and_c_to_a("b_and_c_to_a"),
    case::a_to_c_and_b_to_a("a_to_c_and_b_to_a"),
    case::two_cyclic("two_cyclic"),
    case::three_cyclic("three_cyclic"),
    case::complex_cyclic("complex_cyclic"),
    case::crossing_issues("crossing_issues"),
    case::complex_crossing("complex_crossing"),
    case::flux_capacitor("flux_capacitor"),
    case::t1_2_1("t1_2_1"),
    case::t2_1_2("t2_1_2"),
    case::a_to_4_nodes("a_to_4_nodes"),
    case::simple_scramble("simple_scramble"),
    case::reverse_scramble("reverse_scramble"),
    case::tse_paper_example_2_3("tse_paper_example_2_3"),
    case::tse_paper_example_2_3_scrambled("tse_paper_example_2_3_scrambled"),
    case::tse_paper_example_2_3_extended("tse_paper_example_2_3_extended"),
    case::tse_paper_example_2_3_simplified("tse_paper_example_2_3_simplified"),
    case::in_spread("in_spread"),
    case::large_example("large_example")
)]
fn test_dot_to_svg(test_title: &str) {
    let dot = dot_example_str(test_title);
    let svg = dot_to_svg(dot);

    write_svg_to_file("test_dot_to_svg", &svg);
}
