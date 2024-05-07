dot-rs is a Rust implementation of GraphViz's dot derived from the paper: A Technique for Drawing Directed Graphs,
and the GraphViz source itself.

Closely follows the GraphViz algorithm, and produces very similar output in most cases.  The GraphViz code is
rather difficult to follow, so this code can perhaps be a more gentle introduction to the subject.

This was a learning project for me, not an intent to make a rust library for GraphViz.

This is not production quality because:
* It does not produce error types: will panic on any error.
* Does not implement subgraphs, nor most other exotic options
  in dot files.
* Draws straight lines for edges, not splines.
* Runs quite a bit slower, but this can likely be easily fixed
  with some investigation.
  
Some side by size comparison examples:

dot-rs | GraphViz
--- | --- |
<img src="dot_examples/basic/generated/dot-rs/a_to_b_and_c.svg"> | <img src="dot_examples/basic/generated/GraphViz/a_to_b_and_c.svg">
<img src="dot_examples/basic/generated/dot-rs/b_and_c_to_a.svg"> | <img src="dot_examples/basic/generated/GraphViz/b_and_c_to_a.svg">
<img src="dot_examples/tse_paper/generated/dot-rs/example_2_3.svg"> | <img src="dot_examples/tse_paper/generated/GraphViz/example_2_3.svg">
<img src="dot_examples/tse_paper/generated/dot-rs/example_2_3_extended.svg"> | <img src="dot_examples/tse_paper/generated/GraphViz/example_2_3_extended.svg">
<img src="dot_examples/layout/generated/dot-rs/large_example.svg"> | <img src="dot_examples/layout/generated/GraphViz/large_example.svg">