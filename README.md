Rust implementation of graph-vis dot from the paper: A Technique for Drawing Directed Graphs

Closely follows the GraphViz algorithm, and produces very similar output in most simple cases.

This was a learning project for me, not an intent to make a rust library for GraphViz.

This is not production quality because:
* It does not produce error types: will panic on any error.
* Does not implement subgraphs, nor most other exotic options
  in dot files.
* Draws straight lines for edges, not splines.
* Runs quite a bit slow, but this could probably be easily fixed
  with some investigation.
  
Some side by size comparison examples:

dot-rs | GraphViz
--- | --- |
<img src="dot_examples/basic/generated/dot-rs/a_to_b_and_c.svg"> | <img src="dot_examples/basic/generated/GraphViz/a_to_b_and_c.svg">
<img src="dot_examples/basic/generated/dot-rs/b_and_c_to_a.svg"> | <img src="dot_examples/basic/generated/GraphViz/b_and_c_to_a.svg">
<img src="dot_examples/tse_paper/generated/dot-rs/example_2_3.svg"> | <img src="dot_examples/tse_paper/generated/GraphViz/example_2_3.svg">
<img src="dot_examples/tse_paper/generated/dot-rs/example_2_3_extended.svg"> | <img src="dot_examples/tse_paper/generated/GraphViz/example_2_3_extended.svg">
<img src="dot_examples/layout/generated/dot-rs/large_example.svg"> | <img src="dot_examples/layout/generated/GraphViz/large_example.svg">