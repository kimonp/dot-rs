//! Examples of graphs to be used for testing and display.

use crate::graph::Graph;

pub const DOT_EXAMPLES: [(&str, &str); 20] = [
    ("a_to_b_and_c", "digraph {a -> b; a -> c;}"),
    ("b_and_c_to_a", "digraph {b -> a; c -> a;}"),
    ("a_to_c_and_b_to_a", "digraph {a -> c; b -> a;}"),
    ("two_cyclic", "digraph {a -> b; b -> a;}"),
    ("three_cyclic", "digraph {a -> b; b -> c; c -> a; }"),
    (
        "complex_cyclic",
        "digraph { a -> c; b -> d; c -> e; e -> d; d -> c; }",
    ),
    ("crossing_issues", "digraph {0 -> a1; 0 -> b1; 0 -> c1; a1 -> b2; b1 -> a2; c1 -> a2; c1 -> c2; a1 -> c2; b1 -> b2;}"), 
    (
        "complex_crossing",
        "digraph {
            1->8;
            2->8; 2->20;
            8->16;
            16->19;
            8->11; 8->10;
            11->16; 11->15;
            10->15; 10->14; 10->13;
            16->20; 16->18;
            14->18;
            1->13;
            13->17;
        }",
    ),
    ("flux_capacitor", "digraph {a -> c; b -> c; c -> d}"),
    ("t1_2_1", "digraph { a -> b; a -> c; b -> d; c -> d; }"),
    ("t2_1_2", "digraph { a -> b; c -> b; b -> d; b -> e; }"),
    ("a_to_4_nodes", "digraph {a -> b; a -> c; a -> d; a -> e;}"),
    (
        "simple_scramble",
        "digraph { a -> b; a -> c; c -> d; b -> e; }",
    ),
    (
        "reverse_scramble",
        "digraph { a -> b; a -> c; b -> e; c -> d; }",
    ),
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
    // Testcase from:
    //  A Fast Layout Algorithm for k-Level Graphs by
    //  Christoph Buchheim, Michael Junger and Sebastian Leipert
    (
        "large_example",
        "digraph {
            1->3;
            2->3; 2->20;
            3->5;
            5->7;
            7->9;
            9->12;
            12->20;
            3->4;
            4->6;
            6->16;
            16->19;
            19->22;
            22->23;
            3->23;
            6->23; 6->8;
            8->11; 8->10;
            11->16; 11->15;
            10->15; 10->14; 10->13;
            16->20; 16->18;
            14->18;
            1->13;
            13->17;
            18->21;
            1->21;
            21->23;
          }",
    ),
    // (
    //     // https://graphviz.org/Gallery/directed/go-package.html
    //     "go_package_imports",
    //     "digraph regexp { 
    //         n0 -> n1;
    //         n0 -> n2;
    //         n0 -> n3;
    //         n0 -> n4;
    //         n0 -> n5;
    //         n0 -> n6;
    //         n0 -> n7;
    //         n0 -> n8;
    //         n0 -> n9;
    //         n1 -> n10;
    //         n1 -> n2;
    //         n1 -> n8;
    //         n1 -> n9;
    //         n1 -> n11;
    //         n2 -> n11;
    //         n2 -> n7;
    //         n3 -> n4;
    //         n3 -> n5;
    //         n3 -> n6;
    //         n3 -> n8;
    //         n3 -> n9;
    //         n4 -> n12;
    //         n5 -> n10;
    //         n5 -> n13;
    //         n5 -> n9;
    //         n5 -> n11;
    //         n5 -> n14;
    //         n6 -> n2;
    //         n6 -> n7;
    //         n6 -> n15;
    //         n6 -> n11;
    //         n6 -> n10;
    //         n6 -> n8;
    //         n6 -> n9;
    //         n7 -> n16;
    //         n7 -> n17;
    //         n7 -> n18;
    //         n7 -> n15;
    //         n10 -> n19;
    //         n10 -> n15;
    //         n11 -> n12;
    //         n12 -> n17;
    //         n12 -> n15;
    //         n13 -> n15;
    //         n13 -> n19;
    //         n13 -> n14;
    //         n14 -> n15;
    //         n16 -> n15;
    //         n18 -> n15;
    //        }"
    // )
];

// Program Profile: (https://graphviz.org/Gallery/directed/profile.html)
// digraph prof {
// 	fontname="Helvetica,Arial,sans-serif"
// 	node [fontname="Helvetica,Arial,sans-serif"]
// 	edge [fontname="Helvetica,Arial,sans-serif"]
// 	node [style=filled];
// 	start -> main [color="0.002 0.999 0.999"];
// 	start -> on_exit [color="0.649 0.701 0.701"];
// 	main -> sort [color="0.348 0.839 0.839"];
// 	main -> merge [color="0.515 0.762 0.762"];
// 	main -> term [color="0.647 0.702 0.702"];
// 	main -> signal [color="0.650 0.700 0.700"];
// 	main -> sbrk [color="0.650 0.700 0.700"];
// 	main -> unlink [color="0.650 0.700 0.700"];
// 	main -> newfile [color="0.650 0.700 0.700"];
// 	main -> fclose [color="0.650 0.700 0.700"];
// 	main -> close [color="0.650 0.700 0.700"];
// 	main -> brk [color="0.650 0.700 0.700"];
// 	main -> setbuf [color="0.650 0.700 0.700"];
// 	main -> copyproto [color="0.650 0.700 0.700"];
// 	main -> initree [color="0.650 0.700 0.700"];
// 	main -> safeoutfil [color="0.650 0.700 0.700"];
// 	main -> getpid [color="0.650 0.700 0.700"];
// 	main -> sprintf [color="0.650 0.700 0.700"];
// 	main -> creat [color="0.650 0.700 0.700"];
// 	main -> rem [color="0.650 0.700 0.700"];
// 	main -> oldfile [color="0.650 0.700 0.700"];
// 	sort -> msort [color="0.619 0.714 0.714"];
// 	sort -> filbuf [color="0.650 0.700 0.700"];
// 	sort -> newfile [color="0.650 0.700 0.700"];
// 	sort -> fclose [color="0.650 0.700 0.700"];
// 	sort -> setbuf [color="0.650 0.700 0.700"];
// 	sort -> setfil [color="0.650 0.700 0.700"];
// 	msort -> qsort [color="0.650 0.700 0.700"];
// 	msort -> insert [color="0.650 0.700 0.700"];
// 	msort -> wline [color="0.650 0.700 0.700"];
// 	msort -> div [color="0.650 0.700 0.700"];
// 	msort -> cmpsave [color="0.650 0.700 0.700"];
// 	merge -> insert [color="0.650 0.700 0.700"];
// 	merge -> rline [color="0.650 0.700 0.700"];
// 	merge -> wline [color="0.650 0.700 0.700"];
// 	merge -> unlink [color="0.650 0.700 0.700"];
// 	merge -> fopen [color="0.650 0.700 0.700"];
// 	merge -> fclose [color="0.650 0.700 0.700"];
// 	merge -> setfil [color="0.650 0.700 0.700"];
// 	merge -> mul [color="0.650 0.700 0.700"];
// 	merge -> setbuf [color="0.650 0.700 0.700"];
// 	merge -> cmpsave [color="0.650 0.700 0.700"];
// 	insert -> cmpa [color="0.650 0.700 0.700"];
// 	wline -> flsbuf [color="0.649 0.700 0.700"];
// 	qsort -> cmpa [color="0.650 0.700 0.700"];
// 	rline -> filbuf [color="0.649 0.700 0.700"];
// 	xflsbuf -> write [color="0.650 0.700 0.700"];
// 	flsbuf -> xflsbuf [color="0.649 0.700 0.700"];
// 	filbuf -> read [color="0.650 0.700 0.700"];
// 	term -> unlink [color="0.650 0.700 0.700"];
// 	term -> signal [color="0.650 0.700 0.700"];
// 	term -> setfil [color="0.650 0.700 0.700"];
// 	term -> exit [color="0.650 0.700 0.700"];
// 	endopen -> open [color="0.650 0.700 0.700"];
// 	fopen -> endopen [color="0.639 0.705 0.705"];
// 	fopen -> findiop [color="0.650 0.700 0.700"];
// 	newfile -> fopen [color="0.634 0.707 0.707"];
// 	newfile -> setfil [color="0.650 0.700 0.700"];
// 	fclose -> fflush [color="0.642 0.704 0.704"];
// 	fclose -> close [color="0.650 0.700 0.700"];
// 	fflush -> xflsbuf [color="0.635 0.707 0.707"];
// 	malloc -> morecore [color="0.325 0.850 0.850"];
// 	malloc -> demote [color="0.650 0.700 0.700"];
// 	morecore -> sbrk [color="0.650 0.700 0.700"];
// 	morecore -> getfreehdr [color="0.650 0.700 0.700"];
// 	morecore -> free [color="0.650 0.700 0.700"];
// 	morecore -> getpagesize [color="0.650 0.700 0.700"];
// 	morecore -> putfreehdr [color="0.650 0.700 0.700"];
// 	morecore -> udiv [color="0.650 0.700 0.700"];
// 	morecore -> umul [color="0.650 0.700 0.700"];
// 	on_exit -> malloc [color="0.325 0.850 0.850"];
// 	signal -> sigvec [color="0.650 0.700 0.700"];
// 	moncontrol -> profil [color="0.650 0.700 0.700"];
// 	getfreehdr -> sbrk [color="0.650 0.700 0.700"];
// 	free -> insert [color="0.650 0.700 0.700"];
// 	insert -> getfreehdr [color="0.650 0.700 0.700"];
// 	setfil -> div [color="0.650 0.700 0.700"];
// 	setfil -> rem [color="0.650 0.700 0.700"];
// 	sigvec -> sigblock [color="0.650 0.700 0.700"];
// 	sigvec -> sigsetmask [color="0.650 0.700 0.700"];
// 	doprnt -> urem [color="0.650 0.700 0.700"];
// 	doprnt -> udiv [color="0.650 0.700 0.700"];
// 	doprnt -> strlen [color="0.650 0.700 0.700"];
// 	doprnt -> localeconv [color="0.650 0.700 0.700"];
// 	sprintf -> doprnt [color="0.650 0.700 0.700"];
// cmpa [color="0.000 1.000 1.000"];
// wline [color="0.201 0.753 1.000"];
// insert [color="0.305 0.625 1.000"];
// rline [color="0.355 0.563 1.000"];
// sort [color="0.408 0.498 1.000"];
// qsort [color="0.449 0.447 1.000"];
// write [color="0.499 0.386 1.000"];
// read [color="0.578 0.289 1.000"];
// msort [color="0.590 0.273 1.000"];
// merge [color="0.603 0.258 1.000"];
// unlink [color="0.628 0.227 1.000"];
// filbuf [color="0.641 0.212 1.000"];
// open [color="0.641 0.212 1.000"];
// sbrk [color="0.647 0.204 1.000"];
// signal [color="0.647 0.204 1.000"];
// moncontrol [color="0.647 0.204 1.000"];
// xflsbuf [color="0.650 0.200 1.000"];
// flsbuf [color="0.650 0.200 1.000"];
// div [color="0.650 0.200 1.000"];
// cmpsave [color="0.650 0.200 1.000"];
// rem [color="0.650 0.200 1.000"];
// setfil [color="0.650 0.200 1.000"];
// close [color="0.650 0.200 1.000"];
// fclose [color="0.650 0.200 1.000"];
// fflush [color="0.650 0.200 1.000"];
// setbuf [color="0.650 0.200 1.000"];
// endopen [color="0.650 0.200 1.000"];
// findiop [color="0.650 0.200 1.000"];
// fopen [color="0.650 0.200 1.000"];
// mul [color="0.650 0.200 1.000"];
// newfile [color="0.650 0.200 1.000"];
// sigblock [color="0.650 0.200 1.000"];
// sigsetmask [color="0.650 0.200 1.000"];
// sigvec [color="0.650 0.200 1.000"];
// udiv [color="0.650 0.200 1.000"];
// urem [color="0.650 0.200 1.000"];
// brk [color="0.650 0.200 1.000"];
// getfreehdr [color="0.650 0.200 1.000"];
// strlen [color="0.650 0.200 1.000"];
// umul [color="0.650 0.200 1.000"];
// doprnt [color="0.650 0.200 1.000"];
// copyproto [color="0.650 0.200 1.000"];
// creat [color="0.650 0.200 1.000"];
// demote [color="0.650 0.200 1.000"];
// exit [color="0.650 0.200 1.000"];
// free [color="0.650 0.200 1.000"];
// getpagesize [color="0.650 0.200 1.000"];
// getpid [color="0.650 0.200 1.000"];
// initree [color="0.650 0.200 1.000"];
// insert [color="0.650 0.200 1.000"];
// localeconv [color="0.650 0.200 1.000"];
// main [color="0.650 0.200 1.000"];
// malloc [color="0.650 0.200 1.000"];
// morecore [color="0.650 0.200 1.000"];
// oldfile [color="0.650 0.200 1.000"];
// on_exit [color="0.650 0.200 1.000"];
// profil [color="0.650 0.200 1.000"];
// putfreehdr [color="0.650 0.200 1.000"];
// safeoutfil [color="0.650 0.200 1.000"];
// sprintf [color="0.650 0.200 1.000"];
// term [color="0.650 0.200 1.000"];
// }


// https://graphviz.org/Gallery/directed/unix.html
// /* courtesy Ian Darwin and Geoff Collyer, Softquad Inc. */
// digraph unix {
// 	fontname="Helvetica,Arial,sans-serif"
// 	node [fontname="Helvetica,Arial,sans-serif"]
// 	edge [fontname="Helvetica,Arial,sans-serif"]
// 	node [color=lightblue2, style=filled];
// 	"5th Edition" -> "6th Edition";
// 	"5th Edition" -> "PWB 1.0";
// 	"6th Edition" -> "LSX";
// 	"6th Edition" -> "1 BSD";
// 	"6th Edition" -> "Mini Unix";
// 	"6th Edition" -> "Wollongong";
// 	"6th Edition" -> "Interdata";
// 	"Interdata" -> "Unix/TS 3.0";
// 	"Interdata" -> "PWB 2.0";
// 	"Interdata" -> "7th Edition";
// 	"7th Edition" -> "8th Edition";
// 	"7th Edition" -> "32V";
// 	"7th Edition" -> "V7M";
// 	"7th Edition" -> "Ultrix-11";
// 	"7th Edition" -> "Xenix";
// 	"7th Edition" -> "UniPlus+";
// 	"V7M" -> "Ultrix-11";
// 	"8th Edition" -> "9th Edition";
// 	"1 BSD" -> "2 BSD";
// 	"2 BSD" -> "2.8 BSD";
// 	"2.8 BSD" -> "Ultrix-11";
// 	"2.8 BSD" -> "2.9 BSD";
// 	"32V" -> "3 BSD";
// 	"3 BSD" -> "4 BSD";
// 	"4 BSD" -> "4.1 BSD";
// 	"4.1 BSD" -> "4.2 BSD";
// 	"4.1 BSD" -> "2.8 BSD";
// 	"4.1 BSD" -> "8th Edition";
// 	"4.2 BSD" -> "4.3 BSD";
// 	"4.2 BSD" -> "Ultrix-32";
// 	"PWB 1.0" -> "PWB 1.2";
// 	"PWB 1.0" -> "USG 1.0";
// 	"PWB 1.2" -> "PWB 2.0";
// 	"USG 1.0" -> "CB Unix 1";
// 	"USG 1.0" -> "USG 2.0";
// 	"CB Unix 1" -> "CB Unix 2";
// 	"CB Unix 2" -> "CB Unix 3";
// 	"CB Unix 3" -> "Unix/TS++";
// 	"CB Unix 3" -> "PDP-11 Sys V";
// 	"USG 2.0" -> "USG 3.0";
// 	"USG 3.0" -> "Unix/TS 3.0";
// 	"PWB 2.0" -> "Unix/TS 3.0";
// 	"Unix/TS 1.0" -> "Unix/TS 3.0";
// 	"Unix/TS 3.0" -> "TS 4.0";
// 	"Unix/TS++" -> "TS 4.0";
// 	"CB Unix 3" -> "TS 4.0";
// 	"TS 4.0" -> "System V.0";
// 	"System V.0" -> "System V.2";
// 	"System V.2" -> "System V.3";
// }

// https://graphviz.org/Gallery/directed/world.html
// digraph world {
//     size="7,7";
//         {rank=same; S8 S24 S1 S35 S30;}
//         {rank=same; T8 T24 T1 T35 T30;}
//         {rank=same; 43 37 36 10 2;}
//         {rank=same; 25 9 38 40 13 17 12 18;}
//         {rank=same; 26 42 11 3 33 19 39 14 16;}
//         {rank=same; 4 31 34 21 41 28 20;}
//         {rank=same; 27 5 22 32 29 15;}
//         {rank=same; 6 23;}
//         {rank=same; 7;}
    
//         S8 -> 9;
//         S24 -> 25;
//         S24 -> 27;
//         S1 -> 2;
//         S1 -> 10;
//         S35 -> 43;
//         S35 -> 36;
//         S30 -> 31;
//         S30 -> 33;
//         9 -> 42;
//         9 -> T1;
//         25 -> T1;
//         25 -> 26;
//         27 -> T24;
//         2 -> {3 ; 16 ; 17 ; T1 ; 18}
//         10 -> { 11 ; 14 ; T1 ; 13; 12;}
//         31 -> T1;
//         31 -> 32;
//         33 -> T30;
//         33 -> 34;
//         42 -> 4;
//         26 -> 4;
//         3 -> 4;
//         16 -> 15;
//         17 -> 19;
//         18 -> 29;
//         11 -> 4;
//         14 -> 15;
//         37 -> {39 ; 41 ; 38 ; 40;}
//         13 -> 19;
//         12 -> 29;
//         43 -> 38;
//         43 -> 40;
//         36 -> 19;
//         32 -> 23;
//         34 -> 29;
//         39 -> 15;
//         41 -> 29;
//         38 -> 4;
//         40 -> 19;
//         4 -> 5;
//         19 -> {21 ; 20 ; 28;}
//         5 -> {6 ; T35 ; 23;}
//         21 -> 22;
//         20 -> 15;
//         28 -> 29;
//         6 -> 7;
//         15 -> T1;
//         22 -> T35;
//         22 -> 23;
//         29 -> T30;
//         7 -> T8;
//         23 -> T24;
//         23 -> T1;
//     }

pub fn dot_example_str(title: &str) -> &str {
    for (dot_title, dot) in DOT_EXAMPLES {
        if title == dot_title {
            return dot;
        }
    }
    panic!("Could not find requested example: {title}")
}

pub fn dot_example_graph(title: &str) -> Graph {
    let dot = dot_example_str(title);

    Graph::from(dot)
}
