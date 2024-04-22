//! Examples of graphs to be used for testing and display.

use std::{
    fs::{read_dir, read_to_string},
    path::Path,
};

use crate::graph::Graph;

const EXAMPLE_DIR: &str = "dot_examples";

pub fn get_dot_example(file_name: &str) -> String {
    let path = format!("./{EXAMPLE_DIR}/{file_name}.dot");

    get_dot_example_from_path(&path)
}

/// Return all dot examples of the given category as: Vec<(name, dot_)>
pub fn get_dot_example_category(category: &str) -> Vec<(String, String)> {
    println!("Hello: {category}");
    read_dir(format!("./{EXAMPLE_DIR}/{category}"))
        .unwrap_or_else(|err| panic!("{err:?}"))
        .filter_map(|res| {
            let entry = res.unwrap_or_else(|err| panic!("invalid dir entry: {err:?}"));
            let path = entry.path();
            let file_name = path.to_str().expect("no filename in path");

            if file_name.ends_with(".dot") {
                Some((file_name.to_string(), get_dot_example_from_path(file_name)))
            } else {
                None
            }
        })
        .collect()
}

/// Return all dot examples of the given category as: Vec<(name, dot_)>
pub fn get_all_dot_examples() -> Vec<(String, String)> {
    visit_dirs(Path::new(&format!("./{EXAMPLE_DIR}")))
        .unwrap_or_else(|err| panic!("{err:?}"))
        .iter()
        .map(|file_name| (file_name.to_string(), get_dot_example_from_path(file_name)))
        .collect()
}

/// Given a path (in the form of a &str) return a string of the example in dot.
fn get_dot_example_from_path(path: &str) -> String {
    read_to_string(path)
        .unwrap_or_else(|err| panic!("could not open dot example file: {path}: {err:?}"))
}

fn visit_dirs(dir: &Path) -> std::io::Result<Vec<String>> {
    let mut files = vec![];

    if dir.is_dir() {
        for entry in read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.is_dir() {
                let mut sub_files = visit_dirs(&path)?;

                files.append(&mut sub_files);
            } else {
                let file_name = path.to_str().expect("no filename in path");

                if file_name.ends_with(".dot") {
                    files.push(file_name.to_string());
                }
            }
        }
    }
    Ok(files)
}

pub fn dot_example_graph(title: &str) -> Graph {
    let dot = get_dot_example(title);

    Graph::from(&dot)
}
