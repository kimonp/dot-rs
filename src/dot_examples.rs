//! Examples of graphs to be used for testing and display.

use std::{
    fs::{read_dir, read_to_string, create_dir_all}, path::Path
};

use crate::graph::Graph;

const EXAMPLE_DIR: &str = "dot_examples";

pub enum SvgGenerator {
    DotRs,
    GraphViz,
}

impl SvgGenerator {
    fn sub_dir(&self) -> &str {
        match self {
            SvgGenerator::DotRs => "dot-rs",
            SvgGenerator::GraphViz => "GraphViz",
        }
    }
}

/// Returns the file path of the example, creating all directories as needed
pub fn get_svg_example_path(example_name: &str, generator: SvgGenerator) -> String {
    let (example_path, svg_file) = example_path_and_file(example_name).expect("Cant open example");
    let sub_dir = generator.sub_dir();
    let example_dir = format!("{example_path}/generated/{sub_dir}");
    
    create_dir_all(&example_dir).expect("Could not create example directory");
    
    format!("{example_dir}/{svg_file}")
}

pub fn example_path_and_file(example_name: &str) -> Option<(String, String)> {
    let dot_path_string = example_file_to_path_string(example_name);
    let svg_path_string = format!("{}.svg", dot_path_string.strip_suffix(".dot").expect("must end in .dot"));
    let path = Path::new(&svg_path_string);
    let basename = path.file_name().and_then(|os_str| os_str.to_str()).map(|str| str.to_string());
    let dirname = path.parent().and_then(|path| path.to_str()).map(|path_str| path_str.to_string());
    
    match (dirname, basename) {
        (Some(dirname), Some(basename)) => Some((dirname, basename)),
        _ => None,
    }
}

pub fn get_dot_example(file_name: &str) -> String {
    let path = example_file_to_path_string(file_name);

    get_dot_example_from_path(&path)
}

fn example_file_to_path_string(file_name: &str) -> String {
    format!("./{EXAMPLE_DIR}/{file_name}.dot")
}

/// Return all dot examples of the given category as: Vec<(name, dot_)>
pub fn get_dot_example_category(category: &str) -> Vec<(String, String)> {
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
