//! With Graph::new_from_str(), creates a new graph given a string in the dot language.
//!
//! For details on dot, see: <https://graphviz.org/doc/info/lang.html>
use pest::iterators::Pair;
use pest::iterators::Pairs;

use crate::graph::Graph;
use std::collections::HashMap;

#[derive(Parser)]
#[grammar = "dot.pest"]
pub struct DotParser;

use pest::Parser;
use pest_derive::Parser;

impl Graph {
    /// Given a string in the dot language, build a graph.
    ///
    /// Uses the Pest crate to do all the parsing.
    /// * A unsuccessful parse will cause a panic.
    /// * Going beyond the basic node layout of dot may also cause a panic
    ///   NOTE: Though the full grammar is in the dot.pest grammar file,
    ///         The only portion of dot that is currently supported is reading
    ///         in edge statements (e.g. node_1 -> node2;)
    pub fn new_from_str(dot_str: &str) -> Self {
        let dot_graph = DotParser::parse(Rule::dotgraph, dot_str)
            .expect("unsuccessful parse of dot string")
            .next()
            .unwrap();

        let mut nodes = HashMap::new();
        let mut edges = Vec::new();
        let mut node_attrs= HashMap::new();

        for record in dot_graph.into_inner() {
            match record.as_rule() {
                Rule::dotgraph => (),
                Rule::digraph => (),
                Rule::stmt_list => {
                    let statements = record.into_inner();
                    let statement_list = edges_from_stmt_list(statements.clone());
                    let node_list = nodes_from_stmt_list(statements);

                    for (src_node, dst_node) in statement_list {
                        edges.push((src_node.clone(), dst_node.clone()));
                    }
                    
                    for (node_name, attrs) in node_list {
                        node_attrs.insert(node_name, attrs);
                    }
                }
                Rule::ident => {
                }
                _ => unreachable!("Found rule: {:?}", record.as_rule()),
            }
        }
        
        let mut graph = Graph::new();

        for (src_node_name, dst_node_name) in edges.iter() {
            let src_node_idx = if let Some(node_idx) = nodes.get(src_node_name) {
                *node_idx
            } else {
                let node_idx = graph.add_node(src_node_name);
                nodes.insert(src_node_name, node_idx);

                node_idx
            };
            let dst_node_idx = if let Some(node_idx) = nodes.get(dst_node_name) {
                *node_idx
            } else {
                let node_idx = graph.add_node(dst_node_name);
                nodes.insert(dst_node_name, node_idx);

                node_idx
            };
            graph.add_edge(src_node_idx, dst_node_idx);
        }
        graph.set_node_attr(node_attrs);
        graph
    }
}

impl From<&str> for Graph {
    fn from(str: &str) -> Self {
        Self::new_from_str(str)
    }
}

impl From<&String> for Graph {
    fn from(string: &String) -> Self {
        Self::new_from_str(string)
    }
}

/// Given an edge_stmt (e.g. a -> b;), get the src and dst node names.
fn get_edge(edge_stmt: Pair<'_, Rule>) -> (String, String) {
    let mut edge_rule = edge_stmt.into_inner();
    let src_node_id = edge_rule.next().unwrap();
    let src_node_ident = src_node_id.into_inner().next().unwrap();
    let src_node_text = src_node_ident.as_span().as_str().to_string();

    let mut edge_rhs = edge_rule.next().unwrap().into_inner();
    let edge_op = edge_rhs.next().unwrap();
    let dst_node_text = edge_op.as_span().as_str().to_string();

    (src_node_text, dst_node_text)
}

/// Given a stmt_list, return a vector of src and dst node names from edge_stmts
/// in the stmt_list.
///
/// * All other statements types cause a panic.
/// * This can be addressed by handling additional cases in the "unreachable" arm
///   of the match.
fn edges_from_stmt_list(mut statements: Pairs<'_, Rule>) -> Vec<(String, String)> {
    let mut result = vec![];
    if let Some(statement) = statements.next() {
        match statement.as_rule() {
            Rule::stmt => {
                let edge = statement.into_inner().next().unwrap();
                
                if edge.as_rule() == Rule::edge_stmt {
                    let edge_pair = get_edge(edge);
                    result.push(edge_pair);
                    let mut list = edges_from_stmt_list(statements);
                    result.append(&mut list);
                } else if edge.as_rule() == Rule::node_stmt {
                    // ignore
                    let mut list = edges_from_stmt_list(statements);
                    result.append(&mut list);
                } else {
                    unreachable!("Uncovered rule: {:?}", edge.as_rule());
                }
            }
            Rule::stmt_list => {
                let mut sub_result = edges_from_stmt_list(statement.into_inner());
                
                result.append(&mut sub_result);
            },
            _ => unreachable!(),
        }
    }
    
    result
}

/// Given an edge_stmt (e.g. a -> b;), get the src and dst node names.
fn get_node(node_stmt: Pair<'_, Rule>) -> (String, HashMap<String, String>) {
    let mut node_rule = node_stmt.into_inner();
    let src_node_id = node_rule.next().unwrap();
    let src_node_ident = src_node_id.into_inner().next().unwrap();
    let src_node_text = src_node_ident.as_span().as_str().to_string();

    let mut attrs = HashMap::new();
    let attr_list = node_rule.next().unwrap().into_inner();
    
    for attr in attr_list {
        if attr.as_rule() == Rule::a_list {
            let mut cur_key: Option<String> = None;
            for ident in attr.into_inner() {
                // TODO: This is a bit messed up because idents are recursive for lists, so
                //       we also need to look for a_lists at this level, too.  As written, this
                //       will only find the first attribute...
                if ident.as_rule() == Rule::ident {
                    let ident_str = ident.as_span().as_str().to_string();
                    
                    if let Some(key) = cur_key {
                        let quoted = if ident_str.len() > 2 {
                            &ident_str[1..ident_str.len()-1]
                        } else {
                            ""
                        };

                        attrs.insert(key, quoted.to_string());
                        cur_key = None;
                    } else {
                        cur_key = Some(ident_str);
                    }
                }
            }
        }
    }

    (src_node_text, attrs)
}

fn nodes_from_stmt_list(mut statements: Pairs<'_, Rule>) -> Vec<(String, HashMap<String, String>)> {
    let mut result = vec![];
    if let Some(statement) = statements.next() {
        match statement.as_rule() {
            Rule::stmt => {
                let node = statement.into_inner().next().unwrap();
                
                if node.as_rule() == Rule::node_stmt {
                    let node_list = get_node(node);
                    result.push(node_list);
                    let mut list = nodes_from_stmt_list(statements);
                    result.append(&mut list);
                } else if node.as_rule() == Rule::edge_stmt {
                    // ignore
                    let mut list = nodes_from_stmt_list(statements);
                    result.append(&mut list);
                } else {
                    unreachable!("Uncovered rule: {:?}", node.as_rule());
                }
            }
            Rule::stmt_list => {
                let mut sub_result = nodes_from_stmt_list(statement.into_inner());
                
                result.append(&mut sub_result);
            },
            _ => unreachable!(),
        }
    }
    
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse() {
        let graph = Graph::new_from_str("digraph { a -> b; a -> c; b -> d; c -> d;}");

        println!("***** Done:\n{graph}");
    }
}
