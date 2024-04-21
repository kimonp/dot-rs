use std::{cmp::Ordering, collections::VecDeque};

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub(super) struct Line {
    min_x: u32,
    max_x: u32,
    line_type: LineType,
}

impl Line {
    /// Create a new line that assumes y1 is above y2.
    pub fn new_down(x1: u32, x2: u32) -> Line {
        let (min_x, max_x, line_type) = match x1.cmp(&x2) {
            Ordering::Less => (x1, x2, LineType::Down),
            Ordering::Equal => (x1, x2, LineType::Straight),
            Ordering::Greater => (x2, x1, LineType::Up),
        };

        Line::new(min_x, max_x, line_type)
    }

    /// Create a new line that assumes y1 is bellow y2.
    pub fn new_up(x1: u32, x2: u32) -> Line {
        let (min_x, max_x, line_type) = match x1.cmp(&x2) {
            Ordering::Less => (x1, x2, LineType::Up),
            Ordering::Equal => (x1, x2, LineType::Straight),
            Ordering::Greater => (x2, x1, LineType::Down),
        };

        Line::new(min_x, max_x, line_type)
    }

    pub fn new(min_x: u32, max_x: u32, line_type: LineType) -> Line {
        if line_type == LineType::Straight && min_x != max_x {
            panic!("Straight line is not straight");
        }

        Line {
            min_x,
            max_x,
            line_type,
        }
    }
}

impl Ord for Line {
    fn cmp(&self, other: &Self) -> Ordering {
        self.min_x.cmp(&other.min_x)
    }
}

impl PartialOrd for Line {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Copy, Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub(super) enum LineType {
    Up,
    Down,
    Straight,
}

pub(super) fn count_crosses(mut lines: Vec<Line>) -> u32 {
    let mut active: Vec<Line> = Vec::new();
    lines.sort();

    let mut num_crosses = 0;

    for new_line in lines.iter() {
        // Remove active lines that have passed
        let mut remove = VecDeque::new();
        for (index, line) in active.clone().iter().enumerate() {
            if line.max_x <= new_line.min_x {
                remove.push_front(index);
            }
        }
        for index in remove {
            active.remove(index);
        }

        // count crosses of new_line in active
        num_crosses += active
            .iter()
            .filter(|active_line| lines_are_crossed(new_line, active_line))
            .count() as u32;

        // add new line to the active set
        active.push(*new_line)
    }

    num_crosses
}

// Return true if the line new crosses the line active.
// Given: active.start_x <= new.start_x
fn lines_are_crossed(new: &Line, active: &Line) -> bool {
    // Both lines are Up or both lines are Down
    if new.line_type == active.line_type {
        active.min_x < new.min_x && active.max_x > new.max_x
    } else if active.line_type == LineType::Straight {
        false
    } else if new.line_type == LineType::Straight {
        active.min_x < new.min_x && active.max_x > new.max_x
        // active.max_x > new.max_x
        // One line is Up and the other is Down
    } else {
        active.max_x != new.min_x
    }
}

#[cfg(test)]
mod test {
    use crate::dot_examples::dot_example_str;
    use crate::graph::Graph;

    use super::*;
    use rstest::rstest;
    use LineType::Down;
    use LineType::Straight;
    use LineType::Up;

    #[rstest(lines, expected,
        case::parallel(             vec![(0, 1, Down), (1, 2, Down), (2, 3, Down)], 0),
        case::shared_dst_point(     vec![(0, 2, Down), (1, 2, Down), (2, 2, Straight)], 0),
        case::cross_straight_down(  vec![(0, 2, Down), (1, 1, Straight)], 1),
        case::cross_straight_up(    vec![(0, 2, Up), (1, 1, Straight)], 1),
        case::shared_point_up(      vec![(0, 1, Down), (1, 2, Up)], 0),
        case::cross_opp(            vec![(0, 1, Down), (0, 1, Up)], 1),
        case::all_cross(            vec![(0, 4, Down), (0, 4, Up), (1, 3, Down), (1, 3, Up), (2, 2, Straight)], 1+2+3+4),
        case::shared_src_point(     vec![(0, 2, Down), (0, 1, Down), (0, 0, Straight)], 0),
    )]
    fn test_lines_crossing(lines: Vec<(u32, u32, LineType)>, expected: u32) {
        let lines = lines
            .iter()
            .map(|(min, max, line_type)| Line::new(*min, *max, *line_type))
            .collect::<Vec<Line>>();

        let cross_count = count_crosses(lines);
        assert_eq!(cross_count, expected);
    }

    #[rstest(example_name, expected_crossings,
        case::complex_crossing("complex_crossing", 0),
        case::large_example("large_example", 2),
    )]
    fn test_example_crossings(example_name: &str, expected_crossings: u32) {
        let dot = dot_example_str(example_name);
        let mut graph = Graph::from(dot);

        graph.rank_nodes_vertically();
        graph.set_horizontal_ordering();

        assert_eq!(
            graph.rank_orderings.unwrap().crossing_count(),
            expected_crossings,
            "GraphViz crossing count is: {expected_crossings}"
        );
    }
}
