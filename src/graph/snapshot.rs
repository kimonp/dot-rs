

//! Snapshots for graphs: used for debugging
//!
//! Saves an SVG of the graph during various points in layout.

type SnapShotVec = Vec<(String, String)>;

#[derive(Debug, Clone, Default)]
pub struct Snapshots {
    total: usize,
    groups: Vec<(String, SnapShotVec)>,
}

impl Snapshots {
    pub fn new() -> Snapshots {
        Snapshots {
            total: 0,
            groups: Vec::new(),
        }
    }

    /// Add a new group to save snapshots to.
    ///
    /// New snapshots can only be added to the current group
    pub fn new_group(&mut self, group_title: &str) {
        self.groups.push((group_title.to_string(), Vec::new()));
    }

    /// Add a new snapshot to the current group.
    pub fn add(&mut self, title: &str, snap: &str) {
        if let Some(last) = self.groups.last_mut() {
            last.1.push((title.to_string(), snap.to_string()));
        } else {
            panic!("No snapshot group set!")
        }
        self.total += 1;
    }

    pub fn get(&self, frame: usize) -> Option<(String, String, String)> {
        let mut cur_frame = 0_usize;
        for (title, group) in self.groups.iter() {
            if cur_frame + group.len() > frame {
                let index = frame - cur_frame;
                let snapshot = group.get(index).expect("element not present");

                return Some((title.clone(), snapshot.0.clone(), snapshot.1.clone()));
            }
            cur_frame += group.len();
        }
        None
    }

    pub fn group_count(&self) -> usize {
        self.groups.len()
    }

    pub fn total_count(&self) -> usize {
        self.total
    }

    pub fn steps(&self, frame: usize) -> (usize, usize) {
        let mut cur_frame = 0_usize;
        let mut prev_frame = 0_usize;
        for (_title, group) in self.groups.iter() {
            if cur_frame + group.len() > frame {
                let start_frame = if frame == cur_frame {
                    prev_frame
                } else {
                    cur_frame
                };
                return (start_frame, cur_frame + group.len());
            }
            prev_frame = cur_frame;
            cur_frame += group.len();
        }
        (0, self.total_count())
    }
}
