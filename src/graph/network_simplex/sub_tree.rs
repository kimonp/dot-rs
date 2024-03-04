//! The subtree struct is used for constructing the initial feasible tree used to run
//! network simplex.

use super::heap::HeapIndex;
use core::cmp::Ordering;

use std::cell::RefCell;
use std::rc::Rc;

/// SubTree is analogous to the subtree_s struct found in GraphViz: lib/common/ns.c
///
/// The Subtree allows for internal mutability and cloning, which is convenieng for
/// recursive uses within network simplex.
#[derive(Debug, Clone, Eq)]
pub(crate) struct SubTree(Rc<RefCell<InternalSubTree>>);

impl SubTree {
    pub fn new(node_idx: usize) -> SubTree {
        SubTree(Rc::new(RefCell::new(InternalSubTree::new(node_idx))))
    }

    /// Number of nodes within the subtree.
    pub fn size(&self) -> u32 {
        (*self.0).borrow().size()
    }

    /// Node Index where this sub_tree started building.
    pub fn start_node_idx(&self) -> usize {
        (*self.0).borrow().start_node_idx()
    }

    /// Node Index of the parent of this subtree.
    pub fn parent(&self) -> Option<SubTree> {
        (*self.0).borrow().parent()
    }

    /// Set the size of this subtree (number of nodes within the subtree)
    pub fn set_size(&self, size: u32) {
        (*self.0).borrow_mut().set_size(size)
    }

    /// Set the size of this subtree (number of nodes within the subtree)
    pub fn set_parent(&self, parent: Option<SubTree>) {
        (*self.0).borrow_mut().set_parent(parent)
    }
    
    // Return the root of this sub_tree.
    //
    // If we don't have a parent, we are the root.
    // Otherwise, ask our parent.
    pub fn find_root(&self) -> SubTree {
        if let Some(parent) = self.parent() {
            parent.find_root()
        } else {
            self.clone()
        }
    }
}

impl PartialEq for SubTree {
    fn eq(&self, other: &Self) -> bool {
        self.size() == other.size()
    }
}

impl Ord for SubTree {
    fn cmp(&self, other: &Self) -> Ordering {
        self.size().cmp(&other.size())
    }
}

impl PartialOrd for SubTree {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The HeapIndex trait is needed to work with MinHeap.
impl HeapIndex for SubTree {
    fn heap_idx(&self) -> Option<usize> {
        (*self.0).borrow().heap_idx()
    }

    fn set_heap_idx(&mut self, new_index: Option<usize>) {
        (*self.0).borrow_mut().set_heap_idx(new_index);
    }
}

/// The analogy to the subtree_s struct found in GraphViz: lib/common/ns.c
#[derive(Debug, Clone, Eq)]
struct InternalSubTree {
    /// Index to the node in the tree where this sub_tree started to build.
    start_node_idx: usize,
    /// Total tight tree size of this sub_tree.
    size: u32,
    /// Required to find non-min elements when merged.
    ///
    /// In the original code, this is the pointer to the memory in the heap which
    /// holds the subtree memory.  This becomes -1 when the tree is no
    /// longer in the heap (the heap is one large fixed block of memory which
    /// contains pointers to subtree structs). It is used to:
    /// * Tell if a sub_tree is in the tree (not -1)
    /// * Effeciently fix the heap when the merge chanes a sub_tree's size
    /// node_idx, so heap_idx would alway be equal to node_idx.
    heap_idx: Option<usize>,
    /// Pointer to the parent of this sub_tree or none if this is the root of the sub_tree.
    parent: Option<SubTree>,
}

impl PartialEq for InternalSubTree {
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
    }
}

impl Ord for InternalSubTree {
    fn cmp(&self, other: &Self) -> Ordering {
        self.size.cmp(&other.size)
    }
}

impl PartialOrd for InternalSubTree {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl HeapIndex for InternalSubTree {
    fn heap_idx(&self) -> Option<usize> {
        self.heap_idx
    }

    fn set_heap_idx(&mut self, new_index: Option<usize>) {
        self.heap_idx = new_index;
    }
}

impl InternalSubTree {
    pub fn new(node_idx: usize) -> InternalSubTree {
        InternalSubTree {
            start_node_idx: node_idx,
            size: 0,
            heap_idx: None,
            parent: None,
        }
    }

    pub fn size(&self) -> u32 {
        self.size
    }

    pub fn start_node_idx(&self) -> usize {
        self.start_node_idx
    }

    pub fn parent(&self) -> Option<SubTree> {
        self.parent.clone()
    }

    pub fn set_size(&mut self, size: u32) {
        self.size = size;
    }

    pub fn set_parent(&mut self, parent: Option<SubTree>) {
        self.parent = parent;
    }
}
