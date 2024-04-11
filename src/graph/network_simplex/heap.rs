//! A specialized min heap for network simplex that allows you to
//! change the size of one item in the heap and re-heapify the heap
//! effeciently.
//!
//! Items must support the HeapIndex trait, which lets the heap inform
//! the item of it's index into the heap so that re-order can be called
//! with the correct index.
pub(super) struct MinHeap<T> {
    data: Vec<T>,
}

pub(super) trait HeapIndex {
    fn heap_idx(&self) -> Option<usize>;
    fn set_heap_idx(&mut self, new_index: Option<usize>);
}

impl<T: Ord + HeapIndex + Eq> MinHeap<T> {
    /// Returns a heap with the given maximum capacity.
    pub fn new(capacity: usize) -> MinHeap<T> {
        MinHeap {
            data: Vec::with_capacity(capacity),
        }
    }

    /// Return the current number of items in the heap.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    #[allow(unused)]
    /// Return the maximum capacity of the heap.
    pub fn capacity(&self) -> usize {
        self.data.capacity()
    }

    /// Add an item to the heap, but don't order it.
    pub fn insert_unordered_item(&mut self, item: T) {
        self.data.push(item);
        let max_item = self.len() - 1;
        self.data[max_item].set_heap_idx(Some(max_item));
    }

    /// Reorders the item at heap_idx in the heap.
    ///
    /// Assumes item at heap_idx has increased in size and needs to
    /// be placed deaper in the heap.
    pub fn reorder_item(&mut self, heap_idx: usize) {
        let mut cur_item_idx = heap_idx;
        loop {
            let right = 2 * (cur_item_idx + 1);
            let left = right - 1;

            let mut smallest = cur_item_idx;
            if left < self.len() && self.data[left] < self.data[smallest] {
                smallest = left;
            }
            if right < self.len() && self.data[right] < self.data[smallest] {
                smallest = right;
            }

            // Keep going until cur_tem_idx is smaller than the left and the right.
            // If that's true, we've moved the item to the right place and we are done.
            if smallest != cur_item_idx {
                self.data.swap(cur_item_idx, smallest);

                self.data[cur_item_idx].set_heap_idx(Some(smallest));
                self.data[smallest].set_heap_idx(Some(cur_item_idx));

                cur_item_idx = smallest;
            } else {
                break;
            }

            if cur_item_idx >= self.len() {
                break;
            }
        }
    }

    /// Given a heap in random order, "heapifys" it so that pop will always return
    /// the smallest entry.
    pub fn order_heap(&mut self) {
        let midpoint = self.len() / 2;

        for heap_idx in (0..=midpoint).rev() {
            self.reorder_item(heap_idx)
        }
    }

    /// Remove and return the the top item in the heap.
    ///
    /// If the heap has been ordered, it will be the smallest item.
    pub fn pop(&mut self) -> Option<T> {
        if self.len() != 0 {
            let max_entry = self.len() - 1;
            self.data.swap(0, max_entry);
            let mut min_item = self.data.pop().unwrap();

            min_item.set_heap_idx(None);

            if self.len() != 0 {
                self.data[0].set_heap_idx(Some(0));
                self.reorder_item(0);
            }

            Some(min_item)
        } else {
            None
        }
    }

    /// Return a reference to the top item in the heap.
    ///
    /// If the heap has been ordered, it will be the smallest item.
    #[allow(dead_code)]
    pub fn peek(&self) -> Option<&T> {
        if self.len() != 0 {
            Some(&self.data[0])
        } else {
            None
        }
    }
}

#[cfg(test)]
mod test {
    use itertools::Itertools;

    use super::*;
    use std::{cmp::Ordering, fmt::Display};

    #[derive(Eq, Debug)]
    struct HeapTestItem {
        size: u32,
        index: Option<usize>,
    }

    impl HeapTestItem {
        fn new(size: u32) -> Self {
            Self { size, index: None }
        }
    }

    impl Display for HeapTestItem {
        fn fmt(&self, fmt: &mut std::fmt::Formatter<'_>) -> Result<(), std::fmt::Error> {
            write!(fmt, "{}", self.size)
        }
    }

    impl Ord for HeapTestItem {
        fn cmp(&self, other: &Self) -> Ordering {
            self.size.cmp(&other.size)
        }
    }

    impl PartialOrd for HeapTestItem {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl PartialEq for HeapTestItem {
        fn eq(&self, other: &Self) -> bool {
            self.size == other.size
        }
    }

    impl HeapIndex for HeapTestItem {
        fn heap_idx(&self) -> Option<usize> {
            self.index
        }

        fn set_heap_idx(&mut self, new_index: Option<usize>) {
            self.index = new_index;
        }
    }

    /// Create a vector in "random" order, and insert it into a heap.
    /// Ensure that it pop()s back in min order.
    ///
    /// This should effectively test the full heap flow:
    /// * new()
    /// * insert_unordered_item()
    /// * peek()
    /// * order_heap()
    /// * pop()
    #[test]
    fn test_heap_order() {
        let order = [10, 9, 13, 15, 17, 99, 3, 12, 21];
        let mut heap = MinHeap::new(order.len());
        let sorted_order = order.iter().cloned().sorted().collect::<Vec<u32>>();

        for item in order.iter() {
            heap.insert_unordered_item(HeapTestItem::new(*item));
        }
        assert_eq!(heap.peek(), Some(&HeapTestItem::new(order[0])));
        heap.order_heap();
        assert_eq!(heap.peek(), Some(&HeapTestItem::new(sorted_order[0])));

        for value in sorted_order {
            assert_eq!(heap.pop(), Some(HeapTestItem::new(value)));
        }
    }
}
