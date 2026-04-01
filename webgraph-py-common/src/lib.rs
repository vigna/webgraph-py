use dary_heap::QuaternaryHeap;
use numpy::PyArray2;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::rc::Rc;

/// Return the `k` nodes with the highest degree, computed in parallel.
///
/// Each rayon task maintains a local quaternary min-heap of size `k`;
/// the heaps are then merged into a single top-`k` result sorted by
/// degree descending (ties broken by ascending node ID).
pub fn top_k(
    num_nodes: usize,
    k: usize,
    degree_fn: impl Fn(usize) -> u32 + Sync,
) -> Vec<(usize, u32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut result: Vec<(usize, u32)> = (0..num_nodes)
        .into_par_iter()
        .with_min_len(num_nodes.isqrt())
        .fold(
            QuaternaryHeap::<Reverse<(u32, usize)>>::new,
            |mut heap, n| {
                let deg = degree_fn(n);
                if deg > 0 {
                    if heap.len() < k {
                        heap.push(Reverse((deg, n)));
                    } else {
                        let mut top = heap.peek_mut().unwrap();
                        if deg > top.0.0 {
                            *top = Reverse((deg, n));
                        }
                    }
                }
                heap
            },
        )
        .reduce(QuaternaryHeap::new, |mut a, b| {
            for Reverse((deg, n)) in b {
                if a.len() < k {
                    a.push(Reverse((deg, n)));
                } else {
                    let mut top = a.peek_mut().unwrap();
                    if deg > top.0.0 {
                        *top = Reverse((deg, n));
                    }
                }
            }
            a
        })
        .into_iter()
        .map(|Reverse((deg, n))| (n, deg))
        .collect();
    result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    result
}

/// Convert the output of [`top_k`] into a numpy ``uint64`` array of shape
/// ``(k, 2)`` where column 0 holds node IDs and column 1 holds degrees.
pub fn top_k_to_ndarray<'py>(
    py: Python<'py>,
    data: Vec<(usize, u32)>,
) -> Bound<'py, PyArray2<u64>> {
    let len = data.len();
    let mut arr = numpy::ndarray::Array2::<u64>::zeros((len, 2));
    for (i, (node, deg)) in data.into_iter().enumerate() {
        arr[[i, 0]] = node as u64;
        arr[[i, 1]] = deg as u64;
    }
    PyArray2::from_owned_array(py, arr)
}

/// Check that `node` is in [0 . . `num_nodes`), raising `IndexError` if not.
pub fn check_node(node: usize, num_nodes: usize) -> PyResult<()> {
    if node >= num_nodes {
        Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "node index {} out of range for graph with {} nodes",
            node, num_nodes
        )))
    } else {
        Ok(())
    }
}

/// Check that `node` is in [0 . . `num_nodes`) and matches the given
/// constraint, raising `IndexError` or `ValueError` respectively.
pub fn check_filtered_node(
    node: usize,
    num_nodes: usize,
    matches: bool,
    constraint_str: &str,
) -> PyResult<()> {
    check_node(node, num_nodes)?;
    if !matches {
        Err(pyo3::exceptions::PyValueError::new_err(format!(
            "node {} does not match the node-type constraint {:?}",
            node, constraint_str
        )))
    } else {
        Ok(())
    }
}

/// Iterator over node IDs (successors or predecessors).
///
/// # Safety
///
/// `iter` **must** be declared before `_keepalive` so that it is dropped
/// first. The iterator borrows from the graph held by the `Rc` in
/// `_keepalive`; reordering the fields would cause use-after-free.
#[pyclass(unsendable)]
pub struct PySuccessorsIterator {
    // INVARIANT: `iter` MUST be declared before `_keepalive` (drop order).
    iter: Box<dyn Iterator<Item = usize>>,
    _keepalive: Rc<dyn std::any::Any>,
}

impl PySuccessorsIterator {
    /// Create a new iterator.
    ///
    /// The caller must ensure that `iter` borrows only from data kept
    /// alive by `keepalive`.
    pub fn new(iter: Box<dyn Iterator<Item = usize>>, keepalive: Rc<dyn std::any::Any>) -> Self {
        Self {
            iter,
            _keepalive: keepalive,
        }
    }
}

#[pymethods]
impl PySuccessorsIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> Option<usize> {
        self.iter.next()
    }
}
