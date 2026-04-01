use dary_heap::QuaternaryHeap;
use dsi_bitstream::prelude::BE;
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::collections::VecDeque;
use std::rc::Rc;
use sux::bits::BitVec;
use sux::traits::{BitVecOps, BitVecOpsMut};
use webgraph::prelude::*;

type LoadedBvGraph = webgraph::graphs::bvgraph::BvGraph<
    DynCodesDecoderFactory<BE, MmapHelper<u32>, webgraph::graphs::bvgraph::EF>,
>;

/// Return the `k` nodes with the highest degree, computed in parallel.
///
/// Each rayon task maintains a local quaternary min-heap of size `k`;
/// the heaps are then merged into a single top-`k` result sorted by
/// degree descending (ties broken by ascending node ID).
fn top_k(num_nodes: usize, k: usize, degree_fn: impl Fn(usize) -> u32 + Sync) -> Vec<(usize, u32)> {
    if k == 0 {
        return Vec::new();
    }
    let mut result: Vec<(usize, u32)> = (0..num_nodes)
        .into_par_iter()
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
        .reduce(
            QuaternaryHeap::new,
            |mut a, b| {
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
            },
        )
        .into_iter()
        .map(|Reverse((deg, n))| (n, deg))
        .collect();
    result.sort_unstable_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));
    result
}

/// Iterator over node IDs (successors or predecessors).
#[pyclass(unsendable)]
pub struct PySuccessorsIterator {
    // SAFETY: `iter` borrows from the graph held by `_keepalive`.
    // `iter` is declared before `_keepalive` so it is dropped first.
    iter: Box<dyn Iterator<Item = usize>>,
    _keepalive: Rc<dyn std::any::Any>,
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

/// A compressed graph in WebGraph format.
///
/// Provides forward-only access (successors) and BFS traversal.
///
/// Example::
///
///     bv = BvGraph("/path/to/graph")
///     for root, parent, node, dist in bv.bfs():
///         ...
#[pyclass(unsendable)]
pub struct BvGraph {
    graph: Rc<LoadedBvGraph>,
}

#[pymethods]
impl BvGraph {
    /// Load a BvGraph from the given base path.
    #[new]
    #[pyo3(text_signature = "(basename)")]
    pub fn new(basename: String) -> PyResult<Self> {
        let graph = webgraph::graphs::bvgraph::BvGraph::with_basename(&basename)
            .load()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            graph: Rc::new(graph),
        })
    }

    /// Return the number of nodes.
    #[pyo3(text_signature = "()")]
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Return the number of arcs.
    #[pyo3(text_signature = "()")]
    pub fn num_arcs(&self) -> u64 {
        self.graph.num_arcs()
    }

    /// Return the number of successors of the given node.
    #[pyo3(text_signature = "(node)")]
    pub fn outdegree(&self, node: usize) -> usize {
        self.graph.outdegree(node)
    }

    /// Return an iterator over the successors of the given node.
    #[pyo3(text_signature = "(node)")]
    pub fn successors(&self, node: usize) -> PySuccessorsIterator {
        let graph = self.graph.clone();
        // SAFETY: The Rc in PySuccessorsIterator keeps the graph alive
        // for as long as the iterator exists. The `iter` field is dropped
        // before `_keepalive` due to declaration order.
        let graph_ref: &'static LoadedBvGraph = unsafe { &*Rc::as_ptr(&graph) };
        PySuccessorsIterator {
            iter: Box::new(graph_ref.successors(node)),
            _keepalive: graph,
        }
    }

    /// Return a numpy ``uint32`` array of outdegrees for all nodes, computed
    /// in parallel. The array is indexed by node ID.
    #[pyo3(text_signature = "()")]
    pub fn outdegrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let graph = &*self.graph;
        let degrees = py.detach(|| {
            (0..graph.num_nodes())
                .into_par_iter()
                .map(|n| graph.outdegree(n) as u32)
                .collect::<Vec<u32>>()
        });
        PyArray1::from_vec(py, degrees)
    }

    /// Return the ``k`` nodes with the highest outdegree as a list of
    /// ``(node, outdegree)`` pairs sorted by decreasing degree.
    #[pyo3(text_signature = "(k)")]
    pub fn top_k_out(&self, py: Python<'_>, k: usize) -> Vec<(usize, u32)> {
        let graph = &*self.graph;
        py.detach(|| top_k(graph.num_nodes(), k, |n| graph.outdegree(n) as u32))
    }

    /// BFS over all connected components.
    ///
    /// Yields ``(root, parent, node, distance)`` tuples.
    #[pyo3(text_signature = "()")]
    pub fn bfs(&self) -> PyBfsIterator {
        PyBfsIterator::new(self.graph.clone(), false, 0)
    }

    /// BFS from a single starting node.
    ///
    /// Yields ``(root, parent, node, distance)`` tuples.
    #[pyo3(text_signature = "(node)")]
    pub fn bfs_from_node(&self, node: usize) -> PyBfsIterator {
        PyBfsIterator::new(self.graph.clone(), true, node)
    }
}

/// Iterator for breadth-first traversal.
///
/// Yields ``(root, parent, node, distance)`` tuples. When traversing
/// all components, ``root`` identifies which component the node
/// belongs to.
#[pyclass(unsendable)]
pub struct PyBfsIterator {
    graph: Rc<LoadedBvGraph>,
    visited: BitVec<Vec<usize>>,
    queue: VecDeque<(usize, usize, usize)>, // (parent, node, distance)
    current_root: usize,
    next_unvisited: usize,
    num_nodes: usize,
    single_component: bool,
}

impl PyBfsIterator {
    fn new(graph: Rc<LoadedBvGraph>, single_component: bool, start_node: usize) -> Self {
        let num_nodes = graph.num_nodes();
        let mut visited = BitVec::<Vec<usize>>::new(num_nodes);
        let mut queue = VecDeque::new();

        visited.set(start_node, true);
        queue.push_back((start_node, start_node, 0));

        Self {
            graph,
            visited,
            queue,
            current_root: start_node,
            next_unvisited: if single_component { num_nodes } else { 0 },
            num_nodes,
            single_component,
        }
    }
}

#[pymethods]
impl PyBfsIterator {
    fn __iter__(slf: PyRef<Self>) -> PyRef<Self> {
        slf
    }

    fn __next__(&mut self) -> Option<(usize, usize, usize, usize)> {
        loop {
            if let Some((parent, node, distance)) = self.queue.pop_front() {
                for succ in self.graph.successors(node) {
                    if !self.visited.get(succ) {
                        self.visited.set(succ, true);
                        self.queue.push_back((node, succ, distance + 1));
                    }
                }
                return Some((self.current_root, parent, node, distance));
            }

            if self.single_component {
                return None;
            }

            // Find next unvisited node for a new BFS tree
            while self.next_unvisited < self.num_nodes && self.visited.get(self.next_unvisited) {
                self.next_unvisited += 1;
            }

            if self.next_unvisited >= self.num_nodes {
                return None;
            }

            let root = self.next_unvisited;
            self.current_root = root;
            self.visited.set(root, true);
            self.queue.push_back((root, root, 0));
        }
    }
}

/// Python bindings for the WebGraph compressed graph framework.
#[pymodule]
fn _webgraph(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BvGraph>()?;
    m.add_class::<PySuccessorsIterator>()?;
    m.add_class::<PyBfsIterator>()?;
    Ok(())
}
