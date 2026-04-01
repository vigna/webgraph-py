use dsi_bitstream::prelude::BE;
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::collections::VecDeque;
use std::rc::Rc;
use sux::bits::BitVec;
use sux::traits::{BitVecOps, BitVecOpsMut};
use webgraph::prelude::*;
use webgraph_py_common::{PySuccessorsIterator, check_node, top_k, top_k_to_ndarray};

type LoadedBvGraph = webgraph::graphs::bvgraph::BvGraph<
    DynCodesDecoderFactory<BE, MmapHelper<u32>, webgraph::graphs::bvgraph::EF>,
>;

/// A compressed graph in WebGraph format.
///
/// Provides forward-only access (successors) and BFS traversal.
/// For backward (predecessor) access, load the transposed graph.
///
/// Example::
///
///     bv = BvGraph("/path/to/graph")
///     for root, parent, node, dist in bv.bfs():
///         ...
#[pyclass(unsendable)]
pub struct BvGraph {
    graph: Rc<LoadedBvGraph>,
    basename: String,
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
            basename,
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
    ///
    /// Raises ``IndexError`` if *node* is out of range.
    #[pyo3(text_signature = "(node)")]
    pub fn outdegree(&self, node: usize) -> PyResult<usize> {
        check_node(node, self.graph.num_nodes())?;
        Ok(self.graph.outdegree(node))
    }

    /// Return an iterator over the successors of the given node.
    ///
    /// Raises ``IndexError`` if *node* is out of range.
    #[pyo3(text_signature = "(node)")]
    pub fn successors(&self, node: usize) -> PyResult<PySuccessorsIterator> {
        check_node(node, self.graph.num_nodes())?;
        let graph = self.graph.clone();
        // SAFETY: The Rc in PySuccessorsIterator keeps the graph alive
        // for as long as the iterator exists. The `iter` field is dropped
        // before `_keepalive` due to declaration order.
        let graph_ref: &'static LoadedBvGraph = unsafe { &*Rc::as_ptr(&graph) };
        Ok(PySuccessorsIterator::new(
            Box::new(graph_ref.successors(node)),
            graph,
        ))
    }

    /// Return a numpy ``uint32`` array of outdegrees for all nodes, computed
    /// in parallel. The array is indexed by node ID.
    #[pyo3(text_signature = "()")]
    pub fn outdegrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let graph = &*self.graph;
        let degrees = py.detach(|| {
            (0..graph.num_nodes())
                .into_par_iter()
                .with_min_len(1000)
                .map(|n| graph.outdegree(n) as u32)
                .collect::<Vec<u32>>()
        });
        PyArray1::from_vec(py, degrees)
    }

    /// Return the ``k`` nodes with the highest outdegree as a numpy
    /// ``uint64`` array of shape ``(k, 2)`` where column 0 holds node IDs
    /// and column 1 holds outdegrees, sorted by decreasing degree.
    #[pyo3(text_signature = "(k)")]
    pub fn top_k_out<'py>(&self, py: Python<'py>, k: usize) -> Bound<'py, numpy::PyArray2<u64>> {
        let graph = &*self.graph;
        let result = py.detach(|| top_k(graph.num_nodes(), k, |n| graph.outdegree(n) as u32));
        top_k_to_ndarray(py, result)
    }

    /// BFS over all connected components.
    ///
    /// Yields ``(root, parent, node, distance)`` tuples where *root*
    /// identifies the BFS tree (starting node of the component),
    /// *parent* is the node from which *node* was discovered (equal to
    /// *node* for roots), and *distance* is the hop count from *root*.
    #[pyo3(text_signature = "()")]
    pub fn bfs(&self) -> PyBfsIterator {
        PyBfsIterator::new(self.graph.clone(), false, 0)
    }

    /// BFS from a single starting node.
    ///
    /// Yields ``(root, parent, node, distance)`` tuples where *root*
    /// is always the starting node, *parent* is the node from which
    /// *node* was discovered (equal to *node* for the root), and
    /// *distance* is the hop count from the starting node.
    ///
    /// Raises ``IndexError`` if *node* is out of range.
    #[pyo3(text_signature = "(node)")]
    pub fn bfs_from_node(&self, node: usize) -> PyResult<PyBfsIterator> {
        check_node(node, self.graph.num_nodes())?;
        Ok(PyBfsIterator::new(self.graph.clone(), true, node))
    }

    fn __repr__(&self) -> String {
        format!(
            "BvGraph(basename={:?}, num_nodes={}, num_arcs={})",
            self.basename,
            self.graph.num_nodes(),
            self.graph.num_arcs()
        )
    }
}

/// Iterator for breadth-first traversal.
///
/// Yields ``(root, parent, node, distance)`` tuples where *root*
/// identifies the BFS tree, *parent* is the node from which *node*
/// was discovered (equal to *node* for roots), and *distance* is the
/// hop count from *root*.
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

/// Python bindings for the Rust version of the WebGraph framework.
#[pymodule]
fn _webgraph(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BvGraph>()?;
    m.add_class::<PySuccessorsIterator>()?;
    m.add_class::<PyBfsIterator>()?;
    Ok(())
}
