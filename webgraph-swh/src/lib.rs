use dary_heap::QuaternaryHeap;
use epserde::deser::{DeserInner, ReaderWithPos, check_header};
use numpy::PyArray1;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::cmp::Reverse;
use std::fs::File;
use std::io::BufReader;
use std::path::PathBuf;
use std::rc::Rc;
use sux::array::partial_array;
use sux::bits::BitVec;
use sux::dict::EliasFano;
use sux::rank_sel::SelectZeroAdaptConst;
use sux::traits::SuccUnchecked;
use swh_graph::graph::{
    SwhBackwardGraph, SwhBidirectionalGraph, SwhForwardGraph, SwhGraph as SwhGraphTrait,
    SwhGraphWithProperties,
};
use swh_graph::{NodeConstraint, NodeType, properties::*};

type MyProperties = SwhGraphProperties<
    MappedMaps<swh_graph::mph::DynMphf>,
    MappedTimestamps,
    MappedPersons,
    NoContents,
    MappedStrings,
    NoLabelNames,
>;

type NamesMap = partial_array::PartialArray<Box<str>, partial_array::SparseIndex<Box<[usize]>>>;
type SparseEf = EliasFano<SelectZeroAdaptConst<BitVec<Box<[usize]>>, Box<[usize]>, 12, 3>>;

/// Return the `k` nodes with the highest degree, computed in parallel.
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

/// A bidirectional Software Heritage graph with node properties.
///
/// Loads the graph and all available properties (maps, persons, strings,
/// timestamps) from the given base path. Node IDs are integers in
/// ``[0, num_nodes)``.
///
/// Example::
///
///     g = SwhGraph("/path/to/graph")
///     for s in g.successors(42):
///         print(g.swhid(s), g.node_type(s))
#[pyclass(unsendable)]
pub struct SwhGraph {
    graph: Rc<SwhBidirectionalGraph<MyProperties>>,
    basepath: String,
}

/// Sparse map from contributor IDs to display names.
///
/// Loads a PartialArray-based name map from a binary file produced by
/// the swh-collab pipeline. Not every contributor ID has a name; missing
/// entries return ``None``.
///
/// Example::
///
///     names = ContributorNamesMap("/path/to/names.bin")
///     names.get_name(42)  # "Jane Doe" or None
#[pyclass(unsendable)]
pub struct ContributorNamesMap {
    ef: SparseEf,
    first_invalid_pos: usize,
    values: Box<[Box<str>]>,
}

impl ContributorNamesMap {
    /// TODO: replace with PartialArray methods
    fn lookup_name(&self, id: usize) -> Option<&str> {
        if id >= self.first_invalid_pos {
            return None;
        }
        let (index, pos) = unsafe { self.ef.succ_unchecked::<false>(id) };
        if pos != id {
            return None;
        }
        self.values.get(index).map(|s| s.as_ref())
    }
}

/// SWH node types.
///
/// Integer values match the encoding used in the SWH graph:
/// Content=0, Directory=1, Origin=2, Release=3, Revision=4, Snapshot=5.
#[pyclass(frozen, eq, eq_int, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum PyNodeType {
    Content = 0,
    Directory = 1,
    Origin = 2,
    Release = 3,
    Revision = 4,
    Snapshot = 5,
}

impl From<NodeType> for PyNodeType {
    fn from(nt: NodeType) -> Self {
        match nt {
            NodeType::Content => PyNodeType::Content,
            NodeType::Directory => PyNodeType::Directory,
            NodeType::Origin => PyNodeType::Origin,
            NodeType::Release => PyNodeType::Release,
            NodeType::Revision => PyNodeType::Revision,
            NodeType::Snapshot => PyNodeType::Snapshot,
        }
    }
}

/// A view of an SwhGraph restricted to specific node types.
///
/// Created by ``SwhGraph.subgraph()``. Node IDs are not renumbered:
/// they remain the same as in the original graph. Iteration over
/// successors and predecessors skips nodes that do not match the
/// constraint.
///
/// Example::
///
///     g = SwhGraph("/path/to/graph")
///     revrel = g.subgraph("rev,rel")
///     for s in revrel.successors(42):
///         print(revrel.author_id(s))
#[pyclass(unsendable)]
pub struct FilteredSwhGraph {
    graph: Rc<SwhBidirectionalGraph<MyProperties>>,
    constraint: NodeConstraint,
}

#[pymethods]
impl SwhGraph {
    /// Load the SWH graph and all properties from the given base path.
    #[new]
    #[pyo3(text_signature = "(path)")]
    pub fn new(path: String) -> PyResult<Self> {
        let graph = SwhBidirectionalGraph::new(PathBuf::from(&path))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .init_properties()
            .load_properties(|props| props.load_maps::<swh_graph::mph::DynMphf>())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .load_properties(|props| props.load_persons())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .load_properties(|props| props.load_strings())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            .load_properties(|props| props.load_timestamps())
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(SwhGraph {
            graph: Rc::new(graph),
            basepath: path,
        })
    }

    /// The base path from which the graph was loaded.
    #[getter]
    pub fn basepath(&self) -> &str {
        &self.basepath
    }

    /// Return the number of nodes in the graph.
    #[pyo3(text_signature = "()")]
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Return the number of successors of the given node.
    #[pyo3(text_signature = "(node)")]
    pub fn outdegree(&self, node: usize) -> usize {
        self.graph.outdegree(node)
    }

    /// Return the number of predecessors of the given node.
    #[pyo3(text_signature = "(node)")]
    pub fn indegree(&self, node: usize) -> usize {
        self.graph.indegree(node)
    }

    /// Return an iterator over the predecessors of the given node.
    #[pyo3(text_signature = "(node)")]
    pub fn predecessors(&self, node: usize) -> PySuccessorsIterator {
        let graph = self.graph.clone();
        // SAFETY: same as BvGraph::successors — Rc keeps graph alive,
        // `iter` is dropped before `_keepalive`.
        let graph_ref: &'static SwhBidirectionalGraph<MyProperties> =
            unsafe { &*Rc::as_ptr(&graph) };
        PySuccessorsIterator {
            iter: Box::new(graph_ref.predecessors(node)),
            _keepalive: graph,
        }
    }

    /// Return an iterator over the successors of the given node.
    #[pyo3(text_signature = "(node)")]
    pub fn successors(&self, node: usize) -> PySuccessorsIterator {
        let graph = self.graph.clone();
        let graph_ref: &'static SwhBidirectionalGraph<MyProperties> =
            unsafe { &*Rc::as_ptr(&graph) };
        PySuccessorsIterator {
            iter: Box::new(graph_ref.successors(node)),
            _keepalive: graph,
        }
    }

    /// Return the committer person ID, or ``None`` if not available.
    #[pyo3(text_signature = "(node)")]
    pub fn committer_id(&self, node: usize) -> Option<u32> {
        self.graph.properties().committer_id(node)
    }

    /// Return the author person ID, or ``None`` if not available.
    #[pyo3(text_signature = "(node)")]
    pub fn author_id(&self, node: usize) -> Option<u32> {
        self.graph.properties().author_id(node)
    }

    /// Return the node type as a ``PyNodeType`` enum value.
    #[pyo3(text_signature = "(node)")]
    pub fn node_type(&self, node: usize) -> PyNodeType {
        self.graph.properties().node_type(node).into()
    }

    /// Return the committer timestamp (seconds since epoch), or ``None``.
    #[pyo3(text_signature = "(node)")]
    pub fn committer_timestamp(&self, node: usize) -> Option<i64> {
        self.graph.properties().committer_timestamp(node)
    }

    /// Return the author timestamp (seconds since epoch), or ``None``.
    #[pyo3(text_signature = "(node)")]
    pub fn author_timestamp(&self, node: usize) -> Option<i64> {
        self.graph.properties().author_timestamp(node)
    }

    /// Return the SWHID of the given node as a string.
    #[pyo3(text_signature = "(node)")]
    pub fn swhid(&self, node: usize) -> String {
        self.graph.properties().swhid(node).to_string()
    }

    /// Return the URL of the Software Heritage archive page for the given
    /// node (e.g., ``https://archive.softwareheritage.org/swh:1:rev:...``).
    #[pyo3(text_signature = "(node)")]
    pub fn swh_link(&self, node: usize) -> String {
        format!(
            "https://archive.softwareheritage.org/{}",
            self.graph.properties().swhid(node)
        )
    }

    /// Return the commit/tag message, or ``None`` if not available.
    #[pyo3(text_signature = "(node)")]
    pub fn message(&self, node: usize) -> Option<String> {
        self.graph
            .properties()
            .message(node)
            .map(|m| String::from_utf8_lossy(&m).to_string())
    }

    /// Return the tag name, or ``None`` if not a release or not available.
    #[pyo3(text_signature = "(node)")]
    pub fn tag_name(&self, node: usize) -> Option<String> {
        self.graph
            .properties()
            .tag_name(node)
            .map(|m| String::from_utf8_lossy(&m).to_string())
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

    /// Return a numpy ``uint32`` array of indegrees for all nodes, computed
    /// in parallel. The array is indexed by node ID.
    #[pyo3(text_signature = "()")]
    pub fn indegrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let graph = &*self.graph;
        let degrees = py.detach(|| {
            (0..graph.num_nodes())
                .into_par_iter()
                .map(|n| graph.indegree(n) as u32)
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

    /// Return the ``k`` nodes with the highest indegree as a list of
    /// ``(node, indegree)`` pairs sorted by decreasing degree.
    #[pyo3(text_signature = "(k)")]
    pub fn top_k_in(&self, py: Python<'_>, k: usize) -> Vec<(usize, u32)> {
        let graph = &*self.graph;
        py.detach(|| top_k(graph.num_nodes(), k, |n| graph.indegree(n) as u32))
    }

    /// Return a FilteredSwhGraph restricted to the given node types.
    ///
    /// The constraint string is a comma-separated list of type
    /// abbreviations (``cnt``, ``dir``, ``ori``, ``rel``, ``rev``,
    /// ``snp``) or ``*`` for all types.
    ///
    /// Example::
    ///
    ///     revrel = g.subgraph("rev,rel")
    #[pyo3(text_signature = "(node_types)")]
    pub fn subgraph(&self, node_types: &str) -> PyResult<FilteredSwhGraph> {
        let constraint = node_types
            .parse::<NodeConstraint>()
            .map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(FilteredSwhGraph {
            graph: self.graph.clone(),
            constraint,
        })
    }

    /// Load the forward BvGraph from the same base path.
    ///
    /// Returns a ``webgraph.BvGraph`` instance.
    #[pyo3(text_signature = "()")]
    pub fn forward_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let webgraph_mod = py.import("webgraph")?;
        let bvgraph_cls = webgraph_mod.getattr("BvGraph")?;
        bvgraph_cls.call1((self.basepath.clone(),))
    }

    /// Load the backward (transposed) BvGraph from the same base path.
    ///
    /// Returns a ``webgraph.BvGraph`` instance.
    #[pyo3(text_signature = "()")]
    pub fn backward_graph<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let webgraph_mod = py.import("webgraph")?;
        let bvgraph_cls = webgraph_mod.getattr("BvGraph")?;
        bvgraph_cls.call1((format!("{}-transposed", self.basepath),))
    }
}

#[pymethods]
impl FilteredSwhGraph {
    /// Return the number of nodes in the underlying (unfiltered) graph.
    #[pyo3(text_signature = "()")]
    pub fn num_nodes(&self) -> usize {
        self.graph.num_nodes()
    }

    /// Return the number of successors matching the node-type constraint.
    #[pyo3(text_signature = "(node)")]
    pub fn outdegree(&self, node: usize) -> usize {
        self.graph
            .successors(node)
            .filter(|&n| self.constraint.matches(self.graph.properties().node_type(n)))
            .count()
    }

    /// Return the number of predecessors matching the node-type constraint.
    #[pyo3(text_signature = "(node)")]
    pub fn indegree(&self, node: usize) -> usize {
        self.graph
            .predecessors(node)
            .filter(|&n| self.constraint.matches(self.graph.properties().node_type(n)))
            .count()
    }

    /// Return an iterator over successors matching the node-type constraint.
    #[pyo3(text_signature = "(node)")]
    pub fn successors(&self, node: usize) -> PySuccessorsIterator {
        let graph = self.graph.clone();
        let constraint = self.constraint;
        // SAFETY: same as SwhGraph::successors — Rc keeps graph alive,
        // `iter` is dropped before `_keepalive`.
        let graph_ref: &'static SwhBidirectionalGraph<MyProperties> =
            unsafe { &*Rc::as_ptr(&graph) };
        PySuccessorsIterator {
            iter: Box::new(
                graph_ref
                    .successors(node)
                    .filter(move |&n| constraint.matches(graph_ref.properties().node_type(n))),
            ),
            _keepalive: graph,
        }
    }

    /// Return an iterator over predecessors matching the node-type constraint.
    #[pyo3(text_signature = "(node)")]
    pub fn predecessors(&self, node: usize) -> PySuccessorsIterator {
        let graph = self.graph.clone();
        let constraint = self.constraint;
        // SAFETY: same as SwhGraph::predecessors.
        let graph_ref: &'static SwhBidirectionalGraph<MyProperties> =
            unsafe { &*Rc::as_ptr(&graph) };
        PySuccessorsIterator {
            iter: Box::new(
                graph_ref
                    .predecessors(node)
                    .filter(move |&n| constraint.matches(graph_ref.properties().node_type(n))),
            ),
            _keepalive: graph,
        }
    }

    /// Return a numpy array of filtered outdegrees for all nodes, computed in
    /// parallel.
    ///
    /// Nodes not matching the constraint have degree 0, so that
    /// ``array[node_id]`` is the filtered outdegree for matching nodes.
    #[pyo3(text_signature = "()")]
    pub fn outdegrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let graph = &*self.graph;
        let constraint = self.constraint;
        let degrees = py.detach(|| {
            (0..graph.num_nodes())
                .into_par_iter()
                .map(|n| {
                    if !constraint.matches(graph.properties().node_type(n)) {
                        return 0;
                    }
                    graph
                        .successors(n)
                        .filter(|&s| constraint.matches(graph.properties().node_type(s)))
                        .count() as u32
                })
                .collect::<Vec<u32>>()
        });
        PyArray1::from_vec(py, degrees)
    }

    /// Return a numpy array of filtered indegrees for all nodes, computed in
    /// parallel.
    ///
    /// Nodes not matching the constraint have degree 0, so that
    /// ``array[node_id]`` is the filtered indegree for matching nodes.
    #[pyo3(text_signature = "()")]
    pub fn indegrees<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u32>> {
        let graph = &*self.graph;
        let constraint = self.constraint;
        let degrees = py.detach(|| {
            (0..graph.num_nodes())
                .into_par_iter()
                .map(|n| {
                    if !constraint.matches(graph.properties().node_type(n)) {
                        return 0;
                    }
                    graph
                        .predecessors(n)
                        .filter(|&s| constraint.matches(graph.properties().node_type(s)))
                        .count() as u32
                })
                .collect::<Vec<u32>>()
        });
        PyArray1::from_vec(py, degrees)
    }

    /// Return the committer person ID, or ``None`` if not available.
    #[pyo3(text_signature = "(node)")]
    pub fn committer_id(&self, node: usize) -> Option<u32> {
        self.graph.properties().committer_id(node)
    }

    /// Return the author person ID, or ``None`` if not available.
    #[pyo3(text_signature = "(node)")]
    pub fn author_id(&self, node: usize) -> Option<u32> {
        self.graph.properties().author_id(node)
    }

    /// Return the node type as a ``PyNodeType`` enum value.
    #[pyo3(text_signature = "(node)")]
    pub fn node_type(&self, node: usize) -> PyNodeType {
        self.graph.properties().node_type(node).into()
    }

    /// Return the committer timestamp (seconds since epoch), or ``None``.
    #[pyo3(text_signature = "(node)")]
    pub fn committer_timestamp(&self, node: usize) -> Option<i64> {
        self.graph.properties().committer_timestamp(node)
    }

    /// Return the author timestamp (seconds since epoch), or ``None``.
    #[pyo3(text_signature = "(node)")]
    pub fn author_timestamp(&self, node: usize) -> Option<i64> {
        self.graph.properties().author_timestamp(node)
    }

    /// Return the SWHID of the given node as a string.
    #[pyo3(text_signature = "(node)")]
    pub fn swhid(&self, node: usize) -> String {
        self.graph.properties().swhid(node).to_string()
    }

    /// Return the URL of the Software Heritage archive page for the given
    /// node (e.g., ``https://archive.softwareheritage.org/swh:1:rev:...``).
    #[pyo3(text_signature = "(node)")]
    pub fn swh_link(&self, node: usize) -> String {
        format!(
            "https://archive.softwareheritage.org/{}",
            self.graph.properties().swhid(node)
        )
    }

    /// Return the commit/tag message, or ``None`` if not available.
    #[pyo3(text_signature = "(node)")]
    pub fn message(&self, node: usize) -> Option<String> {
        self.graph
            .properties()
            .message(node)
            .map(|m| String::from_utf8_lossy(&m).to_string())
    }

    /// Return the tag name, or ``None`` if not a release or not available.
    #[pyo3(text_signature = "(node)")]
    pub fn tag_name(&self, node: usize) -> Option<String> {
        self.graph
            .properties()
            .tag_name(node)
            .map(|m| String::from_utf8_lossy(&m).to_string())
    }

    /// Return the ``k`` nodes with the highest filtered outdegree as a list
    /// of ``(node, outdegree)`` pairs sorted by decreasing degree.
    ///
    /// Only nodes matching the constraint are considered.
    #[pyo3(text_signature = "(k)")]
    pub fn top_k_out(&self, py: Python<'_>, k: usize) -> Vec<(usize, u32)> {
        let graph = &*self.graph;
        let constraint = self.constraint;
        py.detach(|| {
            top_k(graph.num_nodes(), k, |n| {
                if !constraint.matches(graph.properties().node_type(n)) {
                    return 0;
                }
                graph
                    .successors(n)
                    .filter(|&s| constraint.matches(graph.properties().node_type(s)))
                    .count() as u32
            })
        })
    }

    /// Return the ``k`` nodes with the highest filtered indegree as a list
    /// of ``(node, indegree)`` pairs sorted by decreasing degree.
    ///
    /// Only nodes matching the constraint are considered.
    #[pyo3(text_signature = "(k)")]
    pub fn top_k_in(&self, py: Python<'_>, k: usize) -> Vec<(usize, u32)> {
        let graph = &*self.graph;
        let constraint = self.constraint;
        py.detach(|| {
            top_k(graph.num_nodes(), k, |n| {
                if !constraint.matches(graph.properties().node_type(n)) {
                    return 0;
                }
                graph
                    .predecessors(n)
                    .filter(|&s| constraint.matches(graph.properties().node_type(s)))
                    .count() as u32
            })
        })
    }

    /// Return whether the given node matches the node-type constraint.
    #[pyo3(text_signature = "(node)")]
    pub fn has_node(&self, node: usize) -> bool {
        node < self.graph.num_nodes()
            && self
                .constraint
                .matches(self.graph.properties().node_type(node))
    }
}

#[pymethods]
impl ContributorNamesMap {
    /// Load a contributor names map from the given binary file.
    #[new]
    #[pyo3(text_signature = "(path)")]
    pub fn new(path: String) -> PyResult<Self> {
        let file = File::open(PathBuf::from(path))
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let mut buf_reader = BufReader::new(file);
        let mut reader = ReaderWithPos::new(&mut buf_reader);
        // TODO: use epserde deserialization once we use the new sux
        //
        // Cannot use NamesMap::load_full due to epserde derive limitation:
        // the V: AsRef<[T]> bound on PartialArray conflicts with epserde's
        // deserialization type substitution (Box<str> -> &str).
        check_header::<NamesMap>(&mut reader)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let ef: SparseEf = unsafe { SparseEf::_deser_full_inner(&mut reader) }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let first_invalid_pos: usize = unsafe { usize::_deser_full_inner(&mut reader) }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let values: Box<[Box<str>]> = unsafe { <Box<[Box<str>]>>::_deser_full_inner(&mut reader) }
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self {
            ef,
            first_invalid_pos,
            values,
        })
    }

    /// Return the display name for the given contributor ID, or ``None``.
    #[pyo3(text_signature = "(contributor_id)")]
    pub fn get_name(&self, contributor_id: usize) -> Option<String> {
        self.lookup_name(contributor_id).map(|s| s.to_string())
    }
}

/// Python bindings for Software Heritage graph access.
#[pymodule]
fn _webgraph_swh(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<SwhGraph>()?;
    m.add_class::<FilteredSwhGraph>()?;
    m.add_class::<ContributorNamesMap>()?;
    m.add_class::<PyNodeType>()?;
    m.add_class::<PySuccessorsIterator>()?;
    Ok(())
}
