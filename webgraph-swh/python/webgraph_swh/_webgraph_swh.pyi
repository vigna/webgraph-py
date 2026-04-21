"""Python bindings for Software Heritage graph access."""

from typing import Iterator

import numpy as np
import numpy.typing as npt
from webgraph import BvGraph

class PyNodeType:
    """SWH node types.

    Integer values match the encoding used in the SWH graph:
    Content=0, Directory=1, Origin=2, Release=3, Revision=4, Snapshot=5.
    """

    Content: int
    Directory: int
    Origin: int
    Release: int
    Revision: int
    Snapshot: int

class PySuccessorsIterator:
    """Iterator over node IDs (successors or predecessors)."""

    def __iter__(self) -> PySuccessorsIterator: ...
    def __next__(self) -> int: ...

class PyBfsIterator:
    """Iterator for breadth-first traversal.

    Yields ``(root, parent, node, distance)`` tuples. When traversing
    all components, ``root`` identifies which component the node
    belongs to.
    """

    def __iter__(self) -> PyBfsIterator: ...
    def __next__(self) -> tuple[int, int, int, int]: ...

class SwhGraph:
    """A bidirectional Software Heritage graph with node properties.

    Loads the graph and all available properties (maps, persons, strings,
    timestamps) from the given base path. Node IDs are integers in
    [0 . . `num_nodes`).
    """

    def __init__(self, path: str) -> None:
        """Load the SWH graph and all properties from the given base path."""
        ...

    @property
    def basepath(self) -> str:
        """The base path from which the graph was loaded."""
        ...

    def num_nodes(self) -> int:
        """Return the number of nodes in the graph."""
        ...

    def outdegree(self, node: int) -> int:
        """Return the number of successors of the given node."""
        ...

    def indegree(self, node: int) -> int:
        """Return the number of predecessors of the given node."""
        ...

    def predecessors(self, node: int) -> PySuccessorsIterator:
        """Return an iterator over the predecessors of the given node."""
        ...

    def successors(self, node: int) -> PySuccessorsIterator:
        """Return an iterator over the successors of the given node."""
        ...

    def committer_id(self, node: int) -> int | None:
        """Return the committer person ID, or ``None`` if not available."""
        ...

    def find_commit_from_committer(self, committer_id: int) -> int | None:
        """Return a node whose committer person ID matches
        *committer_id*.

        Scans nodes in parallel and may stop early. Returns ``None`` if no
        match is found.
        """
        ...

    def author_id(self, node: int) -> int | None:
        """Return the author person ID, or ``None`` if not available."""
        ...

    def find_commit_from_author(self, author_id: int) -> int | None:
        """Return a node whose author person ID matches *author_id*.

        Scans nodes in parallel and may stop early. Returns ``None`` if no
        match is found.
        """
        ...

    def node_type(self, node: int) -> PyNodeType:
        """Return the node type as a ``PyNodeType`` enum value."""
        ...

    def committer_timestamp(self, node: int) -> int | None:
        """Return the committer timestamp (seconds since epoch), or ``None``."""
        ...

    def author_timestamp(self, node: int) -> int | None:
        """Return the author timestamp (seconds since epoch), or ``None``."""
        ...

    def swhid(self, node: int) -> str:
        """Return the SWHID of the given node as a string."""
        ...

    def swh_link(self, node: int) -> str:
        """Return the URL of the Software Heritage archive page for the given
        node (e.g., ``https://archive.softwareheritage.org/swh:1:rev:...``).
        """
        ...

    def message(self, node: int) -> str | None:
        """Return the commit/tag message, or ``None`` if not available."""
        ...

    def tag_name(self, node: int) -> str | None:
        """Return the tag name, or ``None`` if not a release or not available."""
        ...

    def outdegrees(self) -> npt.NDArray[np.uint32]:
        """Return a numpy ``uint32`` array of outdegrees for all nodes, computed
        in parallel. The array is indexed by node ID.
        """
        ...

    def indegrees(self) -> npt.NDArray[np.uint32]:
        """Return a numpy ``uint32`` array of indegrees for all nodes, computed
        in parallel. The array is indexed by node ID.
        """
        ...

    def top_k_out(self, k: int) -> list[tuple[int, int]]:
        """Return the ``k`` nodes with the highest outdegree as a list of
        (*node*, *outdegree*) pairs sorted by decreasing degree.
        """
        ...

    def top_k_in(self, k: int) -> list[tuple[int, int]]:
        """Return the ``k`` nodes with the highest indegree as a list of
        (*node*, *indegree*) pairs sorted by decreasing degree.
        """
        ...

    def node_type_frequencies(self) -> npt.NDArray[np.uint64]:
        """Return a numpy ``uint64`` array indexed by ``PyNodeType`` values,
        with frequencies of each node type.
        """
        ...

    def bfs(self) -> PyBfsIterator:
        """BFS over all connected components.

        Yields ``(root, parent, node, distance)`` tuples.
        """
        ...

    def bfs_from_node(self, node: int) -> PyBfsIterator:
        """BFS from a single starting node.

        Yields ``(root, parent, node, distance)`` tuples.
        """
        ...

    def subgraph(self, node_types: str) -> FilteredSwhGraph:
        """Return a FilteredSwhGraph restricted to the given node types.

        The constraint string is a comma-separated list of type
        abbreviations (``cnt``, ``dir``, ``ori``, ``rel``, ``rev``,
        ``snp``) or ``*`` for all types.
        """
        ...

    def forward_graph(self) -> BvGraph:
        """Load the forward BvGraph from the same base path.

        Returns a ``webgraph.BvGraph`` instance.
        """
        ...

    def backward_graph(self) -> BvGraph:
        """Load the backward (transposed) BvGraph from the same base path.

        Returns a ``webgraph.BvGraph`` instance.
        """
        ...

class FilteredSwhGraph:
    """A view of an SwhGraph restricted to specific node types.

    Created by ``SwhGraph.subgraph()``. Node IDs are not renumbered:
    they remain the same as in the original graph. Iteration over
    successors and predecessors skips nodes that do not match the
    constraint.
    """

    def num_nodes(self) -> int:
        """Return the number of nodes in the underlying (unfiltered) graph."""
        ...

    def precise_num_nodes(self) -> int:
        """Return the number of nodes matching the node-type constraint."""
        ...

    def outdegree(self, node: int) -> int:
        """Return the number of successors matching the node-type constraint."""
        ...

    def indegree(self, node: int) -> int:
        """Return the number of predecessors matching the node-type constraint."""
        ...

    def successors(self, node: int) -> PySuccessorsIterator:
        """Return an iterator over successors matching the node-type constraint."""
        ...

    def predecessors(self, node: int) -> PySuccessorsIterator:
        """Return an iterator over predecessors matching the node-type constraint."""
        ...

    def outdegrees(self) -> npt.NDArray[np.uint32]:
        """Return a numpy array of filtered outdegrees for all nodes, computed in
        parallel.
        """
        ...

    def indegrees(self) -> npt.NDArray[np.uint32]:
        """Return a numpy array of filtered indegrees for all nodes, computed in
        parallel.
        """
        ...

    def node_type_frequencies(self) -> npt.NDArray[np.uint64]:
        """Return a numpy ``uint64`` array indexed by ``PyNodeType`` values,
        with frequencies of matching nodes for each node type.
        """
        ...

    def committer_id(self, node: int) -> int | None:
        """Return the committer person ID, or ``None`` if not available."""
        ...

    def find_commit_from_committer(self, committer_id: int) -> int | None:
        """Return a node whose committer person ID matches
        *committer_id*.

        Scans nodes in parallel and may stop early. Returns ``None`` if no
        match is found.
        Only nodes matching the node-type constraint are considered.
        """
        ...

    def author_id(self, node: int) -> int | None:
        """Return the author person ID, or ``None`` if not available."""
        ...

    def find_commit_from_author(self, author_id: int) -> int | None:
        """Return a node whose author person ID matches *author_id*.

        Scans nodes in parallel and may stop early. Returns ``None`` if no
        match is found.
        Only nodes matching the node-type constraint are considered.
        """
        ...

    def node_type(self, node: int) -> PyNodeType:
        """Return the node type as a ``PyNodeType`` enum value."""
        ...

    def committer_timestamp(self, node: int) -> int | None:
        """Return the committer timestamp (seconds since epoch), or ``None``."""
        ...

    def author_timestamp(self, node: int) -> int | None:
        """Return the author timestamp (seconds since epoch), or ``None``."""
        ...

    def swhid(self, node: int) -> str:
        """Return the SWHID of the given node as a string."""
        ...

    def swh_link(self, node: int) -> str:
        """Return the URL of the Software Heritage archive page for the given
        node.
        """
        ...

    def message(self, node: int) -> str | None:
        """Return the commit/tag message, or ``None`` if not available."""
        ...

    def tag_name(self, node: int) -> str | None:
        """Return the tag name, or ``None`` if not a release or not available."""
        ...

    def top_k_out(self, k: int) -> list[tuple[int, int]]:
        """Return the ``k`` nodes with the highest filtered outdegree as a list
        of (*node*, *outdegree*) pairs sorted by decreasing degree.
        """
        ...

    def top_k_in(self, k: int) -> list[tuple[int, int]]:
        """Return the ``k`` nodes with the highest filtered indegree as a list
        of (*node*, *indegree*) pairs sorted by decreasing degree.
        """
        ...

    def has_node(self, node: int) -> bool:
        """Return whether the given node matches the node-type constraint."""
        ...

class ContributorNamesMap:
    """Sparse map from contributor IDs to display names."""

    def __init__(self, path: str) -> None:
        """Load a contributor names map from the given binary file."""
        ...

    def get_name(self, contributor_id: int) -> str | None:
        """Return the display name for the given contributor ID, or ``None``."""
        ...
