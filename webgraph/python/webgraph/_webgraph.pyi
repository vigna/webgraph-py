"""Python bindings for the WebGraph compressed graph framework."""

import numpy as np
import numpy.typing as npt

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

class BvGraph:
    """A compressed graph in WebGraph format.

    Provides forward-only access (successors) and BFS traversal.
    """

    def __init__(self, basename: str) -> None:
        """Load a BvGraph from the given base path."""
        ...

    def num_nodes(self) -> int:
        """Return the number of nodes."""
        ...

    def num_arcs(self) -> int:
        """Return the number of arcs."""
        ...

    def outdegree(self, node: int) -> int:
        """Return the number of successors of the given node."""
        ...

    def successors(self, node: int) -> PySuccessorsIterator:
        """Return an iterator over the successors of the given node."""
        ...

    def outdegrees(self) -> npt.NDArray[np.uint32]:
        """Return a numpy ``uint32`` array of outdegrees for all nodes, computed
        in parallel. The array is indexed by node ID.
        """
        ...

    def top_k_out(self, k: int) -> list[tuple[int, int]]:
        """Return the ``k`` nodes with the highest outdegree as a list of
        ``(node, outdegree)`` pairs sorted by decreasing degree.
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
