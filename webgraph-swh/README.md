# webgraph-swh

Python bindings for accessing [Software
Heritage](https://www.softwareheritage.org/) graphs, built on top of
[webgraph](https://pypi.org/project/webgraph/).

Provides bidirectional graph traversal with full access to node properties
(SWHIDs, timestamps, persons, messages) for the SWH compressed graph.

## Installation

```bash
pip install webgraph-swh
```

This automatically installs the `webgraph` package as a dependency.

## Building from source

Pre-built wheels are compiled for a generic target. You should however enable
native optimizations (e.g., BMI2 for faster succinct data structures)
by building the wheel from source:

```bash
pip install maturin
git clone https://github.com/vigna/webgraph-py.git
cd webgraph-py/webgraph-swh
maturin build --release -o dist
pip install dist/*.whl
```

The repository's `.cargo/config.toml` sets `target-cpu=native`, so the
resulting wheel will be optimized for your CPU. We suggest to use the
same approach for [`webgraph`](https://pypi.org/project/webgraph/).

## Quick start

```python
from webgraph_swh import SwhGraph, PyNodeType

# Load the SWH graph with all properties
g = SwhGraph("/path/to/swh-graph")

print(f"Nodes: {g.num_nodes()}")

# Traverse successors with properties
for s in g.successors(42):
    print(g.swhid(s), g.node_type(s))

# Filter by node type
revrel = g.subgraph("rev,rel")
for s in revrel.successors(42):
    print(revrel.author_id(s), revrel.author_timestamp(s))

# Access the underlying BvGraph
bv = g.forward_graph()
for root, parent, node, dist in bv.bfs():
    ...
```
