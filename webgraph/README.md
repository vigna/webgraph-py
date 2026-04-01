# webgraph

Python bindings for the [Rust version](https://crates.io/crates/webgraph) of the
[WebGraph](https://webgraph.di.unimi.it/) framework, built with
[PyO3](https://pyo3.rs/).

WebGraph is a framework for graph compression that allows very large graphs
(billions of nodes and arcs) to be stored in a compact representation while
providing fast random access to successors.

## Installation

```bash
pip install webgraph
```

## Quick start

```python
import webgraph

# Load a compressed graph
g = webgraph.BvGraph("/PATH/TO/BASENAME")

print(f"Nodes: {g.num_nodes()}, Arcs: {g.num_arcs()}")

# Iterate over successors
for s in g.successors(42):
    print(s)

# Dgree computation (returns a numpy array)
degrees = g.outdegrees()

# Top-k nodes by degree (parallel computation)
top = g.top_k_out(10)

# Breadth-first search
for root, parent, node, distance in g.bfs_from_node(0):
    print(f"node={node} distance={distance}")
```
