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

## Building from source

Pre-built wheels are compiled for a generic target. You should however enable native optimizations (e.g., BMI2 for faster succinct data structures) by building the wheel from source:

```bash
pip install maturin
git clone https://github.com/vigna/webgraph-py.git
cd webgraph-py/webgraph
maturin build --release -o dist
pip install dist/*.whl
```

The repository's `.cargo/config.toml` sets `target-cpu=native`, so the
resulting wheel will be optimized for your CPU.

## Quick start

```python
import webgraph

# Load a compressed graph
g = webgraph.BvGraph("/PATH/TO/BASENAME")

print(f"Nodes: {g.num_nodes()}, Arcs: {g.num_arcs()}")

# Iterate over successors
for s in g.successors(42):
    print(s)

# Degree computation (returns a numpy array)
degrees = g.outdegrees()

# Top-k nodes by degree (returns a (k, 2) ndarray: column 0 = node, column 1 = degree)
top = g.top_k_out(10)
print(top[:, 0])  # node IDs

# Breadth-first search
for root, parent, node, distance in g.bfs_from_node(0):
    print(f"node={node} distance={distance}")
```
