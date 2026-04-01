# WebGraph Bindings

The packages in this repo act as a bridge between Python and software
based on the [Rust port of WebGraph](https://github.com/vigna/webgraph-rs).

- [`webgraph`](./webgraph) provides bindings to the Rust library, allowing
  Python users to access its functionality.

- [`webgraph-swh`](./webgraph-swh) offers a Python interface to the [Software Heritage
  graph](https://archive.softwareheritage.org/), which is built using
  WebGraph.
