# Change Log

## [0.2.0] - 2026-10-12

### Changed

- `webgraph` now uses the Rust `webgraph` 0.7.0 crate, whose Elias–Fano
  offsets use a new file format: all `.ef` files must be rebuilt with
  `webgraph build ef` (`webgraph-cli` 0.5.0 or later). `webgraph-swh` is
  unaffected, as it follows the versions used by `swh-graph`.

### Improved

- The error raised by `BvGraph` when loading fails now reports the whole
  chain of causes (e.g., an `.ef` file in an obsolete format).

## [0.1.4] - 2026-07-23

### Changed

- License is now MIT or Apache.

## [0.1.3] - 2026-04-02

### Changed

- Top-k methods now return an `ndarray`.

- All `into_par_iter` calls on node ranges are followed by `with_min_len(1000)`.

- All metadata accessors of a filtered graph now raise an error when called on a
  node that is not in the subgraph.

## [0.1.2] - 2026-04-01

### Changed

- Binaries for x86_64 now require a `x86-64-v3` (post-Haswell) architecture.
