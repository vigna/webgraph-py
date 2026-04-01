# Change Log

## [0.1.3] - 2026-04-02

### Changed

- Top-k methods now return an `ndarray`.

- All `into_par_iter` calls on node ranges are followed by `with_min_len(1000)`.

- All metadata accessors of a filtered graph now raise an error when called on a
  node that is not in the subgraph.

## [0.1.2] - 2026-04-01

### Changed

- Binaries for x86_64 now require a `x86-64-v3` (post-Haswell) architecture.
