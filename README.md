# sparse-solver-rs

[![CI](https://github.com/pedroza/sparse-solver-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/pedroza/sparse-solver-rs/actions)
[![Crates.io](https://img.shields.io/crates/v/sparse-solver-rs.svg)](https://crates.io/crates/sparse-solver-rs)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-performance, idiomatic Rust library implementing iterative sparse linear system solvers (Conjugate Gradient and Geometric Multigrid), validated against the 2D Poisson equation.

---

## 📌 Overview

`sparse-solver-rs` is designed for numerical simulations, Finite Element Method (FEM), and Computational Fluid Dynamics (CFD) applications requiring fast solutions of large, sparse symmetric positive-definite (SPD) linear systems $A x = b$.

### Key Capabilities

- **Compressed Sparse Row (CSR)**: Memory-efficient sparse matrix representation optimized for fast matrix-vector multiplication (SpMV).
- **Conjugate Gradient (CG)**: Krylov subspace solver for symmetric positive-definite matrices with high numerical stability.
- **Geometric Multigrid (V-Cycle)**: Multi-level geometric solver achieving $O(N)$ linear-time convergence for structured 2D grids.
- **2D Poisson Validation**: Built-in 5-point finite difference discretization of the Poisson equation with Dirichlet boundary conditions.

---

## 🏗️ Architecture & Modules

```text
sparse-solver-rs/
├── src/
│   ├── lib.rs              # Library entry point & re-exports
│   ├── matrix/
│   │   ├── mod.rs          # Matrix module exports
│   │   └── csr.rs          # Compressed Sparse Row data structure & SpMV
│   ├── solvers/
│   │   ├── mod.rs          # Common solver traits and options
│   │   ├── cg.rs           # Unpreconditioned Conjugate Gradient
│   │   └── multigrid.rs    # Geometric Multigrid (V-Cycle)
│   └── poisson.rs          # 2D Poisson system assembly & exact solutions
├── benches/
│   └── solver_bench.rs     # Criterion benchmarks (SpMV vs CG vs Multigrid)
├── examples/
│   └── poisson_2d.rs       # Executable example comparing CG vs Multigrid
├── tests/
│   └── convergence.rs      # Integration tests and mesh refinement validation
└── docs/
    └── theory.md           # Mathematical foundations & algorithmic derivations
```

---

## 🚀 Quick Start

Add `sparse-solver-rs` to your `Cargo.toml`:

```toml
[dependencies]
sparse-solver-rs = "0.1.0"
```

### Basic Example

```rust
use sparse_solver_rs::matrix::CsrMatrix;
use sparse_solver_rs::solvers::cg::ConjugateGradient;
use sparse_solver_rs::solvers::SolverOptions;

fn main() {
    // Define a 3x3 symmetric positive-definite matrix in triplet format (row, col, value)
    let triplets = vec![
        (0, 0, 4.0), (0, 1, -1.0),
        (1, 0, -1.0), (1, 1, 4.0), (1, 2, -1.0),
        (2, 1, -1.0), (2, 2, 4.0),
    ];
    let a = CsrMatrix::from_triplets(3, 3, &triplets);
    let b = vec![15.0, 10.0, 10.0];

    // Configure solver options
    let options = SolverOptions {
        max_iter: 100,
        tol: 1e-6,
    };

    // Solve Ax = b
    let result = ConjugateGradient::solve(&a, &b, &options);

    assert!(result.converged);
    println!("Converged in {} iterations", result.iterations);
    println!("Solution: {:?}", result.solution);
}
```

---

## 📊 Benchmarks & Performance

Benchmarks measured on x86_64 Linux using `criterion` across grid sizes $N \times N$:

| Benchmark | $N = 7$ ($49$ DOFs) | $N = 15$ ($225$ DOFs) | $N = 31$ ($961$ DOFs) | $N = 63$ ($3969$ DOFs) |
|---|---|---|---|---|
| **SpMV (Kernel)** | 163 ns | 760 ns | 3.3 µs | 13.0 µs |
| **Conjugate Gradient (CG)** | 356 ns | 1.7 µs | 7.6 µs | 28.6 µs |
| **Multigrid (V-Cycle)** | 78 µs | 423 µs | 2.1 ms | 10.3 ms |

Run benchmarks locally:
```bash
cargo bench
```

---

## 📚 Theory & Mathematical Background

For a complete derivation of algorithms, matrix structures, smoothing factors, and the 5-point finite difference stencil, see [`docs/theory.md`](docs/theory.md).

---

## 📄 License

Distributed under the **MIT License**. See [`LICENSE`](LICENSE) for more information.
