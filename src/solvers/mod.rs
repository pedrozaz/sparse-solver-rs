pub mod cg;
pub mod multigrid;

pub use cg::ConjugateGradient;
pub use multigrid::{prolongate_2d, restrict_2d};

/// Configuration options for iterative linear system solvers.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverOptions {
    /// Maximum number of iterations allowed.
    pub max_iter: usize,
    /// Relative residual tolerance for convergence criteria (e.g. 1e-6).
    pub tol: f64,
}

impl Default for SolverOptions {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

/// Statistics and output result of an iterative solver execution.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverResult {
    /// Approximate solution vector x.
    pub solution: Vec<f64>,
    /// Total iterations performed.
    pub iterations: usize,
    /// Final relative residual norm ||b - Ax|| / ||b||.
    pub residual_norm: f64,
    /// True if residual norm met tolerance within max_iter.
    pub converged: bool,
}
