use crate::matrix::CsrMatrix;
use crate::solvers::{SolverOptions, SolverResult};

/// Unpreconditioned Conjugate Gradient (CG) solver for symmetric positive-definite
/// (SPD) linear systems $A x = b$
#[derive(Debug, Clone, Copy, Default)]
pub struct ConjugateGradient;

impl ConjugateGradient {
    pub fn solve(a: &CsrMatrix, b: &[f64], options: &SolverOptions) -> SolverResult {
        assert_eq!(
            a.nrows(),
            a.ncols(),
            "Matrix A must be square for Conjugate Gradient, got {}x{}",
            a.nrows(),
            a.ncols()
        );
        assert_eq!(
            b.len(),
            a.nrows(),
            "RHS vector b length {} does not match matrix rows {}",
            b.len(),
            a.nrows()
        );

        let n = a.nrows();
        let b_norm = dot_product(b, b).sqrt();

        if b_norm == 0.0 {
            return SolverResult {
                solution: vec![0.0; n],
                iterations: 0,
                residual_norm: 0.0,
                converged: true,
            };
        }

        let mut x = vec![0.0; n];
        let mut r = b.to_vec(); // r_0 = b - A*x_0 (since x_0 = 0)
        let mut p = r.clone(); // p_0 = r_0

        let mut r_norm_sq = dot_product(&r, &r);
        let mut iterations = 0;

        for k in 0..options.max_iter {
            iterations = k + 1;

            let r_norm_rel = r_norm_sq.sqrt() / b_norm;
            if r_norm_rel < options.tol {
                return SolverResult {
                    solution: x,
                    iterations: k,
                    residual_norm: r_norm_rel,
                    converged: true,
                };
            }

            let ap = a.spmv(&p); // SpMV kernel
            let p_ap = dot_product(&p, &ap);

            // Matrix is not positive-definite if p^T A p <= 0
            if p_ap <= 0.0 {
                return SolverResult {
                    solution: x,
                    iterations,
                    residual_norm: r_norm_sq.sqrt() / b_norm,
                    converged: false,
                };
            }

            let alpha = r_norm_sq / p_ap;

            // x = x + alpha * p
            for (x_i, &p_i) in x.iter_mut().zip(p.iter()) {
                *x_i += alpha * p_i;
            }

            // r = r - alpha * Ap
            for (r_i, &ap_i) in r.iter_mut().zip(ap.iter()) {
                *r_i -= alpha * ap_i;
            }

            let r_norm_sq_new = dot_product(&r, &r);
            let beta = r_norm_sq_new / r_norm_sq;
            r_norm_sq = r_norm_sq_new;

            // p = r + beta * p
            for (p_i, &r_i) in p.iter_mut().zip(r.iter()) {
                *p_i = r_i + beta * *p_i;
            }
        }

        let final_residual = r_norm_sq.sqrt() / b_norm;
        SolverResult {
            solution: x,
            iterations,
            residual_norm: final_residual,
            converged: final_residual < options.tol,
        }
    }
}

/// Helper function to compute vector dot product a . b
#[inline]
pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn test_cg_diagonal_matrix() {
        let triplets = vec![(0, 0, 2.0), (1, 1, 3.0), (2, 2, 4.0)];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let b = vec![4.0, 9.0, 16.0];

        let options = SolverOptions {
            max_iter: 100,
            tol: 1e-6,
        };
        let result = ConjugateGradient::solve(&a, &b, &options);

        assert!(result.converged);
        assert!((result.solution[0] - 2.0).abs() < 1e-5);
        assert!((result.solution[1] - 3.0).abs() < 1e-5);
        assert!((result.solution[2] - 4.0).abs() < 1e-5);
    }

    #[test]
    fn test_cg_1d_laplacian() {
        // 1D Laplacian 3x3:
        // [ 2, -1,  0]
        // [-1,  2, -1]
        // [ 0, -1,  2]
        let triplets = vec![
            (0, 0, 2.0),
            (0, 1, -1.0),
            (1, 0, -1.0),
            (1, 1, 2.0),
            (1, 2, -1.0),
            (2, 1, -1.0),
            (2, 2, 2.0),
        ];
        let a = CsrMatrix::from_triplets(3, 3, &triplets);
        let b = vec![1.0, 0.0, 1.0]; // Exact solution is [1.0, 1.0, 1.0]

        let options = SolverOptions {
            max_iter: 100,
            tol: 1e-6,
        };
        let result = ConjugateGradient::solve(&a, &b, &options);

        assert!(result.converged);
        for &val in &result.solution {
            assert!((val - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_cg_zero_rhs() {
        let triplets = vec![(0, 0, 2.0), (1, 1, 3.0)];
        let a = CsrMatrix::from_triplets(2, 2, &triplets);
        let b = vec![0.0, 0.0];

        let options = SolverOptions::default();
        let result = ConjugateGradient::solve(&a, &b, &options);

        assert!(result.converged);
        assert_eq!(result.iterations, 0);
        assert_eq!(result.solution, vec![0.0, 0.0]);
    }

    proptest! {
        #[test]
        fn prop_cg_converges_for_spd_matrix(
            n in 2..=10usize,
            b_entries in prop::collection::vec(-10.0..10.0f64, 2..=10)
        ) {
            let dim = n.min(b_entries.len());
            let b = &b_entries[..dim];

            // Construct SPD matrix A = 1D Laplacian-like tridiagonal
            let mut triplets = Vec::new();
            for i in 0..dim {
                triplets.push((i, i, 4.0));
                if i + 1 < dim {
                    triplets.push((i, i + 1, -1.0));
                    triplets.push((i + 1, i, -1.0));
                }
            }

            let a = CsrMatrix::from_triplets(dim, dim, &triplets);
            let options = SolverOptions {
                max_iter: 200,
                tol: 1e-6,
            };

            let result = ConjugateGradient::solve(&a, b, &options);
            prop_assert!(result.converged);

            // Verify residual ||b - Ax|| < tol * ||b||
            let ax = a.spmv(&result.solution);
            let diff: Vec<f64> = b.iter().zip(ax.iter()).map(|(&bi, &axi)| bi - axi).collect();
            let residual_norm = dot_product(&diff, &diff).sqrt();
            let b_norm = dot_product(b, b).sqrt();

            if b_norm > 1e-9 {
                prop_assert!(residual_norm / b_norm < 1e-4);
            }
        }
    }
}
