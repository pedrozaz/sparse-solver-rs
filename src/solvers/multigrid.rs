/// Transfer operators (restrcition and prolongation) for geometric multigrid on 2D structured grids.
///
/// Restricts a vector from a fine grid ($_f \times n_f$) to a coarse grid ($n_c \times n_c$)
/// using 2D Full-Weighting restrcition (9-point stencil).
///
/// # Panics
/// Panics if $n_f$ is not odd or if $fin.len() != n_f \times n_f$.
#[allow(non_snake_case)]
pub fn restrict_2d(fine: &[f64], n_f: usize) -> Vec<f64> {
    assert!(
        n_f >= 3 && n_f % 2 == 1,
        "Fine grid dimension n_f must be an odd number >= 3, got {}",
        n_f
    );
    assert_eq!(
        fine.len(),
        n_f * n_f,
        "Fine vector length does not match n_f * n_f"
    );

    let n_c = (n_f - 1) / 2;
    let mut coarse = vec![0.0; n_c * n_c];

    // Helper closure to safely fetch fine grid value (returns 0.0 if out of bounds)
    let get_fine = |r: isize, c: isize| -> f64 {
        if r >= 0 && r < n_f as isize && c >= 0 && c < n_f as isize {
            fine[r as usize * n_f + c as usize]
        } else {
            0.0
        }
    };

    for J in 0..n_c {
        let j = (2 * J + 1) as isize;
        for I in 0..n_c {
            let i = (2 * I + 1) as isize;

            // 9-point Full-Weighting stencil weights
            let center = 4.0 * get_fine(j, i);
            let edges = 2.0
                * (get_fine(j, i - 1)
                    + get_fine(j, i + 1)
                    + get_fine(j - 1, i)
                    + get_fine(j + 1, i));
            let corners = 1.0
                * (get_fine(j - 1, i - 1)
                    + get_fine(j - 1, i + 1)
                    + get_fine(j + 1, i - 1)
                    + get_fine(j + 1, i + 1));

            coarse[J * n_c + I] = (center + edges + corners) / 16.0;
        }
    }

    coarse
}

/// Prolongates a vector from a coarse grid($n_c \times n_c$) to a fine grid ($n_f \times n_f$)
/// using 2D bilinear interpolation ($n_f = 2 n_c + 1$).
///
/// # Panics
/// Panics if $coarse.len() != n_c \times n_c$.
#[allow(non_snake_case)]
pub fn prolongate_2d(coarse: &[f64], n_c: usize) -> Vec<f64> {
    assert_eq!(
        coarse.len(),
        n_c * n_c,
        "coarse vector length does not match n_c * n_c"
    );

    let n_f = 2 * n_c + 1;
    let mut fine = vec![0.0; n_f * n_f];

    // Helper closure to safely fetch coarse grid value (returns 0.0 if out of bounds)
    let get_coarse = |R: isize, C: isize| -> f64 {
        if R >= 0 && R < n_c as isize && C >= 0 && C < n_c as isize {
            coarse[R as usize * n_c + C as usize]
        } else {
            0.0
        }
    };

    for j in 0..n_f {
        for i in 0..n_f {
            let fine_idx = j * n_f + i;

            if i % 2 == 1 && j % 2 == 1 {
                // Point coincides with coarse grid node
                let I = (i - 1) / 2;
                let J = (j - 1) / 2;
                fine[fine_idx] = get_coarse(J as isize, I as isize);
            } else if i % 2 == 0 && j % 2 == 1 {
                // Horizontal edge between (I-1, J) and (I, J)
                let J = (j - 1) / 2;
                let I_right = i / 2;
                let I_left = I_right as isize - 1;
                fine[fine_idx] = 0.5
                    * (get_coarse(J as isize, I_left) + get_coarse(J as isize, I_right as isize));
            } else if i % 2 == 1 && j % 2 == 0 {
                // Vertical edge between (I, J-1) and (I, J)
                let I = (i - 1) / 2;
                let J_top = j / 2;
                let J_bottom = J_top as isize - 1;
                fine[fine_idx] = 0.5
                    * (get_coarse(J_bottom, I as isize) + get_coarse(J_top as isize, I as isize));
            } else {
                // Center of 4 coarse grid nodes
                let I_right = i / 2;
                let I_left = I_right as isize - 1;
                let J_top = j / 2;
                let J_bottom = J_top as isize - 1;

                fine[fine_idx] = 0.25
                    * (get_coarse(J_bottom, I_left)
                        + get_coarse(J_bottom, I_right as isize)
                        + get_coarse(J_top as isize, I_left)
                        + get_coarse(J_top as isize, I_right as isize));
            }
        }
    }

    fine
}

use core::num;
use std::iter;

use crate::matrix::CsrMatrix;
use crate::poisson::assemble_poisson_2d;
use crate::solvers::{SolverOptions, SolverResult};

/// Geometric Multigrid (V-Cycle) solver for 2D Poisson equation on structured grids.
#[derive(Debug, Clone, Copy, Default)]
pub struct MultigridSolver {
    /// Number of pre-smoothing iterations (nu_1).
    pub nu1: usize,
    /// Number of post-smoothing iterations (nu_2).
    pub nu2: usize,
    /// Relaxation factor for weighted Jacobi smoother (typically 2/3 for 2D Poisson).
    pub omega: f64,
}

impl MultigridSolver {
    /// Creates a new `MultigridSolver` with default parameters (nu1=2, nu2=2, omega=2/3).
    pub fn new() -> Self {
        Self {
            nu1: 2,
            nu2: 2,
            omega: 2.0 / 3.0,
        }
    }

    /// Solves the 2D Poisson system $A u = b$ on an $N \times N$ interior grid using V-Cycles.
    ///
    /// # Panics
    /// Panics if $N$ is not an odd integer >= 3.
    pub fn solve(&self, n: usize, b: &[f64], options: &SolverOptions) -> SolverResult {
        assert!(
            n >= 3 && n % 2 == 1,
            "Grid dimension N must be an odd integer >= 3, got {}",
            n
        );
        let num_eqs = n * n;
        assert_eq!(b.len(), num_eqs, "RHS vector length does not match N * N");

        let (a, _) = assemble_poisson_2d(n);
        let b_norm = b.iter().map(|&v| v * v).sum::<f64>().sqrt();

        if b_norm == 0.0 {
            return SolverResult {
                solution: vec![0.0; num_eqs],
                iterations: 0,
                residual_norm: 0.0,
                converged: true,
            };
        }

        let mut x = vec![0.0; num_eqs];
        let mut iterations = 0;

        for k in 0..options.max_iter {
            iterations = k + 1;

            // Compute current residual r = b - A*x
            let ax = a.spmv(&x);
            let r: Vec<f64> = b
                .iter()
                .zip(ax.iter())
                .map(|(&bi, &axi)| bi - axi)
                .collect();
            let r_norm = r.iter().map(|&v| v * v).sum::<f64>().sqrt();
            let r_rel = r_norm / b_norm;

            if r_rel < options.tol {
                return SolverResult {
                    solution: x,
                    iterations: k,
                    residual_norm: r_rel,
                    converged: true,
                };
            }

            // Perform 1 V-Cycle
            self.v_cycle(n, b, &mut x);
        }

        let ax = a.spmv(&x);
        let r_norm = b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| (bi - axi).powi(2))
            .sum::<f64>()
            .sqrt();
        let final_rel = r_norm / b_norm;

        SolverResult {
            solution: x,
            iterations,
            residual_norm: final_rel,
            converged: final_rel < options.tol,
        }
    }

    /// Performs one V-Cycle recursively on a grid of size $n \times n$.
    fn v_cycle(&self, n: usize, b: &[f64], x: &mut [f64]) {
        let (a, _) = assemble_poisson_2d(n);

        // 1. pre-smoothing
        weighted_jacobi_smooth(&a, b, x, self.nu1, self.omega);

        // Base case: if coarse grid cannot be coarsened further (n <= 3), smooth heavily.
        if n <= 3 {
            weighted_jacobi_smooth(&a, b, x, 20, self.omega);
            return;
        }

        // 2. Compute residual r = b - A*x
        let ax = a.spmv(x);
        let r_fine: Vec<f64> = b
            .iter()
            .zip(ax.iter())
            .map(|(&bi, &axi)| bi - axi)
            .collect();

        // 3. Restrict residual to coarse grid
        let r_coarse = restrict_2d(&r_fine, n);
        let n_c = (n - 1) / 2;

        // 4. Coarse grid error correction (initial guess e_coarse = 0)
        let mut e_coarse = vec![0.0; n_c * n_c];
        self.v_cycle(n_c, &r_coarse, &mut e_coarse);

        // 5. Prolongate error correction to fine grid
        let e_fine = prolongate_2d(&e_coarse, n_c);

        // 6. Update fine grid solution x = x + e_fine
        for (xi, &ei) in x.iter_mut().zip(e_fine.iter()) {
            *xi += ei;
        }

        // 7. post-smoothing
        weighted_jacobi_smooth(&a, b, x, self.nu2, self.omega);
    }
}

/// Weighted Jacobi smoother for Poisson 2D equation $A x = b$.
pub fn weighted_jacobi_smooth(a: &CsrMatrix, b: &[f64], x: &mut [f64], iters: usize, omega: f64) {
    let n = x.len();
    let mut x_new = x.to_vec();

    for _ in 0..iters {
        let ax = a.spmv(x);
        for i in 0..n {
            // For 5-point Poisson stencil, diagonal entry A_{i,i} is 4.0
            let r_i = b[i] - ax[i];
            x_new[i] = x[i] + (omega / 4.0) * r_i;
        }
        x.copy_from_slice(&x_new);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prolongate_constant_vector() {
        let n_c = 3;
        let coarse = vec![1.0; n_c * n_c];

        let fine = prolongate_2d(&coarse, n_c);
        let n_f = 2 * n_c + 1;
        assert_eq!(fine.len(), n_f * n_f);

        // Center node of fine grid (i=3, j=3) should be exactly 1.0
        let center_idx = 3 * n_f + 3;
        assert!((fine[center_idx] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_restrict_constant_vector() {
        let n_f = 7;
        let fine = vec![1.0; n_f * n_f];

        let coarse = restrict_2d(&fine, n_f);
        let n_c = (n_f - 1) / 2;
        assert_eq!(coarse.len(), n_c * n_c);

        // Center node of coarse grid (I=1, J=1) should be 1.0
        let center_idx = n_c + 1;
        assert!((coarse[center_idx] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_prolongate_restrict_consistency() {
        let n_c = 3;
        let coarse_orig = vec![2.5; n_c * n_c];

        let fine = prolongate_2d(&coarse_orig, n_c);
        let n_f = 2 * n_c + 1;
        let coarse_restr = restrict_2d(&fine, n_f);

        // Interior coarse center node (1, 1) should remain 2.5
        assert!((coarse_restr[n_c + 1] - 2.5).abs() < 1e-12);
    }

    #[test]
    #[should_panic]
    fn test_restrict_even_dimension_panics() {
        let fine = vec![1.0; 16];
        restrict_2d(&fine, 4); // Even n_f must panic
    }
}
