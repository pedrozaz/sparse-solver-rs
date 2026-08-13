use crate::matrix::CsrMatrix;
use std::f64::consts::PI;

/// Assembles the 2D Poisson equation system $A u = b$ on a grid of $N \times N$
/// interior points.
///
/// Uses a 5-point finite difference stencil with $h = 1 / (N + 1)$ and homogeneous Dirichlet
/// boundary conditions.
/// The RHS  corresponds to the source function $f(x, y) = 2\pi^2 \sin(\pi x) \sin(\pi y)$.
pub fn assemble_poisson_2d(n: usize) -> (CsrMatrix, Vec<f64>) {
    let num_eqs = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let h2 = h * h;

    let mut triplets = Vec::with_capacity(5 * num_eqs);
    let mut b = vec![0.0; num_eqs];

    for j in 0..n {
        let y = (j as f64 + 1.0) * h;
        for i in 0..n {
            let row = j * n + i;
            let x = (i as f64 + 1.0) * h;

            // Diagonal entry (stencil weight = 4.0)
            triplets.push((row, row, 4.0));

            // Left neighbor (i - 1)
            if i > 0 {
                triplets.push((row, row - 1, -1.0));
            }

            // Right neighbor (i + 1)
            if i + 1 < n {
                triplets.push((row, row + 1, -1.0));
            }

            // Bottom neighbor (j - 1)
            if j > 0 {
                triplets.push((row, row - n, -1.0));
            }

            // Top neighbor (j + 1)
            if j + 1 < n {
                triplets.push((row, row + n, -1.0));
            }

            // RHS source term f(x, y) * h^2
            let f_xy = 2.0 * PI * PI * (PI * x).sin() * (PI * y).sin();
            b[row] = h2 * f_xy;
        }
    }

    let a = CsrMatrix::from_triplets(num_eqs, num_eqs, &triplets);
    (a, b)
}

/// Evaluates the analytical exact solution $u(x, y) = \sin(\pi x) \sin(\pi y)$ on the $N \times N$
/// interior grid.
pub fn exact_poisson_solution(n: usize) -> Vec<f64> {
    let num_eqs = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let mut u_exact = vec![0.0; num_eqs];

    for j in 0..n {
        let y = (j as f64 + 1.0) * h;
        for i in 0..n {
            let row = j * n + i;
            let x = (i as f64 + 1.0) * h;
            u_exact[row] = (PI * x).sin() * (PI * y).sin();
        }
    }

    u_exact
}

/// Computes the discrete $L_2$ norm of error $\|u - u_{\text{exact}}\|_2 = \sqrt{h^2 \sum (u_i - u_{i,\text{exact}})^2}$.
pub fn compute_l2_error(u: &[f64], u_exact: &[f64], n: usize) -> f64 {
    let h = 1.0 / (n as f64 + 1.0);
    let sum_sq: f64 = u
        .iter()
        .zip(u_exact.iter())
        .map(|(&ui, &ue)| (ui - ue).powi(2))
        .sum();
    (h * h * sum_sq).sqrt()
}
