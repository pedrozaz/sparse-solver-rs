use sparse_solver_rs::poisson::{assemble_poisson_2d, compute_l2_error, exact_poisson_solution};
use sparse_solver_rs::solvers::SolverOptions;
use sparse_solver_rs::solvers::cg::ConjugateGradient;

#[test]
fn test_poisson_2d_spatial_convergence_order() {
    let sizes = [8, 16, 32];
    let mut errors = Vec::new();

    for &n in &sizes {
        let (a, b) = assemble_poisson_2d(n);
        let u_exact = exact_poisson_solution(n);

        let options = SolverOptions {
            max_iter: 1000,
            tol: 1e-10,
        };

        let result = ConjugateGradient::solve(&a, &b, &options);
        assert!(
            result.converged,
            "CG solver failed to converge for grid N={}",
            n
        );
        let l2_err = compute_l2_error(&result.solution, &u_exact, n);
        errors.push(l2_err);
    }

    // Verify 2nd order spatial accuracy O(h²):
    // When N doubles (h halves), error should be decrease by a factor of ~4.
    let ratio_8_to_16 = errors[0] / errors[1];
    let ratio_16_to_32 = errors[1] / errors[2];

    assert!(
        (3.5..=4.5).contains(&ratio_8_to_16),
        "Expected O(h²) reduction factor ~4 between N=8 and N=16, got {:.2}",
        ratio_8_to_16
    );

    assert!(
        (3.5..=4.5).contains(&ratio_16_to_32),
        "Expected O(h²) reduction factor ~4 between N=16 and N=32, got {:.2}",
        ratio_16_to_32
    );
}
