use sparse_solver_rs::poisson::{assemble_poisson_2d, compute_l2_error, exact_poisson_solution};
use sparse_solver_rs::solvers::SolverOptions;
use sparse_solver_rs::solvers::cg::ConjugateGradient;

fn main() {
    println!("=== 2D Poisson Equation Solver (Conjugate Gradient) ===");
    println!(
        "{:>6} | {:>10} | {:>12} | {:>12}",
        "Grid N", "Iterations", "Residual", "L2 Error"
    );
    println!("{:-<52}", "");

    let sizes = [8, 16, 32, 64];

    for &n in &sizes {
        let (a, b) = assemble_poisson_2d(n);
        let u_exact = exact_poisson_solution(n);

        let options = SolverOptions {
            max_iter: 2000,
            tol: 1e-8,
        };

        let result = ConjugateGradient::solve(&a, &b, &options);
        let error_l2 = compute_l2_error(&result.solution, &u_exact, n);

        println!(
            "{:>6} | {:>10} | {:>12.4e} | {:>12.4e}",
            n, result.iterations, result.residual_norm, error_l2
        );
    }
}
