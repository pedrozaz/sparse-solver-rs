use sparse_solver_rs::poisson::{assemble_poisson_2d, compute_l2_error, exact_poisson_solution};
use sparse_solver_rs::solvers::cg::ConjugateGradient;
use sparse_solver_rs::solvers::multigrid::MultigridSolver;
use sparse_solver_rs::solvers::SolverOptions;

fn main() {
    println!("=== 2D Poisson Equation Solver: CG vs Multigrid V-Cycle ===");
    println!(
        "{:>6} | {:>12} | {:>12} | {:>12} | {:>12}",
        "Grid N", "CG Iters", "MG Iters", "CG L2 Error", "MG L2 Error"
    );
    println!("{:-<64}", "");

    // Structured grids suitable for multigrid (odd N = 2^k - 1: 7, 15, 31, 63)
    let sizes = [7, 15, 31, 63];
    let mg_solver = MultigridSolver::new();

    for &n in &sizes {
        let (a, b) = assemble_poisson_2d(n);
        let u_exact = exact_poisson_solution(n);

        let options = SolverOptions {
            max_iter: 2000,
            tol: 1e-8,
        };

        let cg_result = ConjugateGradient::solve(&a, &b, &options);
        let cg_error = compute_l2_error(&cg_result.solution, &u_exact, n);

        let mg_result = mg_solver.solve(n, &b, &options);
        let mg_error = compute_l2_error(&mg_result.solution, &u_exact, n);

        println!(
            "{:>6} | {:>12} | {:>12} | {:>12.4e} | {:>12.4e}",
            n, cg_result.iterations, mg_result.iterations, cg_error, mg_error
        );
    }
}
